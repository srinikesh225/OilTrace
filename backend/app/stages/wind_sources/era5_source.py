"""
ERA5WindSource — real ERA5 10-m wind for a Sentinel-1 scene, via cdsapi.

Retrieves `10m_u_component_of_wind` and `10m_v_component_of_wind` from
`reanalysis-era5-single-levels` for the scene's bounding box, hourly, over the
drift window `scene_time - SLICK_AGE_HOURS .. scene_time` (SLICK_AGE_HOURS from
config). Wind speed and direction are computed explicitly from u/v and returned
in the SAME list-of-dicts shape the pipeline already consumes:

    [{"time": "...Z", "speed_ms": float, "dir_deg": float (meteorological FROM),
      "u_ms": float, "v_ms": float}, ...]

Conventions (documented, verified):
  * u = eastward component (m/s), v = northward component (m/s)
  * speed = sqrt(u^2 + v^2)  [m/s]
  * dir_deg = direction the wind blows FROM (meteorological), degrees, 0=N 90=E
              dir_from = degrees(atan2(-u, -v)) mod 360
    This matches the fixture's dir_deg semantics (rewind.py steps upwind toward
    dir_deg), so the same gate/drift consume ERA5 and fixture wind identically.

Robustness: credentials come from the standard cdsapi mechanism (~/.cdsapirc or
CDSAPI_URL / CDSAPI_KEY env) — never hardcoded. On ANY failure (missing
credentials, unreachable API, corrupt/invalid NetCDF) this logs a warning and
FALLS BACK to FileWindSource, recording source="FILE_FALLBACK" with the reason.
It never silently passes fixture wind off as ERA5. Downloads are cached under
config.ERA5_CACHE_DIR (git-ignored); a repeat request is a CACHE HIT with no
network call. cdsapi/xarray are imported lazily so file mode never needs them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone

from . import WindSource
from .file_source import FileWindSource
from ... import config

logger = logging.getLogger("oiltrace.wind")

# ERA5 NetCDF variable/dim aliases (vary by CDS backend / conversion).
_U_NAMES = ("u10", "10u", "u10m", "si10u")
_V_NAMES = ("v10", "10v", "v10m", "si10v")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_TIME_NAMES = ("valid_time", "time")


def _parse_utc(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class ERA5WindSource(WindSource):
    name = "era5"

    def __init__(self, cache_dir: str = None):
        super().__init__()
        self.cache_dir = cache_dir or config.ERA5_CACHE_DIR

    # -- public -------------------------------------------------------------
    def fetch(self, scene, wind_path) -> list[dict]:
        try:
            return self._fetch_era5(scene)
        except Exception as e:
            # Visible fallback — NEVER present fixture wind as ERA5.
            logger.warning(
                "ERA5 unavailable (%s: %s) — FALLING BACK to FileWindSource. "
                "Wind is fixture data, NOT ERA5.", type(e).__name__, e)
            fb = FileWindSource()
            samples = fb.fetch(scene, wind_path)
            self.provenance = {
                "source": "FILE_FALLBACK",
                "reason": f"{type(e).__name__}: {e}",
                "fallback_file": str(wind_path),
                "cache": "n/a",
                "n_samples": len(samples),
            }
            return samples

    # -- internals ----------------------------------------------------------
    def _window_hours(self, scene_time: str):
        """Whole UTC hours spanning scene_time - SLICK_AGE_HOURS .. scene_time,
        bracketed one hour past the scene so a sub-hour acquisition (e.g. 05:31)
        is enclosed by both neighbouring ERA5 hours."""
        st = _parse_utc(scene_time)
        base = st.replace(minute=0, second=0, microsecond=0)
        start = base - timedelta(hours=config.SLICK_AGE_HOURS)
        end = base + timedelta(hours=1)
        hours, t = [], start
        while t <= end:
            hours.append(t)
            t += timedelta(hours=1)
        return hours

    # Pad the requested box so it always spans several ERA5 grid cells. ERA5 is
    # on a 0.25 deg grid; a scene smaller than one cell (e.g. a single SAR tile)
    # makes MARS fail with an empty-area-crop assertion. Padding guarantees at
    # least one grid point; extraction still takes the point nearest the scene
    # centre, so the result is unchanged for large scenes.
    AREA_PAD_DEG = 0.5

    def _area(self, scene):
        b = scene.bounds
        p = self.AREA_PAD_DEG
        # CDS area order is [North, West, South, East]; round to keep cache keys
        # and requests grid-stable.
        return [round(b.max_lat + p, 2), round(b.min_lon - p, 2),
                round(b.min_lat - p, 2), round(b.max_lon + p, 2)]

    def _cache_key(self, area, hours, variables) -> str:
        payload = json.dumps({
            "dataset": config.ERA5_DATASET,
            "area_NWSE": area,
            "hours": [h.strftime("%Y-%m-%dT%H") for h in hours],
            "variables": sorted(variables),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _fetch_era5(self, scene) -> list[dict]:
        area = self._area(scene)
        hours = self._window_hours(scene.acquired_at)
        variables = list(config.ERA5_VARIABLES)
        key = self._cache_key(area, hours, variables)
        os.makedirs(self.cache_dir, exist_ok=True)
        nc = os.path.join(self.cache_dir, f"era5_{key}.nc")

        cache = "MISS"
        if self._valid_cache(nc, hours):
            cache = "HIT"
            logger.info("ERA5 CACHE HIT -> %s (no network)", nc)
        else:
            if os.path.exists(nc):
                logger.warning("ERA5 cache invalid/corrupt -> discarding %s", nc)
                os.remove(nc)
            logger.info("ERA5 CACHE MISS -> downloading %s", nc)
            self._download(nc, area, hours, variables)   # raises if no creds/net
            if not self._valid_cache(nc, hours):
                raise RuntimeError("downloaded ERA5 NetCDF failed validation")

        samples, centre = self._extract(nc, scene)
        # CDS builds the cartesian product of day x time, so a cross-midnight
        # window returns hours outside it. Keep only the drift window
        # [scene_time - SLICK_AGE_HOURS .. bracket], inclusive.
        lo = min(hours).replace(tzinfo=timezone.utc)
        hi = max(hours).replace(tzinfo=timezone.utc)
        samples = [s for s in samples if lo <= _parse_utc(s["time"]) <= hi]
        self.provenance = {
            "source": "ERA5",
            "dataset": config.ERA5_DATASET,
            "variables": variables,
            "area_NWSE": area,
            "spatial_method": "nearest grid point to scene centre",
            "direction_convention": "meteorological FROM (0=N,90=E)",
            "units": "m/s",
            "cache": cache,
            "cache_file": nc,
            "centre": centre,
            "n_samples": len(samples),
        }
        return samples

    def _download(self, nc, area, hours, variables):
        import cdsapi  # lazy; only needed for a real download
        client = cdsapi.Client()   # reads standard creds; raises if missing
        req = {
            "product_type": ["reanalysis"],
            "variable": variables,
            "year": sorted({h.strftime("%Y") for h in hours}),
            "month": sorted({h.strftime("%m") for h in hours}),
            "day": sorted({h.strftime("%d") for h in hours}),
            "time": sorted({h.strftime("%H:00") for h in hours}),
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        client.retrieve(config.ERA5_DATASET, req, nc)

    def _open(self, nc):
        import xarray as xr  # lazy
        return xr.open_dataset(nc)

    @staticmethod
    def _pick(names, avail):
        for n in names:
            if n in avail:
                return n
        return None

    def _valid_cache(self, nc, hours) -> bool:
        if not (os.path.isfile(nc) and os.path.getsize(nc) > 0):
            return False
        try:
            ds = self._open(nc)
        except Exception as e:
            logger.warning("ERA5 cache unreadable (%s): %s", type(e).__name__, e)
            return False
        try:
            names = set(ds.variables)
            u = self._pick(_U_NAMES, names)
            v = self._pick(_V_NAMES, names)
            lat = self._pick(_LAT_NAMES, names)
            lon = self._pick(_LON_NAMES, names)
            tname = self._pick(_TIME_NAMES, set(ds.dims) | names)
            if not all([u, v, lat, lon, tname]):
                return False
            # requested time range must be covered by the file's time axis
            import numpy as np
            tvals = [_parse_utc(str(np.datetime_as_string(t, unit="s")))
                     for t in ds[tname].values]
            if not tvals:
                return False
            need_lo = min(hours).replace(tzinfo=timezone.utc)
            need_hi = max(hours).replace(tzinfo=timezone.utc)
            return min(tvals) <= need_lo and max(tvals) >= need_hi
        except Exception as e:
            logger.warning("ERA5 cache validation error (%s): %s", type(e).__name__, e)
            return False
        finally:
            ds.close()

    def _extract(self, nc, scene):
        import numpy as np
        ds = self._open(nc)
        try:
            names = set(ds.variables)
            un = self._pick(_U_NAMES, names)
            vn = self._pick(_V_NAMES, names)
            latn = self._pick(_LAT_NAMES, names)
            lonn = self._pick(_LON_NAMES, names)
            tname = self._pick(_TIME_NAMES, set(ds.dims) | names)

            clat = (scene.bounds.min_lat + scene.bounds.max_lat) / 2.0
            clon = (scene.bounds.min_lon + scene.bounds.max_lon) / 2.0
            sub = ds.sel({latn: clat, lonn: clon}, method="nearest")
            glat, glon = float(sub[latn]), float(sub[lonn])

            samples = []
            for t in ds[tname].values:
                u = float(sub[un].sel({tname: t}).values)
                v = float(sub[vn].sel({tname: t}).values)
                speed = math.sqrt(u * u + v * v)              # m/s, raw u/v
                dir_from = math.degrees(math.atan2(-u, -v)) % 360.0
                secs = t.astype("datetime64[s]").astype("int64")
                iso = datetime.fromtimestamp(int(secs), tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")
                samples.append({
                    "time": iso,
                    "speed_ms": round(speed, 2),
                    "dir_deg": round(dir_from, 1),
                    "u_ms": round(u, 4),
                    "v_ms": round(v, 4),
                })
            samples.sort(key=lambda s: s["time"])
            centre = {"scene_centre_lat": round(clat, 4), "scene_centre_lon": round(clon, 4),
                      "grid_lat": round(glat, 4), "grid_lon": round(glon, 4)}
            return samples, centre
        finally:
            ds.close()
