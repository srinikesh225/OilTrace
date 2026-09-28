"""
Wind sources for the FILTER (and REWIND) stages.

    WindSource interface   fetch(scene, wind_path) -> list[wind samples]
       |
       +-- FileWindSource   the original bundled wind.json (default, offline)
       +-- ERA5WindSource   real ERA5 10-m wind via cdsapi (opt-in)

The pipeline consumes a list of hourly wind samples exactly as it always has —
`[{"time", "speed_ms", "dir_deg"}, ...]`. Neither filter.py nor rewind.py knows
or cares where the samples came from; only main.py picks the source, via
config.WIND_SOURCE ("file" | "era5"), defaulting to "file".

ERA5WindSource never crashes the app: if credentials are missing, the API is
unreachable, or the cache is corrupt, it FALLS BACK to FileWindSource and marks
that plainly in its provenance (source == "FILE_FALLBACK") — it never presents
fixture wind as ERA5. The ERA5 backend and its dependencies (cdsapi, xarray) are
imported lazily, so the default file path needs neither installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ... import config

logger = logging.getLogger("oiltrace.wind")


class WindSource(ABC):
    """Provides the hourly wind samples the pipeline gates and drifts on.

    fetch() returns the same list-of-dicts the bundled wind.json always used,
    and records self.provenance (a plain dict) describing where the wind came
    from — read by main.py for logging, never injected into the scored response
    (so the fixture path stays byte-identical)."""

    name = "abstract"

    def __init__(self) -> None:
        self.provenance: dict = {}

    @abstractmethod
    def fetch(self, scene, wind_path) -> list[dict]:
        ...


# Import the concrete file source here (cheap, no heavy deps).
from .file_source import FileWindSource  # noqa: E402


def make_wind_source() -> WindSource:
    """Return the configured wind source. Defaults to FileWindSource. An unknown
    value falls back to file with a warning. ERA5 is only constructed when
    explicitly requested (config.WIND_SOURCE == 'era5')."""
    backend = (config.WIND_SOURCE or "file").strip().lower()

    if backend == "era5":
        # Lazy import: importing cdsapi/xarray must not be required for file mode.
        from .era5_source import ERA5WindSource
        logger.info("Wind source: era5 (falls back to file on any failure)")
        return ERA5WindSource()

    if backend != "file":
        logger.warning("Unknown WIND_SOURCE=%r; using file.", backend)
    logger.info("Wind source: file")
    return FileWindSource()


__all__ = ["WindSource", "FileWindSource", "make_wind_source"]
