"""
Stage 2 — FILTER: the wind gate.

Oil only forms a readable slick in a middle band of wind. Too calm and the sea
is glassy, so dark patches are not necessarily oil. Too rough and any slick is
churned away within hours. So we look up the wind at the scene time and reject
any polygon when the wind is outside [WIND_MIN_MS, WIND_MAX_MS].

Nothing is ever silently dropped: a rejected polygon carries a human-readable
reason through to the API response.

# PROTOTYPE: the gate reads a single wind sample at the scene time and applies
# it to every polygon. The real gate uses the wind history over the drift
# window and a spatial wind field, not one scalar.
"""

from __future__ import annotations

from datetime import datetime

from .. import config
from ..models import FilterOutput, Rejection, SeeOutput, WindSample


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def wind_at(scene_time: str, wind_samples: list[dict]) -> WindSample:
    """Nearest hourly wind sample to the scene acquisition time."""
    target = _parse(scene_time)
    nearest = min(wind_samples, key=lambda w: abs((_parse(w["time"]) - target).total_seconds()))
    return WindSample(time=nearest["time"], speed_ms=nearest["speed_ms"], dir_deg=nearest["dir_deg"])


def apply_wind_gate(see: SeeOutput, scene_time: str, wind_samples: list[dict]) -> FilterOutput:
    wind = wind_at(scene_time, wind_samples)

    accepted = []
    rejected: list[Rejection] = []

    for poly in see.polygons:
        if wind.speed_ms < config.WIND_MIN_MS:
            rejected.append(Rejection(
                polygon_id=poly.polygon_id,
                stage="filter",
                reason=(f"Wind {wind.speed_ms:.1f} m/s is below the "
                        f"{config.WIND_MIN_MS:.1f} m/s minimum — sea too calm, "
                        f"a dark patch here is not reliably oil."),
            ))
        elif wind.speed_ms > config.WIND_MAX_MS:
            rejected.append(Rejection(
                polygon_id=poly.polygon_id,
                stage="filter",
                reason=(f"Wind {wind.speed_ms:.1f} m/s is above the "
                        f"{config.WIND_MAX_MS:.1f} m/s maximum — any slick would "
                        f"have been mixed away."),
            ))
        else:
            accepted.append(poly)

    return FilterOutput(accepted=accepted, rejected=rejected, wind_at_scene=wind)
