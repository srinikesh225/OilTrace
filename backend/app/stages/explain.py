"""
Stage 5 — EXPLAIN: score, rank, and assemble the evidence bundle.

Each candidate gets a 0-1 score built from three weighted, separately reported
parts:
  * time    — how close the vessel's matched report is to the release time
  * spatial — how deep inside the release polygon the vessel was
  * course  — how well the vessel's heading matches the slick's long axis

An AIS reporting gap is recorded as a FLAG, never a score bonus: a dark-vessel
event is context for an analyst, not evidence of guilt on its own.

The evidence bundle captures everything needed to reconstruct the run: scene
identity, the config values used, the random seed, every stage's output, and
the full per-candidate score breakdown.

# PROTOTYPE: the score is an illustrative linear heuristic with hand-set
# weights (config.SCORE_WEIGHT_*) and an arbitrary spatial reference scale
# (config.SPATIAL_FULL_CREDIT_KM). It is NOT a validated attribution model and
# the numbers carry no calibrated probability — hence the low SCORE_DECIMALS.
"""

from __future__ import annotations

from .. import config
from ..models import (CandidateMatch, ExplainOutput, FilterOutput, NameOutput,
                      RewindOutput, ScoreBreakdown, ScoredCandidate, SeeOutput,
                      SceneMeta)


def _time_score(time_delta_min: float) -> float:
    window_min = config.MATCH_TIME_WINDOW_HOURS * 60.0
    return max(0.0, 1.0 - (time_delta_min / window_min))


def _spatial_score(dist_inside_km: float) -> float:
    # Fully credited at >= SPATIAL_FULL_CREDIT_KM inside the release polygon.
    return max(0.0, min(1.0, dist_inside_km / config.SPATIAL_FULL_CREDIT_KM))


def _course_score(heading_deg: float, slick_bearing_deg: float) -> float:
    # The slick bearing is an axis (0-180); a heading and its reverse both align.
    diff = abs((heading_deg % 180.0) - (slick_bearing_deg % 180.0))
    diff = min(diff, 180.0 - diff)  # fold into 0-90
    return max(0.0, 1.0 - (diff / 90.0))


def explain(name_out: NameOutput, rewind_out: RewindOutput, scene: SceneMeta,
            see_out: SeeOutput, filter_out: FilterOutput
            ) -> tuple[ExplainOutput, dict]:

    scored: list[ScoredCandidate] = []
    breakdowns = []  # for the evidence bundle

    d = config.SCORE_DECIMALS
    for c in name_out.candidates:
        ts = round(_time_score(c.time_delta_min), d)
        ss = round(_spatial_score(c.dist_inside_km), d)
        cs = round(_course_score(c.heading_deg, rewind_out.slick_bearing_deg), d)
        total = round(
            config.SCORE_WEIGHT_TIME * ts
            + config.SCORE_WEIGHT_SPATIAL * ss
            + config.SCORE_WEIGHT_COURSE * cs,
            d,
        )
        breakdown = ScoreBreakdown(time_score=ts, spatial_score=ss,
                                   course_score=cs, total=total)
        gap_note = None
        if c.has_ais_gap:
            gap_note = (f"AIS reporting gap of {c.max_gap_min:.0f} min "
                        f"(> {config.AIS_GAP_FLAG_MINUTES:.0f} min): possible "
                        f"dark-vessel behaviour. Flag only — not scored.")
        scored.append(ScoredCandidate(
            rank=0,  # filled after sorting
            mmsi=c.mmsi,
            name=c.name,
            score=breakdown,
            matched_position=c.matched_position,
            heading_deg=c.heading_deg,
            has_ais_gap=c.has_ais_gap,
            ais_gap_note=gap_note,
        ))

    # Rank by total descending; break ties deterministically by MMSI so the
    # order never depends on input file ordering.
    scored.sort(key=lambda s: (-s.score.total, s.mmsi))
    for rank, s in enumerate(scored, start=1):
        s.rank = rank
        breakdowns.append({
            "rank": rank,
            "mmsi": s.mmsi,
            "name": s.name,
            "weights": {
                "time": config.SCORE_WEIGHT_TIME,
                "spatial": config.SCORE_WEIGHT_SPATIAL,
                "course": config.SCORE_WEIGHT_COURSE,
            },
            "components": s.score.model_dump(),
            "has_ais_gap": s.has_ais_gap,
            "ais_gap_note": s.ais_gap_note,
        })

    # --- Candidate separation rule ------------------------------------------
    # Interpretation only: if the top two totals are closer than
    # CANDIDATE_SEPARATION_MIN, the evidence does not justify naming one
    # suspect. We flag it and mark both as joint candidates. Scores are read,
    # never modified. Triggers strictly on gap < MIN (equal does not trigger).
    separation_flag = None
    joint_candidates: list[str] = []
    score_gap = None
    if len(scored) >= 2:
        score_gap = round(scored[0].score.total - scored[1].score.total,
                          config.SCORE_DECIMALS)
        if score_gap < config.CANDIDATE_SEPARATION_MIN:
            separation_flag = "scores not separated - insufficient evidence to rank"
            scored[0].joint = True
            scored[1].joint = True
            joint_candidates = [scored[0].mmsi, scored[1].mmsi]

    evidence_id = scene.scene_id

    evidence_bundle = {
        "evidence_id": evidence_id,
        "scene_id": scene.scene_id,
        "scene_timestamp": scene.acquired_at,
        "data_source": config.DATA_SOURCE,
        "provenance_note": config.PROVENANCE_NOTE,
        "score_model": ("Illustrative linear heuristic (PROTOTYPE). Weighted sum "
                        "of time/spatial/course components; not calibrated, no "
                        "probabilistic meaning."),
        "random_seed": config.RANDOM_SEED,
        "config_used": {
            "SLICK_AGE_HOURS": config.SLICK_AGE_HOURS,
            "WIND_MIN_MS": config.WIND_MIN_MS,
            "WIND_MAX_MS": config.WIND_MAX_MS,
            "WIND_DRIFT_FACTOR": config.WIND_DRIFT_FACTOR,
            "N_PARTICLES": config.N_PARTICLES,
            "DRIFT_JITTER_SIGMA_M": config.DRIFT_JITTER_SIGMA_M,
            "REGION_BOUNDS": list(config.REGION_BOUNDS),
            "DARK_THRESHOLD": config.DARK_THRESHOLD,
            "MIN_SLICK_AREA_PX": config.MIN_SLICK_AREA_PX,
            "MATCH_TIME_WINDOW_HOURS": config.MATCH_TIME_WINDOW_HOURS,
            "AIS_GAP_FLAG_MINUTES": config.AIS_GAP_FLAG_MINUTES,
            "SCORE_WEIGHT_TIME": config.SCORE_WEIGHT_TIME,
            "SCORE_WEIGHT_SPATIAL": config.SCORE_WEIGHT_SPATIAL,
            "SCORE_WEIGHT_COURSE": config.SCORE_WEIGHT_COURSE,
            "SPATIAL_FULL_CREDIT_KM": config.SPATIAL_FULL_CREDIT_KM,
            "SCORE_DECIMALS": config.SCORE_DECIMALS,
            "CANDIDATE_SEPARATION_MIN": config.CANDIDATE_SEPARATION_MIN,
        },
        "stage_outputs": {
            "see": see_out.model_dump(),
            "filter": filter_out.model_dump(),
            "rewind": rewind_out.model_dump(),
            "name": name_out.model_dump(),
        },
        "score_breakdown": breakdowns,
        "candidate_separation": {
            "min": config.CANDIDATE_SEPARATION_MIN,
            "gap": score_gap,
            "separation_flag": separation_flag,
            "joint_candidates": joint_candidates,
        },
        "disclaimer": config.DISCLAIMER,
    }

    explain_out = ExplainOutput(
        scene_id=scene.scene_id,
        ranked_candidates=scored,
        separation_flag=separation_flag,
        joint_candidates=joint_candidates,
        score_gap=score_gap,
        evidence_id=evidence_id,
        disclaimer=config.DISCLAIMER,
    )
    return explain_out, evidence_bundle
