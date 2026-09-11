"""
OILTRACE central configuration.

Every tunable number the pipeline uses lives here so that the physics
assumptions are auditable in one place. Nothing else in the codebase should
hard-code these values.
"""

from pathlib import Path

# --- Physical / pipeline tunables -------------------------------------------

SLICK_AGE_HOURS = 12          # single fixed timeframe, not a range
WIND_MIN_MS = 3.0             # below this the sea is flat, reject
WIND_MAX_MS = 12.0            # above this the slick mixes away
WIND_DRIFT_FACTOR = 0.03      # slick moves at 3% of wind speed
N_PARTICLES = 500             # particles scattered for the backward drift
REGION_BOUNDS = (54.5, 10.0, 57.5, 13.5)   # (min_lat, min_lon, max_lat, max_lon) Danish waters

# --- Determinism ------------------------------------------------------------

RANDOM_SEED = 42             # fixed so the same scene always gives the same result

# --- Detector (see.py) ------------------------------------------------------

# Dark-pixel threshold on the 0-255 greyscale radar image. Sea is bright,
# oil is dark, so we keep pixels darker than this.
DARK_THRESHOLD = 90
# Minimum contour area in pixels to be considered a real slick candidate.
MIN_SLICK_AREA_PX = 1500

# --- Ship matching (name.py) ------------------------------------------------

# How wide a window (hours) around the estimated release time counts as a
# temporal match for a vessel position.
MATCH_TIME_WINDOW_HOURS = 3.0
# A gap between consecutive AIS reports longer than this (minutes) is flagged
# as a potential "dark vessel" event.
AIS_GAP_FLAG_MINUTES = 30.0

# --- Backward drift (rewind.py) ---------------------------------------------

# Per-step random jitter added to each particle, standard deviation in metres.
# Spreads the ensemble so the release area has body. Illustrative, not tuned.
DRIFT_JITTER_SIGMA_M = 120.0

# --- Scoring (explain.py) ---------------------------------------------------
# Weights sum to 1.0. Each component is reported separately in the bundle.
# NOTE: these weights and the reference scale below are hand-set placeholders,
# not a calibrated model. See the PROTOTYPE note in explain.py.

SCORE_WEIGHT_TIME = 0.4       # temporal proximity to release time
SCORE_WEIGHT_SPATIAL = 0.3    # how deep inside the release polygon
SCORE_WEIGHT_COURSE = 0.3     # heading agreement with the slick long axis

# Depth (km) inside the release polygon at which the spatial score saturates
# to 1.0. Arbitrary reference scale, kept here rather than buried in the stage.
SPATIAL_FULL_CREDIT_KM = 3.0

# Decimal places for reported scores. Kept low on purpose: the scoring model is
# a heuristic and three-decimal precision would be false confidence.
SCORE_DECIMALS = 2

# --- Candidate separation (explain.py) --------------------------------------
# When the gap between the top two candidates' total scores is smaller than
# this, the evidence does not justify naming a single suspect: the two are
# returned as joint candidates instead. This is a presentation/interpretation
# rule only — it never changes any score. Triggers strictly on gap < MIN, so a
# gap exactly equal to MIN does NOT trigger.
CANDIDATE_SEPARATION_MIN = 0.15

# --- Geodesy ----------------------------------------------------------------
# Simple spherical constants shared by every stage that converts between
# metres/kilometres and degrees. Centralised so no stage carries its own copy.
METERS_PER_DEG_LAT = 111_320.0
KM_PER_DEG_LAT = 111.32

# --- Data provenance --------------------------------------------------------
# This build runs on a generated fixture, and the "culprit" vessel is planted
# by inverting the drift model. Surface that in every response so nobody
# mistakes a demo run for a real detection.
DATA_SOURCE = "SYNTHETIC_FIXTURE"
PROVENANCE_NOTE = (
    "Scene and all vessel tracks are generated sample data "
    "(scripts/make_sample_data.py). The top-ranked vessel is planted by "
    "inverting the drift model, so results demonstrate pipeline mechanics, "
    "not a real-world detection."
)

# --- Scenes / paths ---------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = BACKEND_DIR / "sample_data"

# The bundled demonstration scenes. Each is a self-contained fixture directory
# under SAMPLE_DATA_DIR ("scene_<name>/") holding scene.tif, wind.json,
# ships.json, and scene_meta.json.
SCENE_NAMES = ("normal", "ambiguous", "calm")
DEFAULT_SCENE = "normal"


def scene_paths(scene: str) -> dict:
    """Resolve the four fixture files for a named scene.

    Only the known scene names are accepted, so a request can never reach an
    arbitrary filesystem path.
    """
    if scene not in SCENE_NAMES:
        raise ValueError(
            f"Unknown scene '{scene}'. Valid scenes: {', '.join(SCENE_NAMES)}."
        )
    d = SAMPLE_DATA_DIR / f"scene_{scene}"
    return {
        "dir": d,
        "tif": d / "scene.tif",
        "wind": d / "wind.json",
        "ships": d / "ships.json",
        "meta": d / "scene_meta.json",
    }


# The disclaimer that must ride along on every response, verbatim.
DISCLAIMER = "Detection is not attribution. Attribution is not proof."
