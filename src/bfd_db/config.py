"""
Default pipeline parameters.

All parameters are exposed to users who call the toolchain directly;
the values below are the national-database defaults that will be locked
in for the published frozen snapshot.
"""

from dataclasses import dataclass, field
from pathlib import Path

# ── Repository layout ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # us_baseflow_db/
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS = ROOT / "data" / "outputs"

# ── Labeling defaults ──────────────────────────────────────────────────────────
# X%: a day is labeled BFD by a separation method if
#     baseflow_estimate / total_Q  >=  BF_RATIO_THRESHOLD
BF_RATIO_THRESHOLD: float = 0.95

# Magnitude-adaptive threshold (see docs/adaptive-bfd-threshold.md): the ratio
# test is relaxed to BF_RATIO_THRESHOLD_LOW at or below the gage's own
# LOW_FLOW_ANCHOR flow-duration quantile, held at BF_RATIO_THRESHOLD at or
# above HIGH_FLOW_ANCHOR, and log-interpolated between. None means "off" — a
# flat BF_RATIO_THRESHOLD applies to every day regardless of flow, which is
# the frozen-snapshot default for the national database build. A caller opts
# into the ramp by passing a PipelineConfig with this set (e.g. 0.75).
BF_RATIO_THRESHOLD_LOW: float | None = None
LOW_FLOW_ANCHOR: float = 0.10
HIGH_FLOW_ANCHOR: float = 0.50

# Flatness voter (see bfd_db.labeling.ensemble.flatness_vote): an additional
# vote from flow flatness alone, independent of any separation method's
# qb/Q ratio. Off by default (None): the ratio-based voters plus RF-BFD are
# the whole ensemble, matching pre-flatness-voter behavior exactly.
USE_FLATNESS_VOTE: bool = False
FLATNESS_WINDOW: int = 15
FLATNESS_CV_THRESHOLD: float = 0.05

# Fraction of methods that must vote BFD for the ensemble label to be 1.
# 0.5 = simple majority (≥2 of 3 methods).
VOTING_FRACTION: float = 0.5

# ── Event generation defaults ──────────────────────────────────────────────────
# t_c: maximum non-BFD gap (days) between two adjacent BFD runs that are
#      merged into a single event (inter-event time criterion).
T_C: int = 3

# d_min: minimum event duration (days) — shorter pooled events are discarded.
D_MIN: int = 7

# k: number of consecutive BFD days that must be observed before the causal
#    onset detector fires (look-ahead free).
K_ONSET: int = 5

# ── Data scope ─────────────────────────────────────────────────────────────────
# Date range for the national database build.
START_DATE: str = "1950-01-01"
END_DATE: str = "2024-12-31"   # update each snapshot


@dataclass
class PipelineConfig:
    """Collects all tunable parameters in one object for easy serialization."""
    bf_ratio_threshold: float = BF_RATIO_THRESHOLD
    bf_ratio_threshold_low: float | None = BF_RATIO_THRESHOLD_LOW
    low_flow_anchor: float = LOW_FLOW_ANCHOR
    high_flow_anchor: float = HIGH_FLOW_ANCHOR
    use_flatness_vote: bool = USE_FLATNESS_VOTE
    flatness_window: int = FLATNESS_WINDOW
    flatness_cv_threshold: float = FLATNESS_CV_THRESHOLD
    voting_fraction: float = VOTING_FRACTION
    t_c: int = T_C
    d_min: int = D_MIN
    k_onset: int = K_ONSET
    start_date: str = START_DATE
    end_date: str = END_DATE
    baseflowx_methods: list = field(default_factory=lambda: [
        "eckhardt", "chapman", "lyne_hollick", "boughton"
    ])
