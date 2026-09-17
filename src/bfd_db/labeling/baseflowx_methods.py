"""
BFD labeling via standard baseflow separation algorithms (baseflowx package).

For each digital-filter method, we:
  1. Estimate the baseflow component Q_b(t) using the algorithm.
  2. Label a day as BFD if  Q_b / Q_total >= bf_ratio_threshold  (i.e., the
     baseflow component accounts for ≥X% of observed streamflow).

Methods (real baseflowx 0.2.x function names):
  'eckhardt'     — Eckhardt recursive digital filter (recommended default)
  'chapman'      — Chapman filter
  'lyne_hollick' — Lyne-Hollick filter (baseflowx.lh; original and most common)
  'boughton'     — Boughton two-parameter filter

The recession coefficient `a` is estimated per gage from the hydrograph
(strict_baseflow + recession_coefficient), following the package's own
"Parameter Estimation" recipe. Eckhardt's BFImax is estimated via
maxmium_BFI(); Boughton's C is calibrated via param_calibrate() over a
fixed search range (typical daily-timestep range, Boughton 1993).

See: https://pypi.org/project/baseflowx/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import baseflowx as bfx

from bfd_db.config import BF_RATIO_THRESHOLD

# Search range for calibrating Boughton's C via bfx.param_calibrate().
_BOUGHTON_C_RANGE = np.linspace(0.001, 1.0, 100)


def _boughton_for_calibration(Q, b_LH, a, C, return_exceed=False):
    """Adapter matching baseflowx.param_calibrate's method(Q, b_LH, a, p, ...) call convention."""
    return bfx.boughton(Q, a, C, return_exceed=return_exceed)


def separate_baseflow(q: pd.Series, method: str = "eckhardt") -> pd.Series:
    """Run one baseflowx algorithm and return the estimated baseflow series.

    Parameters
    ----------
    q : pd.Series
        Daily streamflow in ft³/s.
    method : str
        One of 'eckhardt', 'chapman', 'lyne_hollick', 'boughton'.

    Returns
    -------
    pd.Series
        Estimated baseflow Q_b, same index as q.
    """
    Q = q.to_numpy(dtype=float)
    strict = bfx.strict_baseflow(Q)
    a = bfx.recession_coefficient(Q, strict)
    b_lh = bfx.lh(Q)

    if method == "eckhardt":
        bfimax = bfx.maxmium_BFI(Q, b_lh, a)
        q_b = bfx.eckhardt(Q, a, bfimax)
    elif method == "chapman":
        q_b = bfx.chapman(Q, a)
    elif method == "lyne_hollick":
        q_b = b_lh
    elif method == "boughton":
        c = bfx.param_calibrate(_BOUGHTON_C_RANGE, _boughton_for_calibration, Q, b_lh, a)
        q_b = bfx.boughton(Q, a, c)
    else:
        raise ValueError(f"Unknown baseflowx method: {method!r}")

    return pd.Series(q_b, index=q.index, name=f"qb_{method}")


def label_from_separation(
    q: pd.Series,
    q_b: pd.Series,
    bf_ratio_threshold: float = BF_RATIO_THRESHOLD,
) -> pd.Series:
    """Convert baseflow estimates to 0/1 BFD labels using the X% criterion.

    A day is BFD if  Q_b / Q >= bf_ratio_threshold  (e.g., 0.95 = 95%).
    Days where Q == 0 are labeled 0 (no baseflow dominance without flow).
    """
    ratio = q_b / q.replace(0, np.nan)
    labels = (ratio >= bf_ratio_threshold).astype(int)
    labels = labels.fillna(0)
    return labels


def label_gage(
    q: pd.Series,
    methods: list[str] | None = None,
    bf_ratio_threshold: float = BF_RATIO_THRESHOLD,
) -> pd.DataFrame:
    """Apply multiple baseflowx methods to one gage, return per-method labels.

    Returns a DataFrame with one boolean column per method (0/1 BFD label),
    plus a 'q_b_<method>' column for the raw baseflow estimate.

    Parameters
    ----------
    q : pd.Series
        Daily streamflow, DatetimeIndex.
    methods : list[str]
        Methods to run; defaults to ['eckhardt', 'chapman', 'lyne_hollick'].
    bf_ratio_threshold : float
        X% threshold (fraction, not percent).
    """
    if methods is None:
        methods = ["eckhardt", "chapman", "lyne_hollick"]

    out = pd.DataFrame(index=q.index)
    for method in methods:
        q_b = separate_baseflow(q, method)
        out[f"qb_{method}"] = q_b
        out[f"bfx_{method}"] = label_from_separation(q, q_b, bf_ratio_threshold)

    return out
