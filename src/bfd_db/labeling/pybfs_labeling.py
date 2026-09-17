"""
BFD labeling via PyBFS (the non-linear state-space BFS model).

PyBFS plays two roles in this project:
  1. LABELING (this module): run PyBFS in separation mode to estimate Q_b(t);
     label a day as BFD if Q_b / Q_total >= bf_ratio_threshold.
  2. FORECASTING (Objective 3): run PyBFS in forecast mode from a causal onset.

Calibrated PyBFS parameters for 8,729 GAGES-II gages (from the LFA pipeline)
live at data/raw/pybfs/all_params.csv, with columns:
  tmp.site (int, no zero-padding), tmp.area, Lb, X1, Wb, POR, ALPHA, BETA,
  Ks, Kb, Kz, Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise, Error, BFF
data/raw/pybfs/calibration_failed.csv lists site_no values where LFA's
calibration did not converge (no usable parameters).

This mirrors pybfs's own get_values_for_site() column groups (AREA, Lb, X1,
Wb, POR | ALPHA, BETA, Ks, Kb, Kz | Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise)
but does the per-site lookup directly instead of calling that function,
since it rebuilds a dict over the full DataFrame on every call — too slow
to call once per gage across a ~9,000-gage batch build.

pybfs operates in SI units (m³/day), while this project's streamflow is in
cfs (ft³/s); run_pybfs_separation() converts both ways.

See: https://github.com/BYU-Hydroinformatics/pybfs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pybfs import base_table, bfs

from bfd_db.config import DATA_RAW, BF_RATIO_THRESHOLD

PYBFS_PARAMS_PATH = DATA_RAW / "pybfs" / "all_params.csv"

_BASIN_CHAR_COLS = ["AREA", "Lb", "X1", "Wb", "POR"]
_GW_HYD_COLS = ["ALPHA", "BETA", "Ks", "Kb", "Kz"]
_FLOW_COLS = ["Qthresh", "Rs", "Rb1", "Rb2", "Prec", "Frac4Rise"]

# 1 ft^3/s -> m^3/day (0.0283168466 m^3/ft^3 * 86400 s/day)
_CFS_TO_M3DAY = 2446.575

_params_df: pd.DataFrame | None = None


def _load_params_table() -> pd.DataFrame:
    """Load and cache the full calibrated-parameters table (module-level)."""
    global _params_df
    if _params_df is None:
        df = pd.read_csv(PYBFS_PARAMS_PATH)
        df = df.rename(columns={"tmp.site": "site_no", "tmp.area": "AREA"})
        df["STAID"] = df["site_no"].astype(str).str.zfill(8)
        df = df.set_index("STAID")
        _params_df = df
    return _params_df


def load_pybfs_params(site_no: str) -> dict | None:
    """Load the calibrated PyBFS parameter vectors for a single gage.

    Returns None if the site has no calibrated parameters (not all GAGES-II
    gages converged during LFA calibration; skip uncalibrated gages).
    """
    params_df = _load_params_table()
    if site_no not in params_df.index:
        return None
    row = params_df.loc[site_no]
    return {
        "basin_char": row[_BASIN_CHAR_COLS].tolist(),
        "gw_hyd": row[_GW_HYD_COLS].tolist(),
        "flow": row[_FLOW_COLS].tolist(),
    }


def run_pybfs_separation(
    q: pd.Series,
    params: dict,
) -> pd.Series:
    """Run PyBFS in separation mode to estimate baseflow Q_b(t).

    Parameters
    ----------
    q : pd.Series
        Daily streamflow in ft³/s, DatetimeIndex.
    params : dict
        Calibrated parameter dict from load_pybfs_params() — keys
        'basin_char' [AREA, Lb, X1, Wb, POR], 'gw_hyd' [ALPHA, BETA, Ks, Kb,
        Kz], 'flow' [Qthresh, Rs, Rb1, Rb2, Prec, Frac4Rise].

    Returns
    -------
    pd.Series
        Estimated baseflow Q_b in ft³/s, same index as q.
    """
    basin_char = params["basin_char"]
    gw_hyd = params["gw_hyd"]
    flow = params["flow"]
    _, lb, x1, wb, por = basin_char
    _, beta, _, kb, _ = gw_hyd

    streamflow_df = pd.DataFrame({
        "Date": q.index,
        "Streamflow": q.to_numpy(dtype=float) * _CFS_TO_M3DAY,
    })

    sbt = base_table(lb, x1, wb, beta, kb, streamflow_df, por)
    result = bfs(streamflow_df, sbt, basin_char, gw_hyd, flow)

    q_b = pd.Series(
        result["Baseflow"].to_numpy() / _CFS_TO_M3DAY,
        index=pd.DatetimeIndex(result["Date"]),
        name="qb_pybfs",
    )
    return q_b.reindex(q.index)


def label_gage(
    site_no: str,
    q: pd.Series,
    bf_ratio_threshold: float = BF_RATIO_THRESHOLD,
) -> pd.Series | None:
    """Return a daily 0/1 BFD label series using PyBFS for one gage.

    Returns None if no calibrated parameters exist for this gage, or if
    PyBFS fails to converge on this streamflow series.
    """
    params = load_pybfs_params(site_no)
    if params is None:
        return None

    try:
        q_b = run_pybfs_separation(q, params)
    except Exception:
        return None

    ratio = q_b / q.replace(0, np.nan)
    labels = (ratio >= bf_ratio_threshold).astype(int).fillna(0)
    labels.name = "pybfs"
    return labels
