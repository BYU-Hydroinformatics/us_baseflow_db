"""
Per-event summary statistics.

For each BFD event (a row in the events table produced by pooling.py),
compute and attach the flow statistics needed by the database and the papers.

Stored per event:
    mean_Q      — mean daily flow during the event window (ft³/s)
    std_Q       — standard deviation of daily flow
    min_Q       — minimum daily flow
    max_Q       — maximum daily flow
    cv_Q        — coefficient of variation (std/mean)
    fdc_pct     — median percentile of event flow on the gage's full FDC
                  (0 = highest flows, 100 = lowest flows, like exceedance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flow_duration_percentile(q_event: np.ndarray, q_full: np.ndarray) -> float:
    """Return the median exceedance percentile of event flows on the full FDC.

    Exceedance percentile: fraction of all days with flow >= this value.
    A value near 100 means the event is near the low-flow end of the FDC.
    """
    if len(q_event) == 0 or len(q_full) == 0:
        return np.nan
    percentiles = [
        100 * np.mean(q_full >= v) for v in q_event if not np.isnan(v)
    ]
    return float(np.median(percentiles)) if percentiles else np.nan


def compute_event_stats(
    event: pd.Series,
    q: pd.Series,
) -> dict:
    """Compute summary statistics for one event row.

    Parameters
    ----------
    event : pd.Series
        A row from the events DataFrame (must have 'start_date', 'end_date').
    q : pd.Series
        Full daily streamflow series for this gage.

    Returns
    -------
    dict
        Keys: mean_Q, std_Q, min_Q, max_Q, cv_Q, fdc_pct.
    """
    window = q.loc[event["start_date"] : event["end_date"]]
    vals = window.dropna().values

    stats = {
        "mean_Q": float(np.mean(vals)) if len(vals) else np.nan,
        "std_Q": float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan,
        "min_Q": float(np.min(vals)) if len(vals) else np.nan,
        "max_Q": float(np.max(vals)) if len(vals) else np.nan,
        "cv_Q": np.nan,
        "fdc_pct": flow_duration_percentile(vals, q.dropna().values),
    }
    if stats["mean_Q"] and stats["mean_Q"] > 0:
        stats["cv_Q"] = stats["std_Q"] / stats["mean_Q"]

    return stats


def add_event_stats(events_df: pd.DataFrame, q: pd.Series) -> pd.DataFrame:
    """Apply compute_event_stats to every row of an events DataFrame in place.

    Returns the events DataFrame with new stat columns appended.
    """
    stat_records = [compute_event_stats(row, q) for _, row in events_df.iterrows()]
    stats_df = pd.DataFrame(stat_records, index=events_df.index)
    return pd.concat([events_df, stats_df], axis=1)
