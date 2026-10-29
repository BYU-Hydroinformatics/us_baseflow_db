"""
BFD event generation by inter-event time (IET) pooling.

Algorithm (adapted from Tallaksen et al. 1997; Fleig et al. 2006):
  1. Find all maximal runs of consecutive BFD days (label == 1).
  2. Merge two adjacent runs into one event if the non-BFD gap <= t_c days.
  3. Discard any resulting event shorter than d_min days.

Days with no record (missing dates in the index) are never part of an event:
runs and pooling are computed separately on each stretch of consecutive
calendar days, so an event ends at the last day before a data gap.

This single mechanism handles both brief excursions (1–2 day blips above the
threshold) and longer interruptions inside an otherwise BFD-dominant period.

The pooled events table is the main product of Objective 2 and the starting
point for the Objective 3 forecast evaluation.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from bfd_db.config import T_C, D_MIN


def find_runs(series: pd.Series, value: int = 1) -> list[tuple[int, int]]:
    """Find start/end integer positions of all maximal runs of `value` in series.

    Returns a list of (start_pos, end_pos) tuples (both inclusive).
    """
    arr = series.values
    runs = []
    i = 0
    while i < len(arr):
        if arr[i] == value:
            j = i
            while j < len(arr) and arr[j] == value:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def pool_runs(
    runs: list[tuple[int, int]],
    t_c: int,
) -> list[tuple[int, int]]:
    """Merge adjacent runs whose gap is <= t_c into a single event.

    Works on integer positions; the gap between run A and run B is:
        B[0] - A[1] - 1
    """
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        gap = start - merged[-1][1] - 1
        if gap <= t_c:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def contiguous_segments(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Split a sorted daily DatetimeIndex into stretches of consecutive days.

    Returns (start_pos, end_pos) tuples (both inclusive); a new stretch starts
    wherever the next date is more than one day after the previous one.
    """
    if len(index) == 0:
        return []
    breaks = np.flatnonzero(np.diff(index.values) != np.timedelta64(1, "D")) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks - 1, [len(index) - 1]])
    return list(zip(starts.tolist(), ends.tolist()))


def pool_events(
    daily_labels: pd.Series,
    t_c: int = T_C,
    d_min: int = D_MIN,
) -> pd.DataFrame:
    """Derive BFD events from a daily 0/1 label series.

    Parameters
    ----------
    daily_labels : pd.Series[int]
        Daily 0/1 BFD label series, DatetimeIndex. Dates missing from the
        index are treated as having no record and split events.
    t_c : int
        Maximum non-BFD gap (days) for merging adjacent events.
    d_min : int
        Minimum event duration in days; shorter events are discarded.

    Returns
    -------
    pd.DataFrame with columns:
        start_date  — event start (first BFD day)
        end_date    — event end (last BFD day in pooled run)
        duration    — days from start_date to end_date inclusive
        n_bfd_days  — actual count of BFD days within the event window
        gap_days    — total non-BFD days absorbed by pooling
    """
    pooled = []
    for seg_start, seg_end in contiguous_segments(daily_labels.index):
        segment = daily_labels.iloc[seg_start : seg_end + 1]
        runs = find_runs(segment)
        pooled += [(seg_start + s, seg_start + e) for s, e in pool_runs(runs, t_c)]

    records = []
    dates = daily_labels.index
    for start_pos, end_pos in pooled:
        duration = end_pos - start_pos + 1
        if duration < d_min:
            continue
        window = daily_labels.iloc[start_pos : end_pos + 1]
        records.append(
            {
                "start_date": dates[start_pos],
                "end_date": dates[end_pos],
                "duration": duration,
                "n_bfd_days": int(window.sum()),
                "gap_days": duration - int(window.sum()),
            }
        )

    return pd.DataFrame(records)
