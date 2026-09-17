"""
Causal (look-ahead-free) onset detector for BFD periods.

The retrospective events table uses the full record to determine event
boundaries — it can look forward to decide whether to pool a gap. The
forecast evaluation (Objective 3) additionally needs a *causal* onset: the
first day on which we can declare, using only data up to that day, that the
gage has entered a baseflow-dominant period.

Definition: onset fires on day t if days [t-k+1 … t] are all BFD (label==1),
where k is tunable (default K_ONSET = 5). The onset date is reported as the
*start* of that k-day confirmed run (i.e., t - k + 1), not the detection date.
This is how it maps onto the retrospective events: onset = first confirmed day
of the event, available k days after it begins.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from bfd_db.config import K_ONSET


def detect_causal_onsets(
    daily_labels: pd.Series,
    k: int = K_ONSET,
) -> pd.Series:
    """Identify causal BFD onset dates (no look-ahead).

    Parameters
    ----------
    daily_labels : pd.Series[int]
        Daily 0/1 BFD label series, DatetimeIndex.
    k : int
        Number of consecutive BFD days required to confirm onset.

    Returns
    -------
    pd.Series[int]
        Binary series (same index) where 1 marks a causal onset *detection*
        day — the day we have seen k consecutive BFD days and can fire a
        forecast. The corresponding onset start date is k-1 days earlier.

    Notes
    -----
    Only the *first* detection within each run is flagged; subsequent
    consecutive BFD days within the same run are not re-flagged.
    """
    arr = daily_labels.values.astype(int)
    # Rolling sum of last k days: onset fires when rolling sum == k
    rolling_sum = np.convolve(arr, np.ones(k, dtype=int), mode="full")[: len(arr)]
    detections = np.zeros(len(arr), dtype=int)

    in_event = False
    for i in range(k - 1, len(arr)):
        if rolling_sum[i] == k:
            if not in_event:
                detections[i] = 1   # first detection in this run
                in_event = True
        else:
            in_event = False

    return pd.Series(detections, index=daily_labels.index, name="causal_onset")


def onset_start_date(detection_date: pd.Timestamp, k: int = K_ONSET) -> pd.Timestamp:
    """Convert a causal detection date to the corresponding onset start date.

    The onset start is k-1 days before detection (the first day of the k-day run).
    """
    return detection_date - pd.Timedelta(days=k - 1)
