"""
Database schema and provenance record.

Two Parquet tables form the database:

1. daily_labels  (data/outputs/daily_labels.parquet)
   ┌─────────────┬──────────────────────────────────────────────────────┐
   │ STAID       │ 8-digit USGS site number                             │
   │ date        │ daily DatetimeIndex                                   │
   │ Q_cfs       │ observed streamflow (ft³/s)                          │
   │ approval    │ approval code (always 'A'; see bfd_db.data.nwis)     │
   │ rf_bfd      │ RF-BFD label (0/1)                                   │
   │ bfx_eckhardt│ Eckhardt filter label (0/1)                          │
   │ bfx_chapman │ Chapman filter label (0/1)                           │
   │ pybfs       │ PyBFS label (0/1; NaN if no params)                  │
   │ ensemble    │ final ensemble label (0/1) — PRIMARY PRODUCT          │
   └─────────────┴──────────────────────────────────────────────────────┘

2. bfd_events  (data/outputs/bfd_events.parquet)
   ┌──────────────┬─────────────────────────────────────────────────────┐
   │ STAID        │ 8-digit USGS site number                            │
   │ start_date   │ event start (first BFD day)                         │
   │ end_date     │ event end (last BFD day in pooled window)            │
   │ duration     │ calendar days (end - start + 1)                     │
   │ n_bfd_days   │ actual BFD days within the window                   │
   │ gap_days     │ non-BFD days absorbed by pooling                    │
   │ mean_Q       │ mean daily flow during event (ft³/s)                │
   │ std_Q        │ standard deviation of daily flow                    │
   │ cv_Q         │ coefficient of variation                            │
   │ fdc_pct      │ median exceedance percentile on full FDC            │
   │ has_provisional│ True if any day in event has approval != 'A'      │
   │ CLASS        │ GAGES-II reference/non-reference                    │
   │ HYDROmod     │ GAGES-II hydrologic disturbance index               │
   └──────────────┴─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import datetime
from importlib.metadata import version

import pandas as pd


DAILY_LABELS_COLS = [
    "STAID", "date", "Q_cfs", "approval",
    "rf_bfd", "bfx_eckhardt", "bfx_chapman", "bfx_lyne_hollick", "pybfs",
    "ensemble",
]

BFD_EVENTS_COLS = [
    "STAID", "start_date", "end_date", "duration", "n_bfd_days", "gap_days",
    "mean_Q", "std_Q", "cv_Q", "fdc_pct", "has_provisional",
    "CLASS", "HYDROmod_QAchange",
]


def create_provenance_record(
    params: dict,
    retrieval_date: str | None = None,
) -> dict:
    """Create a provenance stamp for a database build.

    Every generated database — user regeneration or published snapshot —
    should carry this record so results are reproducible.

    Parameters
    ----------
    params : dict
        Serialized PipelineConfig (call dataclasses.asdict(config)).
    retrieval_date : str
        ISO date of NWIS data retrieval; defaults to today.

    Returns
    -------
    dict
        Provenance record suitable for storing as JSON alongside the tables.
    """
    if retrieval_date is None:
        retrieval_date = datetime.date.today().isoformat()

    prov: dict = {
        "retrieval_date": retrieval_date,
        "params": params,
        "package_versions": {},
    }
    for pkg in ("bfd_db", "baseflowx", "pybfs", "scikit_learn"):
        try:
            prov["package_versions"][pkg] = version(pkg)
        except Exception:  # noqa: BLE001
            prov["package_versions"][pkg] = "unknown"

    return prov
