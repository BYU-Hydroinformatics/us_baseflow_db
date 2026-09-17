"""
Query helpers for the two-layer BFD database.

Usage example:
    from bfd_db.database.query import load_database, get_events_for_gage

    labels, events = load_database()
    site_events = get_events_for_gage("09380000", events)
    ref_events = filter_by_reference(events)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bfd_db.config import DATA_OUTPUTS


def load_database(
    output_dir: Path | None = None,
    approved_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the daily-labels and events tables from Parquet.

    Parameters
    ----------
    output_dir : Path
        Directory containing the two Parquet files; defaults to DATA_OUTPUTS.
    approved_only : bool
        If True, filter daily labels to NWIS-approved days only ('A').
        Events with has_provisional=True are still returned; filter separately.

    Returns
    -------
    labels : pd.DataFrame  — daily labels
    events : pd.DataFrame  — BFD events
    """
    d = output_dir or DATA_OUTPUTS
    labels = pd.read_parquet(d / "daily_labels.parquet")
    events = pd.read_parquet(d / "bfd_events.parquet")

    if approved_only:
        labels = labels[labels["approval"] == "A"]

    return labels, events


def get_labels_for_gage(site_no: str, labels: pd.DataFrame) -> pd.DataFrame:
    return labels[labels["STAID"] == site_no]


def get_events_for_gage(site_no: str, events: pd.DataFrame) -> pd.DataFrame:
    return events[events["STAID"] == site_no]


def filter_by_reference(events: pd.DataFrame) -> pd.DataFrame:
    """Subset events to GAGES-II reference gages only."""
    return events[events["CLASS"] == "Ref"]


def filter_by_disturbance(
    events: pd.DataFrame,
    max_hydro_mod: float = 10.0,
) -> pd.DataFrame:
    """Subset events to gages below a hydrologic disturbance threshold."""
    return events[events["HYDROmod_QAchange"] <= max_hydro_mod]
