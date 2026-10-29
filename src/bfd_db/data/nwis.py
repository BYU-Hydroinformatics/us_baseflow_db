"""
NWIS daily streamflow loader.

Daily discharge (parameter code 00060) for every gage was downloaded ahead
of time from the National Water Information System and is stored as
per-gage CSV files in data/raw/nwis/<site_no>.csv (columns: date, streamflow).

Approval status legend (stored in 'approval' column):
  'A'  — approved (finalized)

Per-record NWIS quality codes were not retained in the source download, so
every cached record is treated as approved ('A').
"""

from __future__ import annotations

import pandas as pd

from bfd_db.config import DATA_RAW

NWIS_CACHE = DATA_RAW / "nwis"


def load_daily_q(
    site_no: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Load daily discharge for one gage from the local CSV cache.

    Returns a DataFrame with columns:
        date       (DatetimeIndex)
        Q_cfs      float — mean daily discharge in ft³/s
        approval   str   — always 'A' (see module docstring)

    Parameters
    ----------
    site_no : str
        Zero-padded 8-digit USGS site number.
    start_date, end_date : str
        ISO-format date strings (e.g. '1950-01-01').
    """
    csv_path = NWIS_CACHE / f"{site_no}.csv"
    if not csv_path.exists():
        return pd.DataFrame(columns=["Q_cfs", "approval"])

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.set_index("date").rename(columns={"streamflow": "Q_cfs"})
    df = df.loc[start_date:end_date]
    # Non-numeric entries (e.g. 'Ice', 'Eqp', blank) mark days with no
    # measured value; coerce to NaN and drop along with zero/negative reads.
    df["Q_cfs"] = pd.to_numeric(df["Q_cfs"], errors="coerce")
    df = df[df["Q_cfs"] > 0]
    df["approval"] = "A"
    return df[["Q_cfs", "approval"]]
