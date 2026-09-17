"""
RF-BFD classifier (Objective 1).

Applies the trained Random Forest model to produce per-day BFD labels for
any NWIS daily streamflow series. Model and scaler artifacts live in
data/raw/gagesii/ (random_forest_bfd_model.joblib, feature_scaler.joblib),
alongside data/raw/gagesii/create_model.py — the training script that
produced them.

Feature engineering here must exactly mirror create_model.py's
process_file(), since the scaler was fit on that exact column set/order
(verified against scaler.feature_names_in_):
  streamflow/Mean, Mean_streamflow, MW5_d2streamflowabs, MW5_streamflow,
  MW5_dstreamflowabs, r10m, streamflow, streamflow/Chapman, Months
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from bfd_db.config import DATA_RAW

MODEL_PATH = DATA_RAW / "gagesii" / "random_forest_bfd_model.joblib"
SCALER_PATH = DATA_RAW / "gagesii" / "feature_scaler.joblib"


def _chapman_unclipped(Q: np.ndarray, a: float = 0.925) -> np.ndarray:
    """Chapman filter exactly as implemented in create_model.py.

    b[t] = (3a-1)/(3-a) * b[t-1] + (1-a)/(3-a) * (Q[t] + Q[t-1])

    Deliberately does NOT use baseflowx.chapman(): that function clips
    b[t] to never exceed Q[t], but create_model.py's bespoke loop has no
    such clip. The RF-BFD model/scaler were trained on the unclipped
    streamflow/Chapman feature, so reproducing that exact recursion
    (not the hydrologically "corrected" library version) is required for
    predictions to be valid under the trained scaler.
    """
    alpha = (3 * a - 1) / (3 - a)
    beta = (1 - a) / (3 - a)
    b = np.empty_like(Q)
    b[0] = Q[0]
    for t in range(1, len(Q)):
        b[t] = alpha * b[t - 1] + beta * (Q[t] + Q[t - 1])
    return b

# Column order the model/scaler were trained on (see create_model.py).
_FEATURE_ORDER = [
    "streamflow/Mean",
    "Mean_streamflow",
    "MW5_d2streamflowabs",
    "MW5_streamflow",
    "MW5_dstreamflowabs",
    "r10m",
    "streamflow",
    "streamflow/Chapman",
    "Months",
]


def load_rf_bfd_model():
    """Load the trained RF-BFD classifier and scaler from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def compute_features(q: pd.Series) -> pd.DataFrame:
    """Compute the RF-BFD feature matrix for a daily discharge series.

    Mirrors process_file() in create_model.py formula-for-formula (mean and
    10th-percentile are computed once over the whole series, matching how
    the model was trained — these are not rolling/causal statistics).

    Parameters
    ----------
    q : pd.Series
        Daily streamflow in ft³/s, DatetimeIndex, values > 0 (already
        guaranteed by bfd_db.data.nwis.load_daily_q).

    Returns
    -------
    pd.DataFrame
        Feature matrix aligned to q's index, columns in _FEATURE_ORDER.
        Rows with NaN (filter warm-up at the start of the series) are
        forward/backward filled rather than dropped, so every day in the
        input gets a label.
    """
    df = pd.DataFrame(index=q.index)
    df["streamflow"] = q

    q_chapman = pd.Series(_chapman_unclipped(q.to_numpy(dtype=float)), index=q.index)
    df["streamflow/Chapman"] = q / q_chapman.replace(0, np.nan)

    mean_q = q.mean()
    std_q = q.std()
    outliers = (q > mean_q + 2 * std_q) | (q < mean_q - 2 * std_q)
    mean_q_trimmed = q[~outliers].mean()
    if pd.isna(mean_q_trimmed) or mean_q_trimmed == 0:
        mean_q_trimmed = mean_q if not (pd.isna(mean_q) or mean_q == 0) else 1e-10
    df["Mean_streamflow"] = mean_q_trimmed
    df["streamflow/Mean"] = q / mean_q_trimmed

    dq = q.diff()
    d2q = dq.diff()
    df["MW5_streamflow"] = q.rolling(5, min_periods=1).mean()
    df["MW5_dstreamflowabs"] = dq.abs().rolling(5, min_periods=1).mean()
    df["MW5_d2streamflowabs"] = d2q.abs().rolling(5, min_periods=1).mean()

    df["Months"] = q.index.month.astype(float)
    streamflow_monthly = q.groupby(q.index.month).transform("mean")
    p10 = q.quantile(0.1)
    if p10 == 0 or pd.isna(p10):
        p10 = 1e-10
    df["r10m"] = streamflow_monthly / p10

    return df[_FEATURE_ORDER].ffill().bfill()


def label_gage(
    q: pd.Series,
    model=None,
    scaler: StandardScaler | None = None,
) -> pd.Series:
    """Return a daily 0/1 BFD label series for a single gage using RF-BFD.

    Parameters
    ----------
    q : pd.Series
        Daily streamflow, DatetimeIndex.
    model, scaler
        Pre-loaded model and scaler (avoids redundant disk reads in batch runs).
        If None, loads from MODEL_PATH / SCALER_PATH.

    Returns
    -------
    pd.Series[int]
        0 = non-BFD, 1 = BFD, same DatetimeIndex as q.
    """
    if model is None:
        model, scaler = load_rf_bfd_model()

    features = compute_features(q)
    X = scaler.transform(features.values)
    labels = model.predict(X)
    return pd.Series(labels, index=q.index, name="rf_bfd", dtype=int)
