"""
Validation of BFD labels against the Objective 1 expert-labeled benchmark.

The benchmark covers 182 USGS gages hand-labeled by 10 student hydrologists.
These labels must NOT influence RF-BFD training (they were used for Obj 1);
they are used here as an independent test set for the ensemble.

Evaluation levels:
  1. Daily classification: precision, recall, F1, accuracy per method and ensemble.
  2. Event level: hit/miss rate, onset-date error (days), duration error (days).
  3. Spatial cross-validation: leave-HUC2-region-out to test transfer to unseen basins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def load_benchmark_labels(benchmark_path: str) -> pd.DataFrame:
    """Load the 182-gage expert-labeled benchmark from Objective 1.

    Expected format: long-form CSV/Parquet with columns:
        STAID, date, bfd_label (0/1)
    Adjust the path and format to wherever Obj 1 labels are stored.
    """
    return pd.read_parquet(benchmark_path)


def evaluate_daily(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    label: str = "method",
) -> dict:
    """Compute precision, recall, F1, and accuracy for a daily 0/1 prediction.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0/1) from the benchmark.
    y_pred : array-like
        Predicted labels (0/1) from the method.
    label : str
        Name of the method (for display in results tables).
    """
    return {
        "method": label,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "n_days": len(y_true),
    }


def evaluate_all_methods(
    benchmark: pd.DataFrame,
    labels_db: pd.DataFrame,
    methods: list[str] | None = None,
) -> pd.DataFrame:
    """Compare all methods (and ensemble) against the benchmark across 182 gages.

    Parameters
    ----------
    benchmark : pd.DataFrame
        Benchmark labels with columns [STAID, date, bfd_label].
    labels_db : pd.DataFrame
        Daily labels table from the database.
    methods : list[str]
        Label columns to evaluate; defaults to all method + ensemble columns.

    Returns
    -------
    pd.DataFrame
        One row per method with precision, recall, F1, accuracy.
    """
    if methods is None:
        methods = ["rf_bfd", "bfx_eckhardt", "bfx_chapman", "bfx_lyne_hollick", "pybfs", "ensemble"]

    merged = benchmark.merge(
        labels_db,
        on=["STAID", "date"],
        how="inner",
    )

    results = []
    for method in methods:
        if method not in merged.columns:
            continue
        valid = merged.dropna(subset=[method])
        results.append(
            evaluate_daily(valid["bfd_label"], valid[method].astype(int), label=method)
        )

    return pd.DataFrame(results).sort_values("f1", ascending=False)


def spatial_crossval(
    benchmark: pd.DataFrame,
    labels_db: pd.DataFrame,
    region_col: str = "HUC2",
) -> pd.DataFrame:
    """Leave-one-HUC2-region-out spatial cross-validation.

    For each HUC2 region, evaluate the ensemble on that region's benchmark
    gages using labels produced by a model trained without them.
    Returns per-region F1 scores to detect regional performance gaps.

    NOTE: this requires that the RF-BFD model was retrained without each
    region's training data. If using the fixed Obj 1 model, this measures
    transfer skill rather than true leave-out CV.
    """
    regions = benchmark[region_col].unique()
    records = []
    for region in regions:
        mask = benchmark[region_col] == region
        test = benchmark[mask]
        merged = test.merge(labels_db, on=["STAID", "date"], how="inner")
        if merged.empty:
            continue
        valid = merged.dropna(subset=["ensemble"])
        row = evaluate_daily(valid["bfd_label"], valid["ensemble"].astype(int), label=str(region))
        row["region"] = region
        records.append(row)
    return pd.DataFrame(records).sort_values("f1")
