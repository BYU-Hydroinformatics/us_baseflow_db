"""
CONUS-scale spatial analysis of BFD prevalence.

Key outputs for Paper 2:
  - Per-gage BFD fraction (% of days labeled BFD in the full record)
  - Regional summary statistics (by HUC2 and ecoregion)
  - Seasonal signatures (which months drive BFD prevalence?)
  - Map figure: scatter/choropleth of BFD fraction across CONUS gages

These mirror the 182-gage spatial analysis from Objective 1, extended to
the full GAGES-II network of 9,322 gages.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


def compute_bfd_prevalence(
    labels: pd.DataFrame,
    label_col: str = "ensemble",
    approved_only: bool = True,
) -> pd.DataFrame:
    """Compute per-gage BFD fraction from the daily labels table.

    Returns a DataFrame indexed by STAID with columns:
        bfd_fraction  — fraction of days labeled BFD (0–1)
        n_days        — total days with non-null labels
        n_bfd_days    — total BFD days
    """
    if approved_only:
        labels = labels[labels["approval"] == "A"]

    grouped = labels.groupby("STAID")[label_col]
    prevalence = pd.DataFrame(
        {
            "bfd_fraction": grouped.mean(),
            "n_bfd_days": grouped.sum(),
            "n_days": grouped.count(),
        }
    )
    return prevalence


def merge_with_gagesii(
    prevalence: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Join BFD prevalence with GAGES-II lat/lon and metadata."""
    return prevalence.join(meta[["LAT_GAGE", "LNG_GAGE", "CLASS", "HUC2"]])


def plot_conus_map(
    prevalence_with_coords: pd.DataFrame,
    output_path: Path | None = None,
    title: str = "BFD Prevalence Across CONUS (GAGES-II Gages)",
) -> plt.Figure:
    """Scatter map of BFD fraction by gage location.

    Parameters
    ----------
    prevalence_with_coords : pd.DataFrame
        Must have columns: LAT_GAGE, LNG_GAGE, bfd_fraction.
    output_path : Path
        If provided, saves the figure; otherwise just returns it.
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    sc = ax.scatter(
        prevalence_with_coords["LNG_GAGE"],
        prevalence_with_coords["LAT_GAGE"],
        c=prevalence_with_coords["bfd_fraction"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=8,
        alpha=0.7,
        linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="BFD Fraction")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.set_xlim(-130, -60)
    ax.set_ylim(24, 50)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def regional_bfd_stats(
    prevalence_with_coords: pd.DataFrame,
    group_col: str = "HUC2",
) -> pd.DataFrame:
    """Summary statistics of BFD prevalence grouped by HUC2 (or ecoregion).

    Returns mean, median, std BFD fraction per region, sorted by mean.
    """
    return (
        prevalence_with_coords
        .groupby(group_col)["bfd_fraction"]
        .agg(["mean", "median", "std", "count"])
        .sort_values("mean", ascending=False)
    )


def seasonal_bfd_fractions(
    labels: pd.DataFrame,
    label_col: str = "ensemble",
) -> pd.DataFrame:
    """Monthly BFD fraction averaged across all gages — seasonal signature.

    Returns a 12-row DataFrame (one per month) with mean and std BFD fraction.
    """
    labels = labels.copy()
    labels["month"] = pd.to_datetime(labels["date"]).dt.month
    monthly = labels.groupby(["STAID", "month"])[label_col].mean().reset_index()
    return (
        monthly.groupby("month")[label_col]
        .agg(mean="mean", std="std")
        .reset_index()
        .rename(columns={"month": "month"})
    )
