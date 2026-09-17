"""
Per-gage hydrograph visualization: streamflow + multi-method BFD agreement.

Two-panel figure for one gage:
  top    — streamflow hydrograph (log scale), shaded where the ensemble
           labels a day baseflow-dominant (BFD)
  bottom — per-method agreement strip (one row per method) showing which
           individual methods voted BFD on each day

This is the "show the method working" figure: it makes visible where the
RF-BFD classifier, the baseflowx digital filters, and PyBFS agree or
disagree, not just the final ensemble number.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_METHODS = [
    "rf_bfd",
    "bfx_eckhardt",
    "bfx_chapman",
    "bfx_lyne_hollick",
    "bfx_boughton",
    "pybfs",
    "ensemble",
]


def plot_hydrograph(
    labels: pd.DataFrame,
    staid: str,
    start: str | None = None,
    end: str | None = None,
    methods: list[str] | None = None,
    title: str | None = None,
    output_path: Path | None = None,
    yscale: str = "log",
    dpi: int = 300,
) -> plt.Figure:
    """Plot a streamflow hydrograph with multi-method BFD labels for one gage.

    Parameters
    ----------
    labels : pd.DataFrame
        Daily labels table for a single gage (already filtered to one
        STAID), DatetimeIndex, with columns Q_cfs + the method label
        columns in `methods`.
    staid : str
        Gage identifier, used in the default title.
    start, end : str, optional
        Date range to plot (defaults to the full available record).
    methods : list[str], optional
        Method label columns to show in the agreement strip; defaults to
        DEFAULT_METHODS.
    title : str, optional
        Figure title; defaults to a generated one.
    output_path : Path, optional
        If given, saves the figure to this path. A vector .pdf twin is also
        saved alongside it (same stem) for lossless zoom on the line data.
    yscale : str
        Y-axis scale for the streamflow panel: "log" (default) or "linear".
    dpi : int
        Raster resolution for the saved PNG (default 300).
    """
    if methods is None:
        methods = DEFAULT_METHODS

    df = labels
    if start or end:
        df = df.loc[start:end]

    fig, (ax_q, ax_methods) = plt.subplots(
        2,
        1,
        figsize=(14, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.5]},
    )

    ax_q.plot(df.index, df["Q_cfs"], color="steelblue", lw=0.8, label="Streamflow")
    ymin = max(df["Q_cfs"].min() * 0.5, 1e-3) if yscale == "log" else 0
    ax_q.fill_between(
        df.index,
        ymin,
        df["Q_cfs"],
        where=df["ensemble"] == 1,
        color="mediumseagreen",
        alpha=0.35,
        label="Ensemble BFD day",
    )
    ax_q.set_yscale(yscale)
    ax_q.set_ylabel("Streamflow (ft³/s)")
    ax_q.set_title(title or f"Gage {staid} — Streamflow & Baseflow-Dominant Days")
    ax_q.legend(loc="upper right")

    mat = df[methods].T.to_numpy(dtype=float)
    x0, x1 = mdates.date2num(df.index[0]), mdates.date2num(df.index[-1])
    ax_methods.imshow(
        mat,
        aspect="auto",
        cmap="Greens",
        vmin=0,
        vmax=1,
        extent=[x0, x1, 0, len(methods)],
        origin="lower",
        interpolation="nearest",
    )
    ax_methods.set_yticks([i + 0.5 for i in range(len(methods))])
    ax_methods.set_yticklabels(methods)
    ax_methods.set_xlabel("Date")
    ax_methods.xaxis_date()
    ax_methods.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    fig.tight_layout()
    if output_path:
        output_path = Path(output_path)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    return fig
