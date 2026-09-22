"""
Ensemble voting to produce the final daily BFD label.

The three method families (RF-BFD, baseflowx filters, PyBFS) are at least
partially independent — a trained classifier, recursive digital filters, and a
storage-based state-space model — so combining their votes is preferable to any
single method.

Voting rule: a day is labeled BFD (1) if the fraction of methods voting BFD
meets or exceeds `voting_fraction` (default 0.5 = simple majority).

For a gage where PyBFS has no calibrated parameters, only the two available
methods vote (the threshold still applies as a fraction of available methods).

The ratio test each separation method's vote is based on
(baseflow_estimate / Q >= threshold) can be a single flat number, or a
magnitude-adaptive ramp — see `effective_threshold()` and
docs/adaptive-bfd-threshold.md for why a fixed threshold under-labels flat,
low-flow recessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bfd_db.config import (
    BF_RATIO_THRESHOLD,
    FLATNESS_CV_THRESHOLD,
    FLATNESS_WINDOW,
    HIGH_FLOW_ANCHOR,
    LOW_FLOW_ANCHOR,
    VOTING_FRACTION,
    PipelineConfig,
)
from bfd_db.labeling import rf_bfd, baseflowx_methods, pybfs_labeling


def anchor_flows(
    q: pd.Series,
    low_flow_anchor: float = LOW_FLOW_ANCHOR,
    high_flow_anchor: float = HIGH_FLOW_ANCHOR,
) -> tuple[float, float]:
    """The two flows the threshold ramp is pinned to, in cfs.

    Both are quantiles of the gage's own non-zero flow, so `low_flow_anchor=0.10`
    is the 10th percentile of flow -- the hydrologic Q90, exceeded 90% of the
    time -- not the 90th.
    """
    q_pos = q.replace(0, np.nan)
    return float(q_pos.quantile(low_flow_anchor)), float(q_pos.quantile(high_flow_anchor))


def effective_threshold(
    q: pd.Series,
    bf_ratio_threshold: float = BF_RATIO_THRESHOLD,
    bf_ratio_threshold_low: float | None = None,
    low_flow_anchor: float = LOW_FLOW_ANCHOR,
    high_flow_anchor: float = HIGH_FLOW_ANCHOR,
) -> pd.Series:
    """Per-day ratio threshold, ramped by flow magnitude.

    Each separation method plateaus at its own ceiling of qb/Q well below 1.0 at
    low flow (Eckhardt ~0.87, Chapman ~0.75, PyBFS ~0.60 on a typical gage), so a
    flat 0.95 is unreachable exactly where the record is most obviously
    baseflow-dominant. Relaxing the threshold as flow falls recovers those days
    without touching the storm end of the record.

    `bf_ratio_threshold_low=None` (the default) turns the ramp off: every day
    gets the flat `bf_ratio_threshold`, matching pre-ramp behavior exactly. This
    is what `PipelineConfig()` defaults to, so the national database build is
    unaffected unless a caller opts in.
    """
    if bf_ratio_threshold_low is None:
        return pd.Series(bf_ratio_threshold, index=q.index)

    # A "low-flow" threshold above the high-flow one would invert the ramp.
    bf_ratio_threshold_low = min(bf_ratio_threshold_low, bf_ratio_threshold)
    low_flow_anchor = float(np.clip(low_flow_anchor, 0.0, 1.0))
    high_flow_anchor = float(np.clip(high_flow_anchor, 0.0, 1.0))

    q_pos = q.replace(0, np.nan)
    q_lo, q_hi = anchor_flows(q, low_flow_anchor, high_flow_anchor)

    # No usable ramp: degenerate record (near-constant or all-zero flow), or
    # anchors given in the wrong order. Fall back to the flat strict threshold,
    # which the reported threshold range makes visible to callers.
    if not (np.isfinite(q_lo) and np.isfinite(q_hi) and q_hi > q_lo > 0):
        return pd.Series(bf_ratio_threshold, index=q.index)

    span = np.log(q_hi) - np.log(q_lo)
    s = (np.log(q_pos) - np.log(q_lo)) / span
    t = bf_ratio_threshold_low + (bf_ratio_threshold - bf_ratio_threshold_low) * s.clip(0, 1)
    # Days with no flow keep the strict threshold; their ratio is NaN anyway.
    return t.fillna(bf_ratio_threshold)


def label_from_ratio(q_b: pd.Series, q: pd.Series, threshold: float | pd.Series) -> pd.Series:
    """0/1 BFD label from a baseflow estimate and a ratio threshold.

    `threshold` may be a scalar (flat) or a per-day Series (e.g. from
    `effective_threshold()`), sharing `q`'s index.

    The ratio is clamped at 1.0 before comparison: separated baseflow at or
    above total flow is unphysical but does occur (PyBFS especially), and
    counting it as fully baseflow-dominant is more honest than carrying an
    arbitrarily large ratio around. The clamp never changes a label — a ratio
    above 1.0 already passes every threshold <= 1.0.
    """
    ratio = (q_b / q.replace(0, np.nan)).clip(upper=1.0)
    return (ratio >= threshold).astype(int).fillna(0)


def flatness_vote(
    q: pd.Series,
    window: int = FLATNESS_WINDOW,
    cv_threshold: float = FLATNESS_CV_THRESHOLD,
) -> pd.Series:
    """0/1 vote from flow flatness alone, independent of any separation
    method's qb/Q ratio.

    A day votes BFD if the trailing `window`-day coefficient of variation of
    Q is below `cv_threshold` and flow has not risen over that window. This
    catches sustained low, flat recessions where the digital filters'
    recession-decay memory has not caught up to the observed flow yet (their
    qb/Q can sit well below any reasonable threshold on exactly these days —
    see docs/adaptive-bfd-threshold.md #8), independent of the ratio test.

    Off by default everywhere it matters (PipelineConfig.use_flatness_vote):
    this changes labels, so opting in is a deliberate choice, not a silent
    side effect of adding the function.
    """
    roll_mean = q.rolling(window).mean()
    roll_std = q.rolling(window).std()
    cv = roll_std / roll_mean.replace(0, np.nan)
    flat = cv < cv_threshold
    not_rising = q <= q.shift(window - 1) * (1 + cv_threshold)
    vote = (flat & not_rising).astype(int).fillna(0)
    vote.name = "flatness"
    return vote


def apply_flatness_override(ensemble: pd.Series, flatness: pd.Series) -> pd.Series:
    """OR a flatness vote into an already-computed ensemble label.

    Flatness is a rare condition (a few percent of days on a typical gage —
    see flatness_vote()), so folding it in as one more equal-weight voter in
    ensemble_vote() dilutes the majority on *every* day, not just flat ones,
    and net-decreases the BFD rate. An OR-override avoids that: a day the
    ratio-based majority already called BFD stays BFD regardless of
    `flatness`, and flatness only rescues days the majority missed.
    """
    combined = (ensemble.astype(bool) | flatness.astype(bool)).astype(int)
    combined.name = "ensemble"
    return combined


def voter_agreement_stats(labels: pd.DataFrame, ensemble: pd.Series) -> dict:
    """Per-voter scorecard over the given record: how often each method calls
    BFD, and how often it lands on the same side as the ensemble it helped
    produce.
    """
    return {
        v: {
            "bfd_fraction": round(float(labels[v].mean()), 4),
            "agreement": round(float((labels[v] == ensemble).mean()), 4),
        }
        for v in labels.columns
    }


def ensemble_vote(
    method_labels: dict[str, pd.Series],
    voting_fraction: float = VOTING_FRACTION,
) -> pd.Series:
    """Combine per-method 0/1 label series by fractional majority vote.

    Parameters
    ----------
    method_labels : dict[str, pd.Series]
        Keys are method names (e.g. 'rf_bfd', 'bfx_eckhardt', 'pybfs');
        values are 0/1 integer Series sharing the same DatetimeIndex.
    voting_fraction : float
        Minimum fraction of available methods that must vote BFD.

    Returns
    -------
    pd.Series[int]
        0/1 ensemble label, same DatetimeIndex as the inputs.
    """
    df = pd.DataFrame(method_labels)
    vote_frac = df.mean(axis=1)
    ensemble = (vote_frac >= voting_fraction).astype(int)
    ensemble.name = "ensemble"
    return ensemble


def label_gage_all_methods(
    site_no: str,
    q: pd.Series,
    config: PipelineConfig,
    rf_model=None,
    rf_scaler=None,
) -> pd.DataFrame:
    """Run all three method families on one gage and return a full label table.

    Returns a DataFrame with columns:
        rf_bfd          — RF-BFD classifier label
        bfx_eckhardt    — Eckhardt filter label (and other bfx methods)
        pybfs           — PyBFS label (absent if no params/convergence)
        ensemble        — final majority-vote label
        qb_<method>     — raw baseflow estimate per ratio-based method

    The ratio test behind `bfx_*` and `pybfs` uses `config.bf_ratio_threshold`,
    optionally ramped by flow magnitude if `config.bf_ratio_threshold_low` is
    set — see `effective_threshold()`.

    Parameters
    ----------
    site_no : str
        USGS site number (for PyBFS param lookup).
    q : pd.Series
        Daily streamflow, DatetimeIndex.
    config : PipelineConfig
        Pipeline parameters (bf_ratio_threshold[_low], anchors, voting_fraction,
        methods).
    rf_model, rf_scaler
        Pre-loaded RF-BFD artifacts (pass in for batch efficiency).
    """
    labels: dict[str, pd.Series] = {}
    raw: dict[str, pd.Series] = {}

    t_eff = effective_threshold(
        q,
        config.bf_ratio_threshold,
        config.bf_ratio_threshold_low,
        config.low_flow_anchor,
        config.high_flow_anchor,
    )

    # 1. RF-BFD — a classifier, unaffected by the ratio threshold.
    rf_label = rf_bfd.label_gage(q, rf_model, rf_scaler)
    labels["rf_bfd"] = rf_label

    # 2. baseflowx methods
    for method in config.baseflowx_methods:
        q_b = baseflowx_methods.separate_baseflow(q, method)
        raw[f"qb_{method}"] = q_b
        labels[f"bfx_{method}"] = label_from_ratio(q_b, q, t_eff)

    # 3. PyBFS
    pybfs_params = pybfs_labeling.load_pybfs_params(site_no)
    if pybfs_params is not None:
        try:
            q_b_pybfs = pybfs_labeling.run_pybfs_separation(q, pybfs_params)
        except Exception:
            q_b_pybfs = None  # gage has params but PyBFS failed to converge
        if q_b_pybfs is not None:
            raw["qb_pybfs"] = q_b_pybfs
            labels["pybfs"] = label_from_ratio(q_b_pybfs, q, t_eff)

    # 4. Ensemble vote over the ratio + classifier voters
    ensemble = ensemble_vote(labels, config.voting_fraction)

    # 5. Flatness override (opt-in; see flatness_vote() and
    # apply_flatness_override()) — rescues flat recessions the digital
    # filters lag behind on, without diluting the vote on every other day.
    if config.use_flatness_vote:
        flatness = flatness_vote(q, config.flatness_window, config.flatness_cv_threshold)
        labels["flatness"] = flatness
        ensemble = apply_flatness_override(ensemble, flatness)

    labels["ensemble"] = ensemble

    result = pd.DataFrame(labels)
    for col, series in raw.items():
        result[col] = series

    return result
