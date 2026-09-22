"""
Unit tests for the ensemble ratio threshold and voting math
(bfd_db.labeling.ensemble) — the algorithm pieces that have no dependency on
a trained model, calibrated PyBFS params, or real gage data.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bfd_db.labeling.ensemble import (
    anchor_flows,
    apply_flatness_override,
    effective_threshold,
    ensemble_vote,
    flatness_vote,
    label_from_ratio,
    voter_agreement_stats,
)


def _flow_series(values):
    dates = pd.date_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=dates, dtype=float)


def test_anchor_flows_excludes_zeros():
    q = _flow_series([0, 0, 1, 2, 3, 4, 100])
    q_lo, q_hi = anchor_flows(q, low_flow_anchor=0.0, high_flow_anchor=1.0)
    # Zeros dropped before quantiles: min/max of the non-zero values.
    assert q_lo == 1.0
    assert q_hi == 100.0


def test_effective_threshold_flat_when_low_is_none():
    q = _flow_series([1, 10, 100, 1000])
    t = effective_threshold(q, bf_ratio_threshold=0.95, bf_ratio_threshold_low=None)
    assert (t == 0.95).all()


def test_effective_threshold_ramps_between_anchors():
    # Q10=10, Q90=1000 style spread; low_flow_anchor/high_flow_anchor pick
    # quantiles of this series directly since values are evenly spaced ranks.
    q = _flow_series(np.geomspace(1, 1000, num=11))
    t = effective_threshold(
        q, bf_ratio_threshold=0.95, bf_ratio_threshold_low=0.75,
        low_flow_anchor=0.0, high_flow_anchor=1.0,
    )
    # Monotonic non-decreasing with flow, bounded by [t_lo, t_hi].
    assert t.iloc[0] == 0.75
    assert t.iloc[-1] == 0.95
    assert (t.diff().dropna() >= 0).all()
    assert t.between(0.75, 0.95).all()


def test_effective_threshold_clamps_inverted_low_above_high():
    q = _flow_series(np.geomspace(1, 1000, num=11))
    t = effective_threshold(
        q, bf_ratio_threshold=0.8, bf_ratio_threshold_low=0.95,
        low_flow_anchor=0.0, high_flow_anchor=1.0,
    )
    # t_lo clamped down to t_hi: the ramp flattens, never inverts.
    assert (t == 0.8).all()


def test_effective_threshold_degenerate_record_falls_back_flat():
    q = _flow_series([5.0] * 10)  # constant flow -> q_lo == q_hi
    t = effective_threshold(q, bf_ratio_threshold=0.95, bf_ratio_threshold_low=0.75)
    assert (t == 0.95).all()


def test_label_from_ratio_basic_threshold():
    q = _flow_series([10, 10, 10])
    q_b = _flow_series([9.5, 8.0, 10.0])
    labels = label_from_ratio(q_b, q, threshold=0.9)
    assert labels.tolist() == [1, 0, 1]


def test_label_from_ratio_clamp_does_not_change_labels():
    q = _flow_series([10])
    q_b_over = _flow_series([40])  # ratio 4.0, unphysical but occurs (e.g. PyBFS)
    labels = label_from_ratio(q_b_over, q, threshold=1.0)
    assert labels.tolist() == [1]  # clamped to 1.0, still passes threshold==1.0


def test_label_from_ratio_zero_flow_is_not_bfd():
    q = _flow_series([0])
    q_b = _flow_series([0])
    labels = label_from_ratio(q_b, q, threshold=0.5)
    assert labels.tolist() == [0]


def test_label_from_ratio_accepts_per_day_threshold_series():
    q = _flow_series([10, 10])
    q_b = _flow_series([8.5, 8.5])
    threshold = pd.Series([0.8, 0.9], index=q.index)
    labels = label_from_ratio(q_b, q, threshold)
    assert labels.tolist() == [1, 0]


def test_ensemble_vote_majority():
    dates = pd.date_range("2020-01-01", periods=3)
    labels = {
        "a": pd.Series([1, 1, 0], index=dates),
        "b": pd.Series([1, 0, 0], index=dates),
        "c": pd.Series([0, 0, 0], index=dates),
    }
    ensemble = ensemble_vote(labels, voting_fraction=0.5)
    assert ensemble.tolist() == [1, 0, 0]


def test_flatness_vote_flags_constant_stretch():
    # 20 flat days at Q=10 preceded by a falling limb from a peak.
    q = _flow_series([100, 60, 30, 15] + [10.0] * 20)
    vote = flatness_vote(q, window=5, cv_threshold=0.05)
    assert vote.iloc[-5:].tolist() == [1, 1, 1, 1, 1]


def test_flatness_vote_ignores_a_rising_limb():
    q = _flow_series([10.0] * 10 + [12, 20, 40, 80, 150])
    vote = flatness_vote(q, window=5, cv_threshold=0.05)
    assert vote.iloc[-1] == 0


def test_flatness_vote_ignores_a_noisy_stretch():
    q = _flow_series([10, 20, 8, 22, 9, 21, 10, 19, 11, 18])
    vote = flatness_vote(q, window=5, cv_threshold=0.05)
    assert vote.iloc[-1] == 0


def test_apply_flatness_override_only_adds_days():
    dates = pd.date_range("2020-01-01", periods=4)
    ensemble = pd.Series([1, 0, 0, 0], index=dates)
    flatness = pd.Series([0, 1, 0, 0], index=dates)
    combined = apply_flatness_override(ensemble, flatness)
    assert combined.tolist() == [1, 1, 0, 0]


def test_apply_flatness_override_never_removes_a_day():
    dates = pd.date_range("2020-01-01", periods=2)
    ensemble = pd.Series([1, 0], index=dates)
    flatness = pd.Series([0, 0], index=dates)
    combined = apply_flatness_override(ensemble, flatness)
    assert combined.tolist() == [1, 0]


def test_voter_agreement_stats():
    dates = pd.date_range("2020-01-01", periods=4)
    labels = pd.DataFrame({
        "a": [1, 1, 0, 0],
        "b": [1, 0, 0, 0],
    }, index=dates)
    ensemble = pd.Series([1, 1, 0, 0], index=dates)
    stats = voter_agreement_stats(labels, ensemble)
    assert stats["a"]["bfd_fraction"] == 0.5
    assert stats["a"]["agreement"] == 1.0
    assert stats["b"]["bfd_fraction"] == 0.25
    assert stats["b"]["agreement"] == 0.75
