"""
Unit tests for BFD event pooling (the algorithm with no external dependencies).
This is the first test to write and run — it validates the core IET pooling logic.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from bfd_db.events.pooling import find_runs, pool_runs, pool_events


def test_find_runs_basic():
    s = pd.Series([0, 1, 1, 1, 0, 0, 1, 1])
    runs = find_runs(s)
    assert runs == [(1, 3), (6, 7)]


def test_find_runs_empty():
    s = pd.Series([0, 0, 0])
    assert find_runs(s) == []


def test_pool_runs_merges_small_gap():
    # Gap of 2 between runs (1,3) and (6,7); t_c=2 should merge them
    runs = [(1, 3), (6, 7)]
    merged = pool_runs(runs, t_c=2)
    assert merged == [(1, 7)]


def test_pool_runs_keeps_large_gap():
    runs = [(1, 3), (7, 9)]
    # Gap = 7-3-1 = 3; t_c=2 should NOT merge
    merged = pool_runs(runs, t_c=2)
    assert merged == [(1, 3), (7, 9)]


def test_pool_events_dmin_filter():
    # 10-day series: days 0-3 BFD, gap 4-5, days 6-9 BFD
    dates = pd.date_range("2020-01-01", periods=10)
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], index=dates)

    # With t_c=2: merges into one 10-day event (duration=10 >= d_min=7)
    events = pool_events(labels, t_c=2, d_min=7)
    assert len(events) == 1
    assert events.iloc[0]["duration"] == 10

    # With t_c=1: gap of 2 is NOT merged; each 4-day run is below d_min=5
    events2 = pool_events(labels, t_c=1, d_min=5)
    assert len(events2) == 0


def test_pool_events_n_bfd_days():
    dates = pd.date_range("2020-01-01", periods=10)
    labels = pd.Series([1, 1, 1, 1, 0, 0, 1, 1, 1, 1], index=dates)
    events = pool_events(labels, t_c=2, d_min=1)
    assert events.iloc[0]["n_bfd_days"] == 8
    assert events.iloc[0]["gap_days"] == 2
