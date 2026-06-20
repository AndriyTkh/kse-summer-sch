"""evaluate: temporal split (never random) + PR-AUC."""

import pandas as pd

from src import evaluate, index


def test_temporal_split_cut_is_time_based():
    g = index.build_master_index(
        start="2024-01-01", end="2024-03-25 23:00", oblasts=["kyivska", "lvivska"]
    )
    train, test = evaluate.temporal_split(g, test_weeks=4)
    tr = train.index.get_level_values("ts_utc")
    te = test.index.get_level_values("ts_utc")
    assert tr.max() < te.min()                         # no temporal overlap
    span = te.max() - te.min()
    assert pd.Timedelta(days=27) <= span <= pd.Timedelta(weeks=4)   # ~last 4 weeks
    assert len(train) + len(test) == len(g)


def test_pr_auc_perfect_and_baseline():
    y = [0, 0, 1, 1]
    assert evaluate.pr_auc(y, [0.1, 0.2, 0.8, 0.9]) == 1.0   # perfect ranking
    # all-equal scores -> average precision collapses to the positive rate
    assert evaluate.pr_auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
