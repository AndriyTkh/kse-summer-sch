"""model_onset: onset target = NEW alert starts in (t,t+H], from a quiet state only."""

import numpy as np
import pandas as pd

from src import index, model_onset


def _grid(start, end, oblasts):
    g = index.build_master_index(start=start, end=end, oblasts=oblasts)
    g["alert"] = 0
    return g


def test_onset_target_rising_edge_in_window():
    # Single alert block at 04:00-05:00 -> exactly one onset (rising edge) at 04:00.
    g = _grid("2024-05-01 00:00", "2024-05-01 08:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([4, 5]), "alert"] = 1
    y = model_onset.make_onset_target(g, "3h").xs("kyivska", level="oblast")

    # window (t, t+3h] reaches the 04:00 onset from t = 01:00, 02:00, 03:00.
    assert y.loc["2024-05-01 01:00+00:00"] == 1
    assert y.loc["2024-05-01 03:00+00:00"] == 1
    assert y.loc["2024-05-01 00:00+00:00"] == 0   # onset is 4h ahead, outside window
    assert y.loc["2024-05-01 06:00+00:00"] == 0   # no further onset ahead


def test_onset_excludes_active_state_rows():
    # Rows with an alert ACTIVE at t are NaN (quiet-state-only evaluation).
    g = _grid("2024-05-01 00:00", "2024-05-01 08:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([4, 5]), "alert"] = 1
    y = model_onset.make_onset_target(g, "3h").xs("kyivska", level="oblast")

    assert pd.isna(y.loc["2024-05-01 04:00+00:00"])
    assert pd.isna(y.loc["2024-05-01 05:00+00:00"])


def test_onset_ignores_continuation_no_new_edge():
    # A sustained alert (4,5,6) has ONE onset at 4, not three; continuation isn't onset.
    g = _grid("2024-05-01 00:00", "2024-05-01 10:00", ["kyivska"])
    ts = g.index.get_level_values("ts_utc")
    g.loc[ts.hour.isin([4, 5, 6]), "alert"] = 1
    y = model_onset.make_onset_target(g, "1h").xs("kyivska", level="oblast")

    # 1h window = next hour only. Onset (at 4) is "next" only from t = 03:00.
    assert y.loc["2024-05-01 03:00+00:00"] == 1
    # From the quiet hours 07:00+ there is no future onset -> 0 (5,6 are continuation).
    assert y.loc["2024-05-01 07:00+00:00"] == 0


def test_onset_tail_is_nan():
    g = _grid("2024-05-01 00:00", "2024-05-01 06:00", ["kyivska"])
    y = model_onset.make_onset_target(g, "1h").xs("kyivska", level="oblast")
    assert pd.isna(y.loc["2024-05-01 06:00+00:00"])   # no future window


def test_onset_train_predict_smoke():
    g = _grid("2024-05-01 00:00", "2024-05-08 23:00", ["kyivska", "lvivska"])
    ts = g.index.get_level_values("ts_utc")
    # Onset every day at 02:00 (preceded by quiet 01:00) -> a learnable hour signal.
    g.loc[ts.hour.isin([2, 3]), "alert"] = 1
    X = g.copy()
    X["hour"] = ts.hour
    X["sig"] = (ts.hour + 1).astype(float)

    models = model_onset.train_all(X, g, horizons=["1h", "6h"])
    assert set(models) == {"1h", "6h"}
    probs = model_onset.predict_all(models, X)
    assert list(probs.columns) == ["1h", "6h"]
    assert ((probs.values >= 0) & (probs.values <= 1)).all()
