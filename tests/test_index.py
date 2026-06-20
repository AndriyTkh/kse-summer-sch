"""index grid + leak-guard joins (issue #1): verified on known dates."""

import pandas as pd
import pytest

from src import config, index


def test_grid_shape_and_tz():
    obl = ["kyiv-city", "lvivska"]
    g = index.build_master_index(
        start="2022-02-24", end="2022-02-24 05:00", oblasts=obl
    )
    # 6 hours (00..05) x 2 oblasts
    assert len(g) == 6 * 2
    ts = g.index.get_level_values("ts_utc")
    assert str(ts.tz) == "UTC"
    assert g.index.names == ["oblast", "ts_utc"]


def test_grid_requires_oblasts():
    with pytest.raises(ValueError):
        index.build_master_index(end="2022-02-24 01:00", oblasts=[])


def _grid():
    return index.build_master_index(
        start="2022-03-01 00:00", end="2022-03-01 05:00", oblasts=["kyivska"]
    )


def test_expand_alerts_overlap():
    g = _grid()
    alerts = pd.DataFrame({
        "oblast": ["kyivska"],
        # spans 01:30 -> 03:10 -> touches hours 01, 02, 03
        "start_utc": [pd.Timestamp("2022-03-01 01:30", tz="UTC")],
        "end_utc": [pd.Timestamp("2022-03-01 03:10", tz="UTC")],
    })
    out = index.expand_alerts_to_grid(g, alerts)
    on = out[out["alert"] == 1].index.get_level_values("ts_utc").hour.tolist()
    assert sorted(on) == [1, 2, 3]


def test_expand_alerts_edge_boundaries():
    g = _grid()
    alerts = pd.DataFrame({
        "oblast": ["kyivska", "kyivska"],
        # exact-hour interval [02:00,03:00) -> only cell 02; zero-length at 04:00 -> nothing
        "start_utc": [pd.Timestamp("2022-03-01 02:00", tz="UTC"),
                      pd.Timestamp("2022-03-01 04:00", tz="UTC")],
        "end_utc": [pd.Timestamp("2022-03-01 03:00", tz="UTC"),
                    pd.Timestamp("2022-03-01 04:00", tz="UTC")],
    })
    out = index.expand_alerts_to_grid(g, alerts)
    on = out[out["alert"] == 1].index.get_level_values("ts_utc").hour.tolist()
    assert on == [2]


def test_expand_alerts_unknown_oblast_ignored():
    g = _grid()
    alerts = pd.DataFrame({
        "oblast": ["lvivska"],   # not in this grid
        "start_utc": [pd.Timestamp("2022-03-01 01:00", tz="UTC")],
        "end_utc": [pd.Timestamp("2022-03-01 02:00", tz="UTC")],
    })
    out = index.expand_alerts_to_grid(g, alerts)
    assert (out["alert"] == 0).all()


def test_expand_alerts_empty():
    g = _grid()
    out = index.expand_alerts_to_grid(g, alerts=pd.DataFrame(
        columns=["oblast", "start_utc", "end_utc"]))
    assert (out["alert"] == 0).all()


def test_asof_strict_less_than_no_leak():
    """A source row at exactly t must NOT attach to grid row t (strict <)."""
    g = _grid()
    src = pd.DataFrame({
        "ts": [pd.Timestamp("2022-03-01 02:00", tz="UTC")],
        "val": [99],
    })
    out = index.asof_join(g, src, on="ts", lag="0h")
    at_2 = out.xs(pd.Timestamp("2022-03-01 02:00", tz="UTC"), level="ts_utc")["val"]
    at_3 = out.xs(pd.Timestamp("2022-03-01 03:00", tz="UTC"), level="ts_utc")["val"]
    assert at_2.isna().all()      # same-timestamp does NOT leak
    assert (at_3 == 99).all()     # next hour sees it


def test_asof_lag_shifts_window():
    """With a 7-day lag, a row dated t attaches only >= t+7d."""
    g = index.build_master_index(
        start="2022-03-01", end="2022-03-10", oblasts=["kyivska"]
    )
    src = pd.DataFrame({
        "ts": [pd.Timestamp("2022-03-01 00:00", tz="UTC")],
        "impact": [5],
    })
    out = index.asof_join(g, src, on="ts", lag=f"{config.UCDP_LAG_DAYS}D")
    before = out.xs(pd.Timestamp("2022-03-07 00:00", tz="UTC"), level="ts_utc")["impact"]
    after = out.xs(pd.Timestamp("2022-03-08 06:00", tz="UTC"), level="ts_utc")["impact"]
    assert before.isna().all()    # < t+7d : not yet visible
    assert (after == 5).all()     # >= t+7d : visible


def test_asof_by_group_isolation():
    g = index.build_master_index(
        start="2022-03-01 00:00", end="2022-03-01 03:00",
        oblasts=["kyivska", "lvivska"],
    )
    src = pd.DataFrame({
        "ts": [pd.Timestamp("2022-03-01 00:00", tz="UTC")],
        "oblast": ["kyivska"],
        "val": [7],
    })
    out = index.asof_join(g, src, on="ts", lag="0h", by="oblast")
    kyiv = out.xs("kyivska", level="oblast")["val"]
    lviv = out.xs("lvivska", level="oblast")["val"]
    assert (kyiv.dropna() == 7).all()
    assert lviv.isna().all()      # group isolation: lvivska sees nothing
