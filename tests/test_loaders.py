"""loaders: geo-normalize (#4), Kyiv->UTC + DST, wave/daily shape on real data."""

import pandas as pd
import pytest

from src import config, loaders


def test_normalize_oblast_basic():
    assert loaders.normalize_oblast("Kyiv oblast") == "kyivska"
    assert loaders.normalize_oblast("Kyiv") == "kyiv-city"
    assert loaders.normalize_oblast("Dnipro oblast") == "dnipropetrovska"
    assert loaders.normalize_oblast("Dnipropetrovsk oblast") == "dnipropetrovska"
    assert loaders.normalize_oblast("Kryvyi Rih") == "dnipropetrovska"  # city -> oblast


def test_normalize_oblast_canonical_adm1_names():
    # Sirens file uses full ADM1 names = code + " oblast"; must round-trip to the code.
    assert loaders.normalize_oblast("Cherkaska oblast") == "cherkaska"
    assert loaders.normalize_oblast("Kyivska oblast") == "kyivska"
    assert loaders.normalize_oblast("Ivano-Frankivska oblast") == "ivano-frankivska"
    assert loaders.normalize_oblast("Kyiv City") == "kyiv-city"


def test_normalize_oblast_national_and_unknown():
    assert loaders.normalize_oblast("Ukraine") is None
    assert loaders.normalize_oblast("south") is None
    assert loaders.normalize_oblast(None) is None
    assert loaders.normalize_oblast("") is None


def test_normalize_oblast_codes_are_valid():
    for code in config.OBLAST_ALIASES.values():
        assert code in config.OBLAST_CODES


def test_to_utc_dst_summer_and_winter():
    s = pd.Series(["2026-06-13 18:00", "2023-01-01 12:00"])
    utc = loaders._to_utc(s)
    # EEST (+3) summer: 18:00 -> 15:00 ; EET (+2) winter: 12:00 -> 10:00
    assert utc.iloc[0] == pd.Timestamp("2026-06-13 15:00", tz="UTC")
    assert utc.iloc[1] == pd.Timestamp("2023-01-01 10:00", tz="UTC")


def test_to_utc_date_only_row():
    utc = loaders._to_utc(pd.Series(["2026-06-13"]))
    assert utc.iloc[0] == pd.Timestamp("2026-06-12 21:00", tz="UTC")  # 00:00 EEST


def test_parse_oblast_list():
    cell = "['Kyiv oblast', 'Dnipro oblast', 'Ukraine']"
    assert loaders._parse_oblast_list(cell) == ["dnipropetrovska", "kyivska"]
    assert loaders._parse_oblast_list(None) == []
    assert loaders._parse_oblast_list("not-a-list") == []


# --- real-data smoke (skips if not downloaded) -------------------------------
_WAVES = config.DATA_DIR / "missile_attacks_daily.csv"
_ALERTS = config.DATA_DIR / "official_data_en.csv"


@pytest.mark.skipif(not _ALERTS.exists(), reason="alerts CSV not downloaded")
def test_load_alerts_shape_and_geo():
    a = loaders.load_alerts()
    assert len(a) > 100_000
    assert str(a.start_utc.dt.tz) == "UTC" and str(a.end_utc.dt.tz) == "UTC"
    assert a[["oblast", "start_utc", "end_utc"]].notna().all().all()
    # every oblast rolls up to a valid ADM1 code; no permanent-siren regions present
    assert set(a.oblast) <= config.OBLAST_CODE_SET
    assert {"crimea", "sevastopol"}.isdisjoint(a.oblast)
    assert (a.end_utc >= a.start_utc).all()             # no negative intervals
    assert not a.duplicated(["oblast", "start_utc", "end_utc"]).any()


@pytest.mark.skipif(not _WAVES.exists(), reason="waves CSV not downloaded")
def test_load_massive_attacks_shape():
    w = loaders.load_massive_attacks()
    assert len(w) > 3000
    assert w.time_start_utc.isna().sum() == 0          # mixed-format parse fix
    assert str(w.time_start_utc.dt.tz) == "UTC"
    assert (w.channels.map(len) > 0).all()             # every wave classified (#9)


@pytest.mark.skipif(not _WAVES.exists(), reason="waves CSV not downloaded")
def test_load_missile_daily_gapfree():
    d = loaders.load_missile_daily()
    gaps = d.date_utc.diff().dropna().unique()
    assert list(gaps) == [pd.Timedelta(days=1)]        # no missing calendar days
    assert (d.launched >= 0).all()
