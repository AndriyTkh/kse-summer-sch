"""threat_map coverage (issue #9): Latin, Cyrillic, combos, typos, casing, decoy split."""

from pathlib import Path

import pytest

from src import config, threat_map as tm


@pytest.mark.parametrize("raw,expected", [
    ("Shahed-136", "drone-strike"),
    ("Geran-2", "drone-strike"),
    ("Gerbera", "drone-decoy"),          # issue #8: decoy is its own channel
    ("Kh-101", "air-cruise"),
    ("X-101/X-555", "air-cruise"),       # real data: Latin X- prefix
    ("X-59", "air-cruise"),
    ("Iskander-K", "air-cruise"),        # ground cruise, NOT ballistic (the -K)
    ("Kalibr 3M14", "sea-cruise"),
    ("3M22 Zircon", "sea-cruise"),
    ("Iskander-M", "ballistic"),
    ("9M723", "ballistic"),
    ("C-300", "ballistic"),              # real data: Latin C-300, ground-attack
    ("Kinzhal", "kinzhal"),
    ("X-47 Kinzhal", "kinzhal"),         # Kinzhal wins over generic cruise X-
    ("Orlan-10", "drone-recon"),         # ISR UAV, not strike
    ("ZALA", "drone-recon"),
    ("Lancet", "drone-strike"),
])
def test_latin(raw, expected):
    assert tm.classify(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Шахед", "drone-strike"),
    ("Шахід-136", "drone-strike"),       # uk і spelling
    ("Калібр", "sea-cruise"),
    ("Іскандер", "ballistic"),
    ("Х-101", "air-cruise"),             # Cyrillic Х -> kh
    ("Кинжал", "kinzhal"),               # ru и -> y variant covered
])
def test_cyrillic(raw, expected):
    assert tm.classify(raw) == expected


def test_casing_and_whitespace():
    assert tm.classify("  shAHed-136  ") == "drone-strike"


def test_typo_variants():
    assert tm.classify("Iskandr") == "ballistic"
    assert tm.classify("herbera") == "drone-decoy"


def test_decoy_not_merged_into_strike():
    assert tm.classify("Gerbera") == "drone-decoy"
    assert tm.classify("Shahed-136") == "drone-strike"
    assert tm.classify("Gerbera") != tm.classify("Shahed-136")


def test_combo_classify_all():
    hits = tm.classify_all("Shahed-136 / Gerbera + Kh-101")
    assert hits == {"drone-strike", "drone-decoy", "air-cruise"}


def test_unmatched_returns_none_and_counted():
    tm.UNMATCHED.clear()
    assert tm.classify("definitely-not-a-weapon") is None
    assert tm.UNMATCHED["definitely-not-a-weapon"] == 1


def test_every_pattern_maps_to_valid_type():
    for _pat, cat in tm._PATTERNS:
        assert cat in tm.THREAT_TYPES


# --- coverage against the real downloaded dataset (skips if data absent) -------
_REF = config.DATA_DIR / "missiles_and_uavs.csv"
_WAVES = config.DATA_DIR / "missile_attacks_daily.csv"


@pytest.mark.skipif(not _REF.exists(), reason="ref CSV not downloaded")
def test_ref_table_fully_covered():
    import pandas as pd
    ref = pd.read_csv(_REF)
    unmatched = [m for m in ref.model if tm.classify(m) is None]
    assert unmatched == [], f"unmapped ref models: {unmatched}"


@pytest.mark.skipif(not _WAVES.exists(), reason="waves CSV not downloaded")
def test_wave_models_fully_covered():
    import pandas as pd
    w = pd.read_csv(_WAVES)
    unmatched = [m for m in w.model.dropna().unique() if not tm.classify_all(m)]
    assert unmatched == [], f"unmapped wave models: {unmatched}"
