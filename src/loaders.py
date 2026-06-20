"""Raw-source loaders -> tidy DataFrames in UTC.

Sources (all bulk/CSV/API — no scraping in MVP):
  - Vadimkin air-raid-sirens-dataset    -> alert on/off per oblast (TARGET)   [not yet downloaded]
  - piterfm Kaggle dataset:
      missile_attacks_daily.csv         -> wave-level launches (LAUNCH + TEMPO)
      missiles_and_uavs.csv             -> model->category ref (used by threat_map)

NOTE (data reality): the Kaggle dataset ships ONE wave-level file (despite the
"daily" name) plus a model reference table — not the 3 separate files STRUCTURE
assumed. `load_massive_attacks` reads the wave file at row grain; `load_missile_daily`
aggregates the same file to national daily tempo.

Each loader: read raw -> parse timestamps -> convert Kyiv-local -> UTC ->
normalize oblast to ADM1 codes. No feature logic here.
"""

from __future__ import annotations

import ast

import pandas as pd

from . import config, threat_map

# Kaggle source timestamps are Kyiv-local wall-clock.
_SRC_TZ = config.TZ_LOCAL


def normalize_oblast(name: str) -> str | None:
    """Map a raw region string to an ADM1 code (issue #4), or None if national/unknown.

    Lowercases, strips a trailing ' oblast', then looks up config.OBLAST_ALIASES.
    'Ukraine' and anything unmapped -> None (caller decides drop vs national-broadcast).
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("", "ukraine"):
        return None
    if key in config.OBLAST_ALIASES:
        return config.OBLAST_ALIASES[key]
    key = key.removesuffix(" oblast").strip()
    return config.OBLAST_ALIASES.get(key)


def _to_utc(series: pd.Series) -> pd.Series:
    """Parse Kyiv-local wall-clock strings -> tz-aware UTC. DST-safe.

    Mixed 'YYYY-MM-DD HH:MM' and date-only rows both parse (date-only -> 00:00).
    Ambiguous/nonexistent DST instants are pushed forward rather than dropped.
    """
    # format="mixed": parse each element independently — the file mixes
    # 'YYYY-MM-DD HH:MM' with date-only rows (pandas 2.x else infers one format).
    naive = pd.to_datetime(series, format="mixed", errors="coerce")
    local = naive.dt.tz_localize(
        _SRC_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    return local.dt.tz_convert(config.TZ_GRID)


def _parse_oblast_list(cell) -> list[str]:
    """`affected region` is a stringified Python list -> list of ADM1 codes (deduped)."""
    if pd.isna(cell):
        return []
    try:
        names = ast.literal_eval(cell)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(names, (list, tuple)):
        names = [names]
    codes = {c for c in (normalize_oblast(n) for n in names) if c}
    return sorted(codes)


def load_alerts(path=None) -> pd.DataFrame:
    """Vadimkin sirens -> tidy alert intervals [oblast, start_utc, end_utc].

    Not yet downloaded (public GitHub). Implement when the file lands; downstream
    `index.expand_alerts_to_grid` turns these intervals into the hourly target.
    """
    raise NotImplementedError("Vadimkin sirens dataset not downloaded yet")


def load_massive_attacks(path=None) -> pd.DataFrame:
    """Wave-level launch records, tidy + UTC.

    Returns one row per wave:
      [time_start_utc, time_end_utc, model, launch_place, launched, destroyed,
       is_shahed, oblasts (list[ADM1]), channels (set[threat-type])]
    Threat channels via threat_map.classify_all (combo-aware). Geo via affected-region
    list (issue #4); national-only waves -> empty oblasts list (broadcast in features).
    """
    path = path or config.DATA_DIR / "missile_attacks_daily.csv"
    df = pd.read_csv(path)

    out = pd.DataFrame({
        "time_start_utc": _to_utc(df["time_start"]),
        "time_end_utc": _to_utc(df["time_end"]),
        "model": df["model"],
        "launch_place": df.get("launch_place"),
        "launched": pd.to_numeric(df.get("launched"), errors="coerce"),
        "destroyed": pd.to_numeric(df.get("destroyed"), errors="coerce"),
        "is_shahed": df.get("is_shahed"),
    })
    out["oblasts"] = df["affected region"].map(_parse_oblast_list)
    out["channels"] = out["model"].map(threat_map.classify_all)
    out = out.dropna(subset=["time_start_utc"]).reset_index(drop=True)
    return out


def load_missile_daily(path=None) -> pd.DataFrame:
    """National daily tempo, aggregated from the wave file (feeds A baseline).

    Returns [date_utc, launched, destroyed, n_waves] indexed by UTC calendar day.
    """
    waves = load_massive_attacks(path)
    waves = waves.assign(date_utc=waves["time_start_utc"].dt.floor("D"))
    daily = waves.groupby("date_utc").agg(
        launched=("launched", "sum"),
        destroyed=("destroyed", "sum"),
        n_waves=("model", "size"),
    )
    # Reindex to a gap-free daily calendar; missing day = 0 launches (issue #3).
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D", tz=config.TZ_GRID)
    return daily.reindex(full, fill_value=0).rename_axis("date_utc").reset_index()


def load_acled(path=None) -> pd.DataFrame:
    """ACLED geolocated strikes -> per-oblast impact events.

    Skipped in current MVP run (registration limits). features.add_acled_features
    stays a no-op until this is wired. Join MUST use ACLED_LAG_DAYS (issue #2).
    """
    raise NotImplementedError("ACLED skipped (registration limits)")
