"""Refresh raw datasets in data/ to their latest published versions.

Sources & how each is fetched:
  - alerts   (official_data_en.csv)   Vadimkin GitHub raw mirror   -> urllib, no auth
  - missiles (missile_attacks_daily.csv + missiles_and_uavs.csv)
                                       piterfm Kaggle dataset       -> kaggle API, NEEDS creds
  - ucdp     (ged251_ukraine.csv)      UCDP GED annual release      -> STATIC, not auto-updated

Safe-swap: download to a temp file, sanity-check the header, back up the old file to
<name>.bak, then replace. Run scripts/data_freshness.py after to see the new lag.

Usage:
    .venv/Scripts/python.exe scripts/update_data.py            # all available sources
    .venv/Scripts/python.exe scripts/update_data.py alerts     # one source
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from project .env into os.environ (no overwrite)."""
    import os

    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))

ALERTS_URL = (
    "https://raw.githubusercontent.com/Vadimkin/"
    "ukrainian-air-raid-sirens-dataset/master/datasets/official_data_en.csv"
)
KAGGLE_MISSILES = "piterfm/massive-missile-attacks-on-ukraine"
MISSILE_FILES = ("missile_attacks_daily.csv", "missiles_and_uavs.csv")


def _swap(dest: Path, tmp: Path, expect_header: str | None) -> None:
    """Validate tmp, back up dest -> .bak, move tmp into place."""
    head = tmp.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
    if expect_header and (not head or expect_header not in head[0]):
        raise ValueError(f"{dest.name}: unexpected header {head!r}")
    if dest.exists():
        shutil.copy2(dest, dest.with_suffix(dest.suffix + ".bak"))
    shutil.move(str(tmp), str(dest))
    print(f"  updated {dest.name} ({dest.stat().st_size/1e6:.2f} MB)")


def update_alerts() -> None:
    print("alerts   <- Vadimkin GitHub raw")
    dest = DATA / "official_data_en.csv"
    fd, tmp = tempfile.mkstemp(suffix=".csv")
    try:
        with urllib.request.urlopen(ALERTS_URL, timeout=120) as r, open(fd, "wb") as f:
            shutil.copyfileobj(r, f)
        _swap(dest, Path(tmp), expect_header="started_at")
    finally:
        Path(tmp).unlink(missing_ok=True)


def _has_kaggle_creds() -> bool:
    import os

    if os.getenv("KAGGLE_KEY") and os.getenv("KAGGLE_USERNAME"):
        return True
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    home = Path(os.path.expanduser("~"))
    return (home / ".kaggle" / "kaggle.json").exists() or (
        home / ".kaggle" / "access_token"
    ).exists()


def update_missiles() -> None:
    print(f"missiles <- Kaggle {KAGGLE_MISSILES}")
    # `import kaggle` calls sys.exit(1) on missing creds (SystemExit, not Exception),
    # so check creds up front rather than guarding the import.
    if not _has_kaggle_creds():
        print("  SKIP: no Kaggle credentials found.")
        print("  -> put kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME/KAGGLE_KEY), then rerun.")
        print(f"  -> manual: https://www.kaggle.com/datasets/{KAGGLE_MISSILES}")
        return
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "datasets", "download",
             "-d", KAGGLE_MISSILES, "-p", td, "--unzip"],
            check=True,
        )
        for name in MISSILE_FILES:
            src = Path(td) / name
            if not src.exists():
                print(f"  WARN: {name} not in download")
                continue
            tmp = Path(td) / (name + ".tmp")
            shutil.copy2(src, tmp)
            _swap(DATA / name, tmp, expect_header=None)


def update_ucdp() -> None:
    print("ucdp     <- STATIC annual release (ged251 = UCDP GED v25.1, through 2024)")
    print("  no auto-update: next GED is an annual drop (~mid-year). For sub-annual,")
    print("  swap to the UCDP Candidate monthly dataset (roadmap, not wired).")


JOBS = {"alerts": update_alerts, "missiles": update_missiles, "ucdp": update_ucdp}


def main(argv: list[str]) -> None:
    _load_dotenv()
    targets = argv or list(JOBS)
    for t in targets:
        if t not in JOBS:
            print(f"unknown source {t!r}; choose from {list(JOBS)}")
            continue
        try:
            JOBS[t]()
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR updating {t}: {e}")
    print("\ndone. run: .venv/Scripts/python.exe scripts/data_freshness.py")


if __name__ == "__main__":
    main(sys.argv[1:])
