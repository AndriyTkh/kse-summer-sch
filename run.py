"""One-shot driver: install deps -> compute all dashboard data -> launch the viz.

Replaces the fistful of run_*.py scripts that used to sit at the repo root. Those
partials still exist under scripts/runs/ (runnable standalone, see their docstrings);
this driver just runs the subset the dashboard actually needs, back-to-back, then
serves the React app.

Walk-forward CV is intentionally EXCLUDED from the one-go run (dropped 2026-06-21:
dominant compute cost for <=0.005 PR-AUC — recency weighting already absorbs drift).
Run it by hand if needed:  python scripts/runs/run_walkforward.py

Usage:
    python run.py                 # install deps + compute + launch dashboard
    python run.py --no-install    # skip the pip/npm install step
    python run.py --no-viz        # compute the JSON only, don't launch the dashboard
    python run.py --skip-compute  # (install +) just launch the dashboard on existing JSON
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VIZ = ROOT / "viz"

# Compute steps run in order. Each is (label, command). export_predictions writes the
# headline predictions.json/metrics.json/onset.json; the partials each emit one more viz
# JSON. Walk-forward is deliberately absent (see module docstring).
COMPUTE_STEPS = [
    ("predictions + metrics + onset", [sys.executable, "-m", "src.export_predictions"]),
    ("nowcast (operational edge)",    [sys.executable, "scripts/runs/run_forecast.py"]),
    ("operational backtest-vs-live",  [sys.executable, "scripts/runs/run_operational_eval.py"]),
    ("survival (alert duration)",     [sys.executable, "scripts/runs/run_survival.py"]),
    ("quantile intervals (Bq)",       [sys.executable, "scripts/runs/run_phase3.py"]),
]


def run(cmd, cwd=ROOT, env=None) -> None:
    """Run a subprocess, streaming its output; raise on non-zero exit."""
    printable = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def npm_cmd() -> str:
    """npm executable name — npm.cmd on Windows, npm elsewhere."""
    return "npm.cmd" if os.name == "nt" else "npm"


def install() -> None:
    print("=== install: python deps ===")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("\n=== install: viz (npm) ===")
    npm = shutil.which(npm_cmd())
    if npm is None:
        print("  npm not found on PATH — skipping viz install. Install Node.js to build the dashboard.")
        return
    run([npm, "install"], cwd=VIZ)


def compute() -> None:
    # PYTHONUTF8 keeps the unicode in logs/JSON happy on Windows consoles.
    env = {**os.environ, "PYTHONUTF8": "1"}
    t0 = time.time()
    for label, cmd in COMPUTE_STEPS:
        print(f"\n=== compute: {label} ===")
        run(cmd, env=env)
    print(f"\n=== compute done: {len(COMPUTE_STEPS)} steps, viz/public/*.json refreshed "
          f"[{time.time()-t0:.0f}s] ===")


def serve() -> None:
    npm = shutil.which(npm_cmd())
    if npm is None:
        print("npm not found on PATH — cannot launch the dashboard. Install Node.js, then "
              "run:  cd viz && npm run dev")
        return
    print("\n=== launch: viz dev server (Ctrl-C to stop) ===")
    run([npm, "run", "dev"], cwd=VIZ)


def main() -> None:
    ap = argparse.ArgumentParser(description="Install, compute dashboard data, and launch the viz.")
    ap.add_argument("--no-install", action="store_true", help="skip the pip/npm install step")
    ap.add_argument("--no-viz", action="store_true", help="compute the JSON only; don't launch the dashboard")
    ap.add_argument("--skip-compute", action="store_true", help="don't recompute; just launch on existing JSON")
    args = ap.parse_args()

    if not args.no_install:
        install()
    if not args.skip_compute:
        compute()
    if not args.no_viz:
        serve()


if __name__ == "__main__":
    main()
