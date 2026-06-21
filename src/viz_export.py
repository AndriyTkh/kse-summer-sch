"""Shared helpers to serialize Phase-2 outputs as JSON for the viz dashboard.

Each run_*.py script computes its slice and calls `write_viz_json` to drop a file
into viz/public/. Split on purpose: the heavy rolling-origin / degradation sweeps
stay out of the light export_predictions.py refresh (see README run options).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

VIZ_DIR = config.ROOT / "viz" / "public"


def serializable(v):
    """JSON-safe scalar: round floats, cast numpy/pandas, ISO timestamps."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if (f != f) else round(f, 6)  # NaN -> null
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    return v


def write_viz_json(name: str, obj: dict) -> None:
    """Write `obj` to viz/public/<name> (pretty), creating the dir if needed."""
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    (VIZ_DIR / name).write_text(json.dumps(obj, indent=2, default=serializable))
    print(f"viz -> {VIZ_DIR / name}")
