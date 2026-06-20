"""`model` free-text -> threat-type category.

Built and TESTED first (issue #9): combos, Cyrillic/Latin mix, typos, casing.

Categories (issue #8 — decoy is its OWN category):
  ballistic | air-cruise | sea-cruise | drone-strike | drone-decoy | kinzhal

⚠️ Decoy verification (issue #8): confirm `model` actually tags decoys (Gerbera) at
download. If early-war records leave decoys untagged, they fall under drone-strike —
accept and NOTE it; do not silently merge.

Matching strategy: normalize raw text (lowercase, Cyrillic->Latin translit, collapse
separators) then substring-match against an ORDERED pattern list. Order = priority:
specific/short-horizon channels first so e.g. Kinzhal `kh-47` wins over generic `kh-`.
"""

from __future__ import annotations

from collections import Counter

THREAT_TYPES = (
    "ballistic",
    "air-cruise",
    "sea-cruise",
    "drone-strike",
    "drone-decoy",
    "drone-recon",   # data-driven addition: Orlan/ZALA/Supercam etc (non-strike ISR)
    "kinzhal",
)

# Cyrillic (uk + ru) -> Latin. Keeps substring matching translit-agnostic.
# Note: и->y (uk) means ru "кинжал" -> "kynzhal"; patterns include that variant.
_CYR = {
    "а": "a", "б": "b", "в": "v", "г": "h", "ґ": "g", "д": "d", "е": "e",
    "є": "ie", "ж": "zh", "з": "z", "и": "y", "і": "i", "ї": "i", "й": "i",
    "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "у": "u", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch",
    "ш": "sh", "щ": "shch", "ь": "", "ю": "iu", "я": "ia", "э": "e",
    "ы": "y", "ё": "e", "ъ": "",
}

# Ordered (pattern, category). First match by priority wins for `classify`;
# `classify_all` collects every hit (combos like "X-101/X-555 and Kalibr").
# Patterns are NORMALIZED substrings. normalize() keeps Latin "x" as-is but maps
# Cyrillic "х"->"kh" and "с"->"s", so missile codes need BOTH variants:
#   real data has Latin "X-101" / "C-300" AND Cyrillic "Х-101" / "С-300".
# Priority order = shortest-warning / highest-threat first (matters for single classify).
_PATTERNS: list[tuple[str, str]] = [
    # --- kinzhal (air-launched ballistic, own short-horizon channel) — before generic ---
    ("kinzhal", "kinzhal"),
    ("kynzhal", "kinzhal"),
    ("kindzhal", "kinzhal"),
    ("x-47", "kinzhal"),       # "X-47 Kinzhal" (Latin)
    ("kh-47", "kinzhal"),      # Cyrillic "Х-47"
    # --- ballistic (ground-launched + SAM-in-ballistic-mode + Iskander-M/KN-23) ---
    ("iskander-m", "ballistic"),
    ("9m723", "ballistic"),
    ("9m72", "ballistic"),
    ("kn-23", "ballistic"),
    ("kn23", "ballistic"),
    ("tochka", "ballistic"),
    ("ballistic", "ballistic"),
    ("intercontinental", "ballistic"),
    ("icbm", "ballistic"),
    ("c-300", "ballistic"),    # Latin "C-300" (data uses Latin C); ground-attack = ballistic-like
    ("c-400", "ballistic"),
    ("s-300", "ballistic"),    # Cyrillic "С-300" -> s-300
    ("s-400", "ballistic"),
    # --- sea-cruise (sea-launched) ---
    ("kalibr", "sea-cruise"),
    ("kalib", "sea-cruise"),
    ("3m14", "sea-cruise"),
    ("oniks", "sea-cruise"),
    ("onyks", "sea-cruise"),
    ("zircon", "sea-cruise"),
    ("zirkon", "sea-cruise"),
    ("3m22", "sea-cruise"),
    # --- air-cruise (air/ground cruise, long horizon). Iskander-K (ground cruise) lumped here ---
    ("iskander-k", "air-cruise"),
    ("iskander", "ballistic"),   # bare/typo Iskander defaults to ballistic (-M most common)
    ("iskandr", "ballistic"),
    ("x-101", "air-cruise"),
    ("kh-101", "air-cruise"),
    ("x-555", "air-cruise"),
    ("kh-555", "air-cruise"),
    ("x-55", "air-cruise"),
    ("kh-55", "air-cruise"),
    ("x-59", "air-cruise"),
    ("kh-59", "air-cruise"),
    ("x-69", "air-cruise"),
    ("kh-69", "air-cruise"),
    ("x-22", "air-cruise"),
    ("kh-22", "air-cruise"),
    ("x-32", "air-cruise"),
    ("kh-32", "air-cruise"),
    ("x-31", "air-cruise"),
    ("kh-31", "air-cruise"),
    ("x-35", "air-cruise"),
    ("kh-35", "air-cruise"),
    ("banderol", "air-cruise"),
    ("gbu", "air-cruise"),     # guided glide bomb, air-launched standoff (rare)
    ("kab", "air-cruise"),
    ("unknown missile", "air-cruise"),  # ref lists it under cruise
    ("aerial bomb", "air-cruise"),      # guided/aerial bomb, air-launched standoff
    ("bomb", "air-cruise"),
    # --- drone-decoy (issue #8 — separate channel; absent in current data, kept) ---
    ("gerbera", "drone-decoy"),
    ("herbera", "drone-decoy"),
    ("decoy", "drone-decoy"),
    ("imitator", "drone-decoy"),
    # --- drone-recon (ISR UAVs — non-strike; before drone-strike to avoid mislabel) ---
    ("orlan", "drone-recon"),
    ("zala", "drone-recon"),
    ("supercam", "drone-recon"),
    ("eleron", "drone-recon"),
    ("forpost", "drone-recon"),
    ("granat", "drone-recon"),
    ("merlin", "drone-recon"),
    ("mohajer", "drone-recon"),
    ("orion", "drone-recon"),
    ("reconnaissance", "drone-recon"),
    ("feniks", "drone-recon"),     # Фенікс (Phoenix ISR UAV)
    ("kartohraf", "drone-recon"),  # Картограф (mapper ISR UAV)
    # --- drone-strike (loitering munitions + unknown UAV, conservative) ---
    ("shahed", "drone-strike"),
    ("shahid", "drone-strike"),
    ("shakhed", "drone-strike"),   # Cyrillic Шахед -> х=kh translit
    ("shakhid", "drone-strike"),   # Cyrillic Шахід (uk)
    ("geran", "drone-strike"),
    ("heran", "drone-strike"),
    ("lancet", "drone-strike"),
    ("lantset", "drone-strike"),
    ("molniia", "drone-strike"),   # Молнія -> molniia
    ("molnia", "drone-strike"),
    ("kub", "drone-strike"),
    ("pryvet", "drone-strike"),    # Привет-82 (jet decoy/strike drone)
    ("privet", "drone-strike"),
    ("unknown uav", "drone-strike"),
]

# Counts raw strings that matched nothing — surfaced, never silently dropped.
UNMATCHED: Counter = Counter()


def normalize(text: str) -> str:
    """Lowercase, Cyrillic->Latin translit, drop noise chars. Idempotent on Latin."""
    out = []
    for ch in str(text).lower():
        if ch in _CYR:
            out.append(_CYR[ch])
        elif ch.isalnum() or ch in "-/+ ":
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def classify(model_raw: str) -> str | None:
    """Map one raw `model` string to a threat type, or None if unmatched.

    First pattern (priority order) found as a substring wins. Unmatched inputs are
    counted in UNMATCHED so the caller can audit coverage (issue #9), not dropped silently.
    """
    norm = normalize(model_raw)
    for pat, cat in _PATTERNS:
        if pat in norm:
            return cat
    if str(model_raw).strip():
        UNMATCHED[str(model_raw).strip()] += 1
    return None


def classify_all(model_raw: str) -> set[str]:
    """All threat types present in a combo string (e.g. 'Shahed-136 / Gerbera').

    Splits on common separators and classifies each token; returns the set of hits.
    Use for wave records that list multiple munitions in one `model` field.
    """
    norm = normalize(model_raw)
    hits = {cat for pat, cat in _PATTERNS if pat in norm}
    return hits
