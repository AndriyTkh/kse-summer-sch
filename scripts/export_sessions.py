#!/usr/bin/env python
"""Export local Claude Code session logs for this repo into `_sessions/`.

Claude Code stores every conversation as a `.jsonl` file under
`~/.claude/projects/<slugified-cwd>/`. Those files are raw event streams
(tool calls, file snapshots, hook noise) and are far too long / messy to hand
to a reviewer. This script:

  1. Locates the project's session directory (matched by `cwd`, not a fragile
     slug guess).
  2. Renders each session to a readable Markdown transcript in `_sessions/`.
  3. Ranks every session by how much *human* steering it contains (typed
     prompts + characters) and writes `_sessions/INDEX.md`.

The ranking exists so you can eyeball which conversations are the real
architecture/decision drivers (high human input) vs. routine grind, and move
the important ones into `_sessions/important/` by hand.

Usage:
    python scripts/export_sessions.py            # export everything
    python scripts/export_sessions.py --list     # just print the ranking
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "_sessions"
PROJECTS_ROOT = Path.home() / ".claude" / "projects"

# tool results / file dumps get clipped so the log stays readable
TOOL_RESULT_CLIP = 600

# Secrets that must never reach a public transcript. Applied to EVERY rendered
# string (human prompts, assistant text, tool results). Add patterns as needed.
SCRUB_PATTERNS = [
    re.compile(r"KGAT_[A-Za-z0-9]+"),                       # Kaggle API token
    re.compile(r"(KAGGLE_KEY\s*[=:]\s*)\S+"),               # KAGGLE_KEY=...
    re.compile(r"(KAGGLE_API_TOKEN\s*[=:]\s*)\S+"),         # KAGGLE_API_TOKEN=...
    re.compile(r"(api[_-]?key\s*[=:]\s*)\S+", re.I),        # generic api_key=...
    re.compile(r"(password\s*=\s*)\"[^\"]+\""),             # password="..."
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),          # GitHub PATs
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),                 # OpenAI-style keys
]


def scrub(text: str) -> str:
    """Strip known secret shapes out of a transcript string."""
    if not text:
        return text
    for pat in SCRUB_PATTERNS:
        if pat.groups:
            text = pat.sub(lambda m: m.group(1) + "[REDACTED]", text)
        else:
            text = pat.sub("[REDACTED]", text)
    return text


def find_session_dir() -> Path:
    """Return the ~/.claude/projects subdir whose sessions ran in this repo.

    Matches on the `cwd` recorded inside the events (robust), falling back to a
    slug of the repo path if no events carry a cwd.
    """
    repo = str(REPO_ROOT).replace("\\", "/").rstrip("/").lower()
    if PROJECTS_ROOT.exists():
        for d in PROJECTS_ROOT.iterdir():
            if not d.is_dir():
                continue
            for jf in d.glob("*.jsonl"):
                cwd = _first_cwd(jf)
                if cwd and cwd.replace("\\", "/").rstrip("/").lower() == repo:
                    return d
                break  # one probe per dir is enough
    # fallback: slugify the repo path the way Claude Code does
    slug = re.sub(r"[^a-zA-Z0-9]", "-", str(REPO_ROOT))
    cand = PROJECTS_ROOT / slug
    if cand.exists():
        return cand
    raise SystemExit(f"No session dir found under {PROJECTS_ROOT} for {REPO_ROOT}")


def _first_cwd(jf: Path) -> str | None:
    try:
        with jf.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("cwd"):
                    return d["cwd"]
    except OSError:
        return None
    return None


def _text_items(content) -> list[str]:
    """Pull plain text out of a message content (str or list of blocks)."""
    if isinstance(content, str):
        return [content]
    out = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                out.append(c.get("text", ""))
    return out


def _strip_injected(text: str) -> str:
    """Remove harness-injected blocks so 'human input' reflects real typing."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.S)
    text = re.sub(r"<ide_selection>.*?</ide_selection>", "", text, flags=re.S)
    text = re.sub(r"<ide_opened_file>.*?</ide_opened_file>", "", text, flags=re.S)
    return text.strip()


def parse_session(jf: Path) -> dict:
    """Parse one .jsonl into ordered render events + human-input stats."""
    events: list[dict] = []
    title = None
    human_prompts = 0
    human_chars = 0
    assistant_msgs = 0
    tool_calls = 0
    first_prompt = None
    ts_start = ts_end = None

    with jf.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            ts = d.get("timestamp")
            if ts:
                ts_start = ts_start or ts
                ts_end = ts
            if t == "ai-title":
                title = d.get("aiTitle")
            elif t == "user":
                msg = d.get("message", {})
                content = msg.get("content")
                # a real typed prompt is flagged with promptSource; everything
                # else under role=user is a tool_result echoed back to the model
                if d.get("promptSource"):
                    human = "\n".join(_text_items(content))
                    human = _strip_injected(human)
                    if human:
                        human_prompts += 1
                        human_chars += len(human)
                        first_prompt = first_prompt or human
                        events.append({"role": "user", "text": human})
                else:
                    for c in content if isinstance(content, list) else []:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            events.append({"role": "tool_result", "block": c})
            elif t == "assistant":
                msg = d.get("message", {})
                content = msg.get("content", [])
                rendered = []
                had_text = False
                for c in content if isinstance(content, list) else []:
                    ct = c.get("type")
                    if ct == "text" and c.get("text", "").strip():
                        rendered.append({"kind": "text", "text": c["text"]})
                        had_text = True
                    elif ct == "thinking" and c.get("thinking", "").strip():
                        rendered.append({"kind": "thinking", "text": c["thinking"]})
                    elif ct == "tool_use":
                        tool_calls += 1
                        rendered.append({"kind": "tool_use", "name": c.get("name"),
                                         "input": c.get("input", {})})
                if rendered:
                    if had_text:
                        assistant_msgs += 1
                    events.append({"role": "assistant", "blocks": rendered})

    return {
        "file": jf.name,
        "id": jf.stem,
        "title": title,
        "events": events,
        "human_prompts": human_prompts,
        "human_chars": human_chars,
        "assistant_msgs": assistant_msgs,
        "tool_calls": tool_calls,
        "first_prompt": first_prompt,
        "ts_start": ts_start,
        "ts_end": ts_end,
    }


def _short_input(inp: dict) -> str:
    """One-line summary of a tool_use input (no giant file bodies)."""
    if not isinstance(inp, dict):
        return ""
    for k in ("command", "file_path", "pattern", "path", "query", "description", "prompt"):
        if k in inp and isinstance(inp[k], str):
            v = inp[k].replace("\n", " ")
            return f"{k}={v[:120]}"
    return json.dumps(inp)[:120]


def render_markdown(s: dict) -> str:
    L: list[str] = []
    title = s["title"] or "(untitled session)"
    L.append(f"# {title}\n")
    L.append(f"- session id: `{s['id']}`")
    L.append(f"- started: {s['ts_start']}  ·  ended: {s['ts_end']}")
    L.append(f"- human prompts: **{s['human_prompts']}**  ·  "
             f"human chars: **{s['human_chars']}**  ·  "
             f"assistant replies: {s['assistant_msgs']}  ·  "
             f"tool calls: {s['tool_calls']}")
    L.append("\n---\n")

    for ev in s["events"]:
        r = ev["role"]
        if r == "user":
            L.append("### 🧑 User\n")
            L.append(ev["text"] + "\n")
        elif r == "assistant":
            for b in ev["blocks"]:
                if b["kind"] == "text":
                    L.append("### 🤖 Claude\n")
                    L.append(b["text"] + "\n")
                elif b["kind"] == "thinking":
                    L.append("<details><summary>💭 thinking</summary>\n")
                    L.append("\n" + b["text"] + "\n\n</details>\n")
                elif b["kind"] == "tool_use":
                    L.append(f"> 🔧 **{b['name']}** — {_short_input(b['input'])}\n")
        elif r == "tool_result":
            block = ev["block"]
            content = block.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            content = str(content)
            clipped = content[:TOOL_RESULT_CLIP]
            extra = len(content) - len(clipped)
            tail = f"\n…[+{extra} chars]" if extra > 0 else ""
            L.append(f"<details><summary>↩️ tool result</summary>\n\n```\n"
                     f"{clipped}{tail}\n```\n\n</details>\n")
    return scrub("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="print the human-input ranking and exit (no files written)")
    args = ap.parse_args()

    sdir = find_session_dir()
    sessions = [parse_session(jf) for jf in sorted(sdir.glob("*.jsonl"))]
    sessions = [s for s in sessions if s["events"]]  # drop empty/aborted
    # rank by human steering: chars first, prompts as tiebreak
    sessions.sort(key=lambda s: (s["human_chars"], s["human_prompts"]), reverse=True)

    if args.list:
        for i, s in enumerate(sessions, 1):
            print(f"{i:2}. {s['human_chars']:6}c {s['human_prompts']:3}p  "
                  f"{(s['title'] or s['id'])[:60]}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "important").mkdir(exist_ok=True)

    index = ["# Session log index",
             "",
             f"Exported {datetime.now():%Y-%m-%d %H:%M} from `{sdir}`.",
             "",
             "Sessions ranked by **human input** (typed prompt characters) — the "
             "higher the rank, the more this conversation was human-steered rather "
             "than autonomous grind. Move the decision-driving ones into "
             "`_sessions/important/` by hand.",
             "",
             "| # | human chars | prompts | tool calls | date | title | file |",
             "|--:|--:|--:|--:|--|--|--|"]

    for i, s in enumerate(sessions, 1):
        date = (s["ts_start"] or "")[:10] or "unknown"
        fname = f"{date}_{s['id'][:8]}.md"
        (OUT_DIR / fname).write_text(render_markdown(s), encoding="utf-8")
        title = (s["title"] or "(untitled)").replace("|", "/")
        index.append(f"| {i} | {s['human_chars']} | {s['human_prompts']} | "
                     f"{s['tool_calls']} | {date} | {title} | [{fname}]({fname}) |")

    (OUT_DIR / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"Wrote {len(sessions)} transcripts + INDEX.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
