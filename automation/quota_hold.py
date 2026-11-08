"""No build starts while the channel's Gemini day is spent (08.11).

    python automation/quota_hold.py active        # exit 0 and the reason while the day is spent
    python automation/quota_hold.py waiting       # exit 0 while topics stopped on it wait their turn
    python automation/quota_hold.py set SLUG      # after a stage stopped on the spent day

The key GEMINI_API_KEY_AR serves the vision check and the text fallback, on one free day
(gemini-flash-lite-latest; it turns at midnight Pacific, 08:00 UTC). On 07.11 1950s-suburb
spent it at 23:14 UTC in its shot plan, and within six minutes the watchdog and the monitor
had started 1950s-groceries twice: each run failed in its script stage (Groq 413 on the long
prompt, Gemini 429) and used what little was left. Now the stage that meets the spent day
writes state/quota_hold.json: until 08:05 UTC the pick job of episode.yml starts nothing
(a slug given by hand still builds), the watchdog neither reruns nor starts, and after it
the watchdog starts again the topics that stopped, in their order, on the newest code.
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOLD = ROOT / "state" / "quota_hold.json"
MARKS = ("VISION_QUOTA_SPENT", "GEMINI_DAY_SPENT")


def now():
    return datetime.now(timezone.utc)


def reset_after(at):
    """The free tier's day turns at 08:00 UTC; 5 minutes' margin."""
    r = at.replace(hour=8, minute=5, second=0, microsecond=0)
    return r if r > at else r + timedelta(days=1)


def load():
    try:
        return json.loads(HOLD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def until(h):
    return datetime.fromisoformat(h["until"].replace("Z", "+00:00"))


def active(at=None):
    h = load()
    return h if h and (at or now()) < until(h) else None


def set_(slug, at=None):
    at = at or now()
    h = load() or {}
    if not h or until(h) <= at:
        h = {"until": reset_after(at).strftime("%Y-%m-%dT%H:%M:%SZ"), "slugs": []}
    if slug and slug not in h["slugs"]:
        h["slugs"].append(slug)
    h["since"] = h.get("since") or at.strftime("%Y-%m-%dT%H:%M:%SZ")
    HOLD.parent.mkdir(exist_ok=True)
    HOLD.write_text(json.dumps(h, indent=1) + "\n", encoding="utf-8")
    return h


def clear():
    if HOLD.exists():
        HOLD.unlink()


def spent(text):
    """True when an error says the Gemini day is spent (and no other provider answered)."""
    t = str(text)
    return any(m in t for m in MARKS) or ("no provider answered" in t and "gemini: HTTP 429" in t)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "active":
        h = active()
        if h:
            print(f"Gemini's day is spent until {h['until']} (stopped: {', '.join(h['slugs'])})")
            sys.exit(0)
        sys.exit(1)
    if cmd == "waiting":
        # 08.11: the day turned at 08:05 and the monitor started 1950s-groceries (no slug) before the
        # watchdog could start 1950s-suburb: while topics wait their turn, a run with no slug starts none
        h = load()
        if h:
            state = "spent until" if active() else "turned at"
            print(f"Gemini's day {state} {h['until']}; waiting their turn: {', '.join(h.get('slugs') or [])}")
            sys.exit(0)
        sys.exit(1)
    if cmd == "set":
        print(json.dumps(set_(sys.argv[2] if len(sys.argv) > 2 else "")))
