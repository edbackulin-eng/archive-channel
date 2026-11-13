"""Builds paused after a QA failure or a build that failed three times, until the code or the plan changes.

    python automation/pause.py active          # exit 0 and the reason while paused
    python automation/pause.py from-state SLUG # after a build: pause if SLUG failed QA
    python automation/pause.py set "why"       # pause now

Owner 07.11: after 1950s-suburb failed QA (06.11) the watchdog and the monitor started
episode.yml again every 45 minutes all night, and each run failed the same way in its
script stage. A build that cannot pass is not retried: the pause holds the fingerprint of
what decides a build - the code (pipeline/, automation/, config/, the episode workflow) and
the plan (topics/) - and is lifted only when that fingerprint changes. The pick job of
episode.yml honours it whoever starts the run; a slug given by hand still builds.
"""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAUSE = ROOT / "state" / "build_pause.json"
STATE = ROOT / "state" / "episodes.json"
DECIDES = ("pipeline", "automation", "config", "topics", ".github/workflows/episode.yml")


def fingerprint():
    """The git objects of what decides a build (HEAD's, so a checkout of any depth works)."""
    parts = []
    for p in DECIDES:
        r = subprocess.run(["git", "rev-parse", f"HEAD:{p}"], cwd=ROOT, capture_output=True, text=True)
        parts.append(f"{p}={r.stdout.strip()}")
    return hashlib.sha1("\n".join(parts).encode()).hexdigest()[:16]


def load():
    return json.loads(PAUSE.read_text(encoding="utf-8")) if PAUSE.exists() else None


def active(fp=None):
    """The pause while the code and the plan are what they were when it was set; None after."""
    p = load()
    if p and p.get("fingerprint") == (fp or fingerprint()):
        return p
    return None


def lift_if_changed(fp=None):
    """Remove a pause whose code or plan has changed since; True if one was removed."""
    p = load()
    if p and p.get("fingerprint") != (fp or fingerprint()):
        PAUSE.unlink()
        return True
    return False


def set_(why, slug=None, fp=None):
    PAUSE.parent.mkdir(parents=True, exist_ok=True)
    row = {"why": why, "slug": slug, "fingerprint": fp or fingerprint(),
           "since": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "lifted_by": "a change to " + ", ".join(DECIDES)}
    PAUSE.write_text(json.dumps(row, indent=1) + "\n", encoding="utf-8")
    return row


def from_state(slug):
    """Owner 12.11: a QA failure no longer pauses the channel - the watchdog builds the topic again
    once and then goes on with the next topic (automation/watchdog.py qa_rebuild)."""
    return None


def main(argv):
    if argv[:1] == ["active"]:
        p = active()
        if p:
            print(p["why"])
            return 0
        return 1
    if argv[:1] == ["from-state"] and len(argv) == 2:
        p = from_state(argv[1])
        print(f"paused: {p['why']}" if p else "no pause")
        return 0
    if argv[:1] == ["set"] and len(argv) >= 2:
        print(set_(" ".join(argv[1:]))["why"])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
