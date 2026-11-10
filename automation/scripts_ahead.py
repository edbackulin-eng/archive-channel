"""Scripts written ahead (owner 10.11): whenever the text quota allows, the next topics get their
script written and fact-checked, so a build never waits on a text provider on its own day.

    python automation/scripts_ahead.py [--n 3] [--skip <slug>]

The next `n` queued topics (in queue order, with their Wikipedia articles, not built yet) that have
no episodes/<slug>/script.txt are written by automation/write_episode.py, one after the other; the
topic being built now (--skip) is left to its own build. Nothing is written while the Gemini day is
held (automation/quota_hold.py), and a provider out of its day stops the run (the hold is set).
scripts-ahead.yml commits what was written.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import quota_hold  # noqa: E402
from core.logger import get_logger  # noqa: E402

log = get_logger("scripts-ahead")


def upcoming(n, skip=()):
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    st = ROOT / "state" / "episodes.json"
    done = {r["slug"] for r in json.loads(st.read_text(encoding="utf-8"))} if st.exists() else set()
    out = []
    for t in q["topics"]:
        if t.get("status", "queued") != "queued" or not t.get("wikipedia") or t["slug"] in done \
                or t["slug"] in skip:
            continue
        out.append(t["slug"])
        if len(out) >= n:
            break
    return out


def main(args):
    n = int(args[args.index("--n") + 1]) if "--n" in args else 3
    skip = {args[i + 1] for i, a in enumerate(args) if a == "--skip" and i + 1 < len(args) and args[i + 1]}
    if quota_hold.active():
        log.info("the Gemini day is held; nothing written now")
        return 0
    written = []
    for slug in upcoming(n, skip):
        if (ROOT / "episodes" / slug / "script.txt").exists():
            continue
        log.info("writing ahead: %s", slug)
        r = subprocess.run([sys.executable, str(ROOT / "automation" / "write_episode.py"), slug])
        if r.returncode != 0:
            log.warning("%s: not written now (exit %d)", slug, r.returncode)
            if quota_hold.active():
                break                                    # out of providers for the day: the rest later
            continue
        written.append(slug)
    log.info("written ahead: %s", ", ".join(written) or "nothing (all have their script)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
