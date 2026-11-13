"""Owner 12.11: a topic is written only once there is film for it.

    python automation/material.py <slug>      # checks and records topics/queue.json "material"

Counts the films the archive search finds for the topic's query and years - each with a verified
free licence (pipeline/archive.search) - before any script is written. A topic under MIN_FILMS
(or under MIN_COLOUR in colour) is skipped ("skipped-material") and the next one is taken; the
cut later still defers a topic with under 15 minutes of unique film.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
QUEUE = ROOT / "topics" / "queue.json"
MIN_FILMS, MIN_COLOUR = 30, 10        # the topics built so far find 90+ films, 30+ in colour


def check(t):
    from pipeline import archive
    try:
        films = archive.search(t["query"], *(t.get("years") or [None, None]), rows=100)
    except Exception as e:                                           # noqa: BLE001
        return {"ok": None, "error": str(e)[:200]}                   # unknown: the topic is not skipped for it
    colour = sum(1 for f in films if f.get("colour"))
    return {"films": len(films), "colour": colour, "ok": len(films) >= MIN_FILMS and colour >= MIN_COLOUR}


def usable(t, q=None):
    """True when the topic may be written; records the check in the queue (once)."""
    if "material" not in t:
        t["material"] = check(t)
        if t["material"].get("ok") is False:
            t["status"] = "skipped-material"
        if q is not None:
            QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return t.get("material", {}).get("ok") is not False


if __name__ == "__main__":
    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    t = next(x for x in q["topics"] if x["slug"] == sys.argv[1])
    t.pop("material", None)
    print(usable(t, q), t.get("material"))
