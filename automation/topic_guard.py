"""Which topic may be rendered (owner 17.11, after one topic was built ten times and went up twice).

    python automation/topic_guard.py <slug>      # exit 0 and the reason when the topic must NOT render
    python automation/topic_guard.py --next      # the next topic that may render (what next_topic.py prints)

A topic is refused when
  - the state holds a video for it (youtube_id) or it is built / scheduled / published / awaiting:
    the render, the hours and Gemini's day would be spent for nothing;
  - the queue marks it built, merged or skipped (a hand-given slug passes this one with force=1);
  - it covers the same ground as a topic that already has a video: two or more Wikipedia articles
    in common, or most of the name/related words (Levittown twice, 20.11: "Suburban Dreams" and
    "The $7,990 Dream").
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HAS_VIDEO = {"built", "scheduled", "published", "uploaded", "awaiting_approval", "awaiting_turn"}
QUEUE_DONE = {"built", "uploaded", "published", "merged", "skipped"}
STOP = {"a", "an", "the", "in", "of", "and", "to", "for", "on", "at", "by", "with", "was", "were", "is", "color",
        "colour", "america", "american", "americans", "us", "life", "living", "1950s", "1960s", "1940s", "1950",
        "year", "years", "history", "united", "states", "post", "world", "war", "ii", "era", "day", "how", "what"}


def _load(p, default):
    f = ROOT / p
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else default


def words(*texts):
    return {w for t in texts for w in re.findall(r"[a-z0-9$]+", str(t).lower()) if w not in STOP and len(w) > 2}


def _covered(q, rows):
    """The topics that already have a video: their name, articles and words (merged ones included)."""
    by_slug = {t["slug"]: t for t in q["topics"]}
    have = {r["slug"] for r in rows if r.get("youtube_id") or r.get("state") in HAS_VIDEO}
    have |= {t["slug"] for t in q["topics"] if t.get("status") in ("built", "uploaded", "published")}
    out = []
    for slug in have:
        t = by_slug.get(slug)
        title = next((r.get("title") for r in rows if r["slug"] == slug and r.get("title")), "")
        if t:
            out.append((slug, set(t.get("wikipedia") or []), words(t["name"], *(t.get("related") or []), title)))
            for m in t.get("merged_from") or []:
                mt = by_slug.get(m)
                if mt:
                    out.append((m, set(mt.get("wikipedia") or []), words(mt["name"], *(mt.get("related") or []))))
        else:
            out.append((slug, set(), words(title)))
    return out


def similar(topic, q, rows):
    """(slug, why) of a covered topic this one repeats, or None."""
    mine_w = set(topic.get("wikipedia") or [])
    mine = words(topic["name"], *(topic.get("related") or []))
    for slug, wiki, ws in _covered(q, rows):
        if slug == topic["slug"]:
            continue
        shared = mine_w & wiki
        if len(shared) >= 2:
            return slug, f"{len(shared)} articles in common: {sorted(shared)[:3]}"
        if mine and ws and len(mine & ws) / len(mine | ws) >= 0.5:
            return slug, f"the same words: {sorted(mine & ws)[:5]}"
    return None


def blocked(slug, force=False):
    """The reason this topic must not render, or ''."""
    q = _load("topics/queue.json", {"topics": []})
    rows = _load("state/episodes.json", [])
    row = next((r for r in rows if r.get("slug") == slug), {})
    if row.get("youtube_id"):
        return f"{slug} is already on YouTube ({row['youtube_id']})"
    if row.get("state") in HAS_VIDEO:
        return f"{slug} is {row['state']}: a build exists, it is not rendered again"
    t = next((t for t in q["topics"] if t["slug"] == slug), None)
    if t is None:
        return f"{slug} is not in topics/queue.json"
    if not force and t.get("status") in QUEUE_DONE:
        return f"{slug} is marked {t['status']} in the queue"
    if not force:
        s = similar(t, q, rows)
        if s:
            return f"{slug} repeats {s[0]} ({s[1]})"
    return ""


def next_topic():
    """The first queued topic that may render."""
    q = _load("topics/queue.json", {"topics": []})
    rows = _load("state/episodes.json", [])
    done = {r["slug"] for r in rows}
    from automation import material
    for t in q["topics"]:
        if t["slug"] in done or t.get("status", "queued") != "queued" or not t.get("wikipedia"):
            continue
        why = blocked(t["slug"])
        if why:
            print(f"skipped {t['slug']}: {why}", file=sys.stderr)
            continue
        if material.usable(t, q):
            return t["slug"]
    return ""


if __name__ == "__main__":
    if sys.argv[1] == "--next":
        print(next_topic())
        sys.exit(0)
    why = blocked(sys.argv[1], force=os.environ.get("FORCE") == "1")
    if why:
        print(why)
        sys.exit(0)
    sys.exit(1)
