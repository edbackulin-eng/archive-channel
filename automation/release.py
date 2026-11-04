"""After a build: the episode goes to YouTube, or into the buffer.

    python automation/release.py record <slug>     # after QA: buffer it; upload if a token exists
    python automation/release.py upload            # every buffered episode, once a token exists
    python automation/release.py approve <slug>    # approve.yml: the nearest free slot for an approved one

state/episodes.json holds every episode:
  built              rendered, QA passed, in the buffer (files in the build run's artifact)
  awaiting_approval  uploaded private without a date: the first two episodes wait for
                     the owner (config qa.approve_first_episodes) - approve.yml schedules them
  awaiting_turn      uploaded private without a date while the first two still wait, so
                     it never goes out before them; the second approval gives each its
                     slot, in upload order
  scheduled          uploaded private with a publishAt
  qa-failed          the QA gate refused it; nothing was uploaded

Nothing is uploaded without a passing QA report (episodes/<slug>/out/qa.json).
One episode, one video: the channel is asked first (youtube.find_by_title) and the
id is written to the state the moment YouTube returns it.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import youtube  # noqa: E402

log = get_logger("release")
STATE = ROOT / "state" / "episodes.json"
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))


def rows():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def save(r):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


def row_of(r, slug):
    row = next((x for x in r if x["slug"] == slug), None)
    if row is None:
        row = {"slug": slug}
        r.append(row)
    return row


def _approved_count(r):
    return sum(1 for x in r if x.get("approved"))


def _first(r):
    """The slugs of the first episodes uploaded: the ones the owner approves."""
    n = CFG["qa"]["approve_first_episodes"]
    return [x["slug"] for x in r if x.get("youtube_id") and x.get("state") != "qa-failed"][:n]


def _upload(slug, r):
    out = ROOT / "episodes" / slug / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    row = row_of(r, slug)
    if row.get("youtube_id"):
        return row
    n = CFG["qa"]["approve_first_episodes"]
    if len(_first(r)) < n:
        state, when = "awaiting_approval", None          # one of the first: the owner approves it
    elif _approved_count(r) < n:
        state, when = "awaiting_turn", None              # the first still wait; it does not jump them
    else:
        state, when = "scheduled", youtube.next_slot([x["publish_at"] for x in r if x.get("publish_at")])
    vid = youtube.upload(out / "video.mp4", meta["title"], meta["description"], meta["tags"], when)
    row.update({"youtube_id": vid, "state": state,
                "title": meta["title"], **({"publish_at": when.strftime("%Y-%m-%dT%H:%M:%SZ")} if when else {})})
    save(r)                                   # the id is kept before anything else can fail
    try:
        youtube.set_thumbnail(vid, out / "thumbnail.jpg")
    except Exception as e:                                           # noqa: BLE001
        row["thumbnail"] = f"failed: {str(e)[:120]}"
        save(r)
    return row


def record(slug):
    r = rows()
    qa = json.loads((ROOT / "episodes" / slug / "out" / "qa.json").read_text(encoding="utf-8"))
    row = row_of(r, slug)
    if not qa.get("passed"):
        row["state"] = "qa-failed"
        row["qa_failed"] = [k for k, v in qa.items() if isinstance(v, dict) and v.get("pass") is False]
        save(r)
        log.error("%s failed QA: %s", slug, row["qa_failed"])
        return 1
    row.setdefault("state", "built")
    save(r)
    if youtube.has_token():
        _upload(slug, r)
    else:
        log.info("%s built and buffered; no YouTube token yet", slug)
    return 0


def upload():
    r = rows()
    for row in [x for x in r if x.get("state") == "built" and not x.get("youtube_id")]:
        _upload(row["slug"], r)
    return 0


def approve(slug):
    r = rows()
    row = row_of(r, slug)
    if row.get("state") != "awaiting_approval":
        raise SystemExit(f"{slug} is {row.get('state')}, not awaiting approval")
    when = youtube.next_slot([x["publish_at"] for x in r if x.get("publish_at")])
    youtube.schedule(row["youtube_id"], when)
    row.update({"state": "scheduled", "approved": True, "publish_at": when.strftime("%Y-%m-%dT%H:%M:%SZ")})
    save(r)
    print(f"{slug}: {row['youtube_id']} public at {row['publish_at']}")
    if _approved_count(r) >= CFG["qa"]["approve_first_episodes"]:
        # the first are approved: the ones that waited their turn get slots after them, in order
        for w in [x for x in r if x.get("state") == "awaiting_turn"]:
            at = youtube.next_slot([x["publish_at"] for x in r if x.get("publish_at")])
            youtube.schedule(w["youtube_id"], at)
            w.update({"state": "scheduled", "publish_at": at.strftime("%Y-%m-%dT%H:%M:%SZ")})
            save(r)
            print(f"{w['slug']}: {w['youtube_id']} public at {w['publish_at']}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    sys.exit({"record": lambda: record(sys.argv[2]), "upload": upload,
              "approve": lambda: approve(sys.argv[2])}[cmd]())
