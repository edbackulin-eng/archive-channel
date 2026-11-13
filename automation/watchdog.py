"""The watchdog: keeps the channel running with nobody watching (watchdog.yml).

Each pass:
  1. an episode run that failed or was cut off has its failed jobs rerun (the
     artifacts of the jobs that passed are kept), up to three attempts; after
     that the owner gets one email with the reason;
  2. while fewer than two episodes are built or scheduled ahead and no episode
     run is going, the next episode is started;
  3. once a YouTube token exists, buffered episodes are uploaded (upload.yml);
  4. state/heartbeat.json is committed once a day, so the repository never goes
     60 days without a commit and GitHub never turns its schedules off.

Owner 12.11 - no pause that waits for the owner:
  - a run that failed all its attempts: the topic is built again by a new run (today's code); a
    topic that fails twice is skipped (topics/queue.json "skipped") and the next one is built,
    one letter;
  - an episode whose QA found a real fault (only faults block: pipeline/qa.py) is built again once,
    then skipped the same way;
  - only when the same step fails for two different topics is the code taken to be broken: builds
    pause (automation/pause.py) until the code changes, one letter.
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automation import pause, quota_hold  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/archive-channel")
STATE = ROOT / "state" / "episodes.json"
MEM = ROOT / "state" / "watchdog.json"
BEAT = ROOT / "state" / "heartbeat.json"
AHEAD = 2
MAX_ATTEMPTS = 3
TOPIC_FAILS = 2           # owner 12.11: failed builds of one topic before the next topic is built
QA_REBUILDS = 1           # builds again after a QA fault before the topic is skipped
QUEUE = ROOT / "topics" / "queue.json"


def run_slug(run_id):
    """The topic a run built: the SLUG its jobs show in their log."""
    import re
    r = subprocess.run(["gh", "run", "view", str(run_id), "-R", REPO, "--log"], capture_output=True, text=True)
    m = re.search(r"\bSLUG: ([a-z0-9-]+)", r.stdout or "")
    return m.group(1) if m else None


def failed_step(run_id):
    try:
        jobs = gh("api", f"repos/{REPO}/actions/runs/{run_id}/jobs")["jobs"]
    except Exception:                                                # noqa: BLE001
        return "?"
    for j in jobs:
        if j.get("conclusion") == "failure":
            st = next((s["name"] for s in j.get("steps", []) if s.get("conclusion") == "failure"), "?")
            return f"{j['name']} / {st}"
    return "?"


def notify(subject, body):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "notify.py"), subject, body])


def skip_topic(slug, why):
    """The topic is left; the next queued one is built (next_topic.py takes only "queued")."""
    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    for t in q["topics"]:
        if t["slug"] == slug:
            t.update({"status": "skipped", "skipped_why": why, "skipped_at": now().strftime("%Y-%m-%dT%H:%M:%SZ")})
    QUEUE.write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    rows = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    for r in rows:
        if r.get("slug") == slug and r.get("state") in ("qa-failed", "rebuilding"):
            r["state"] = "skipped"
    STATE.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    notify(f"{slug} skipped - the next topic is being built", why)


def failed_build(last, mem, fp, actions):
    """Owner 12.11: a run that failed all its attempts - the topic again, the next topic, or (the same
    step failing for two topics) a pause until the code changes. True when a run was started."""
    slug = run_slug(last["id"]) or (last.get("display_title") or "").removeprefix("episode").strip() or "?"
    sig = failed_step(last["id"])
    fails = mem.setdefault("topic_fails", [])
    fails.append({"slug": slug, "sig": sig, "run": last["id"]})
    mem["topic_fails"] = fails[-100:]
    others = sorted({f["slug"] for f in fails if f["sig"] == sig and f["slug"] not in (slug, "?")})
    url = f"https://github.com/{REPO}/actions/runs/{last['id']}"
    if others:
        pause.set_(f"the code is broken: '{sig}' failed for {slug} and {', '.join(others)}", slug, fp=fp)
        notify(f"The code is broken: '{sig}' fails for two topics; builds pause until the code changes",
               f"{url}\nTopics: {slug}, {', '.join(others)}. Any change to the code lifts the pause.")
        actions.append(f"builds paused: '{sig}' failed for {slug} and {others}")
        return False
    n = sum(1 for f in fails if f["slug"] == slug)
    if slug != "?" and n >= TOPIC_FAILS:
        skip_topic(slug, f"its build failed {n} times ('{sig}'), last {url}")
        actions.append(f"{slug} skipped after {n} failed builds; the next topic follows")
        return False
    if slug != "?":
        gh("workflow", "run", "episode.yml", "-R", REPO, "-f", f"slug={slug}", parse=False)
        actions.append(f"{slug}: its run failed all attempts ('{sig}'); built again by a new run")
        return True
    return False


def qa_rebuild(rows, busy, actions):
    """Owner 12.11: an episode whose QA found a real fault is built again once, then skipped."""
    if busy:
        return False
    for r in rows:
        if r.get("state") != "qa-failed":
            continue
        n = int(r.get("qa_rebuilds", 0))
        if n >= QA_REBUILDS:
            skip_topic(r["slug"], f"QA found a real fault in {n + 1} builds: {r.get('qa_failed')}")
            actions.append(f"{r['slug']} skipped after {n + 1} QA faults; the next topic follows")
            return False
        r.update({"state": "rebuilding", "qa_rebuilds": n + 1})
        STATE.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
        gh("workflow", "run", "episode.yml", "-R", REPO, "-f", f"slug={r['slug']}", parse=False)
        actions.append(f"{r['slug']}: QA fault {r.get('qa_failed')}; built again ({n + 1}/{QA_REBUILDS})")
        return True
    return False


def now():
    return datetime.now(timezone.utc)


def gh(*args, parse=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return json.loads(r.stdout) if parse and r.stdout.strip() else r.stdout


def quota_stop(run_id):
    """True when the run stopped because Gemini's day was spent (pipeline/vision.py QuotaSpent)."""
    r = subprocess.run(["gh", "run", "view", str(run_id), "-R", REPO, "--log-failed"], capture_output=True, text=True)
    return "VISION_QUOTA_SPENT" in (r.stdout or "")


def after_reset(at, now_):
    """The free tier's day turns at midnight Pacific (08:00 UTC); 5 minutes' margin."""
    reset = at.replace(hour=8, minute=5, second=0, microsecond=0)
    if reset <= at:
        reset += timedelta(days=1)
    return now_ >= reset


def runs(wf):
    return gh("api", f"repos/{REPO}/actions/workflows/{wf}/runs?per_page=10")["workflow_runs"]


def main():
    mem = json.loads(MEM.read_text(encoding="utf-8")) if MEM.exists() else {"emailed": []}
    actions = []
    ep = runs("episode.yml")
    busy = any(r["status"] != "completed" for r in ep)
    last = next((r for r in ep if r["status"] == "completed"), None)
    fp = pause.fingerprint()
    if pause.lift_if_changed(fp):
        actions.append("the code or the plan changed: builds resume")
    held = pause.active(fp)
    hold = quota_hold.load()
    recent = bool(last and last["conclusion"] in ("failure", "cancelled", "timed_out")
                  and now() - datetime.fromisoformat(last["updated_at"].replace("Z", "+00:00")) <= timedelta(minutes=12))
    if recent:
        busy = True                      # 08.11: a run that failed minutes ago is looked at first, not passed by
    if hold and quota_hold.active(now()):
        # the Gemini day is spent: nothing reruns or starts until it turns (automation/quota_hold.py)
        actions.append(f"Gemini's day is spent until {hold['until']}; {', '.join(hold['slugs'])} wait")
        busy = True
    elif hold and not busy:
        # the day turned: the topics that stopped start again, the first now, the rest after it
        slug = next(iter(hold.get("slugs") or []), "")
        if slug:
            gh("workflow", "run", "episode.yml", "-R", REPO, "-f", f"slug={slug}", parse=False)
            actions.append(f"Gemini's day turned: episode {slug} started again on today's code")
        rest = [s for s in hold.get("slugs") or [] if s != slug]
        if rest:
            hold["slugs"] = rest
            quota_hold.HOLD.write_text(json.dumps(hold, indent=1) + "\n", encoding="utf-8")
        else:
            quota_hold.clear()
        busy = True
    elif not busy and last and last["conclusion"] in ("failure", "cancelled", "timed_out"):
        ended = datetime.fromisoformat(last["updated_at"].replace("Z", "+00:00"))
        if not held and last["conclusion"] == "failure" and quota_stop(last["id"]):
            # Gemini's day was spent: not a fault - rerun after it turns, and no pause
            if not after_reset(ended, now()):
                actions.append(f"episode run {last['id']} stopped on Gemini's spent day; rerun after 08:05 UTC")
                busy = True
            else:
                # 08.11: a new run of the same topic, not a rerun - a rerun keeps the old commit's
                # code, a new run has today's and the vision answers kept so far (actions/cache)
                slug = (last.get("display_title") or "").removeprefix("episode").strip()
                n = mem.setdefault("quota_runs", {}).get(slug, 0)
                on_yt = any(x.get("slug") == slug and x.get("youtube_id")
                            for x in (json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []))
                if on_yt:
                    actions.append(f"{slug} is already on YouTube; its stopped run is not started again")
                elif slug and n < 6:
                    gh("workflow", "run", "episode.yml", "-R", REPO, "-f", f"slug={slug}", parse=False)
                    mem["quota_runs"][slug] = n + 1
                    actions.append(f"episode {slug} started again after Gemini's day turned (run {last['id']})")
                elif last.get("run_attempt", 1) < MAX_ATTEMPTS + 3:
                    gh("run", "rerun", str(last["id"]), "-R", REPO, "--failed", parse=False)
                    actions.append(f"reran episode run {last['id']} after Gemini's day turned")
                busy = True
        elif held:
            pass                                     # a paused build is not rerun either
        elif "QA gate" in failed_step(last["id"]):
            # 13.11: the run ended on the QA verdict - a rerun keeps the old commit's code and gives
            # the same verdict; qa_rebuild below builds the topic again by a new run of today's code
            pass
        elif last.get("run_attempt", 1) < MAX_ATTEMPTS:
            try:
                gh("run", "rerun", str(last["id"]), "-R", REPO, "--failed", parse=False)
            except RuntimeError:
                gh("run", "rerun", str(last["id"]), "-R", REPO, parse=False)
            actions.append(f"reran episode run {last['id']}")
            busy = True
        elif f"run-{last['id']}" not in mem["emailed"]:
            mem["emailed"].append(f"run-{last['id']}")
            busy = failed_build(last, mem, fp, actions) or busy
            held = pause.active(fp)
    rows = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    ahead = [r for r in rows if r.get("state") in ("built", "awaiting_approval", "awaiting_turn")
             or (r.get("state") == "scheduled" and r.get("publish_at", "") > now().strftime("%Y-%m-%dT%H:%M:%SZ"))]
    keys = os.environ.get("KEYS_READY") == "true"
    if not held and keys:
        busy = qa_rebuild(rows, busy, actions) or busy
    if held:
        actions.append(f"builds paused: {held['why']} (since {held['since']}); "
                       "they resume when the code or the plan changes")
    elif len(ahead) < AHEAD and not busy:
        if not keys:
            # without the text and vision keys every build would fail QA and use up a topic
            actions.append("waiting for GROQ_API_KEY_AR and GEMINI_API_KEY_AR; no build started")
        else:
            gh("workflow", "run", "episode.yml", "-R", REPO, parse=False)
            actions.append(f"only {len(ahead)} ahead; next episode started")
    if os.environ.get("YT_TOKEN_JSON") and any(r.get("state") == "built" and not r.get("youtube_id") for r in rows):
        gh("workflow", "run", "upload.yml", "-R", REPO, parse=False)
        actions.append("buffered episodes sent to upload")
    old = json.loads(BEAT.read_text(encoding="utf-8")) if BEAT.exists() else {}
    beat = {"date": f"{now():%Y-%m-%d}", "checked_at": f"{now():%Y-%m-%dT%H:%M:%SZ}",
            "ahead": [[r["slug"], r.get("state"), r.get("publish_at")] for r in ahead], "actions": actions}
    BEAT.parent.mkdir(exist_ok=True)
    BEAT.write_text(json.dumps(beat, indent=1), encoding="utf-8")
    MEM.write_text(json.dumps(mem, indent=1), encoding="utf-8")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as f:
            f.write(f"commit={'true' if actions or old.get('date') != beat['date'] else 'false'}\n")
    print(json.dumps(beat, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
