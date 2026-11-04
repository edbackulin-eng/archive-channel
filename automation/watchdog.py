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
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/archive-channel")
STATE = ROOT / "state" / "episodes.json"
MEM = ROOT / "state" / "watchdog.json"
BEAT = ROOT / "state" / "heartbeat.json"
AHEAD = 2
MAX_ATTEMPTS = 3


def now():
    return datetime.now(timezone.utc)


def gh(*args, parse=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    return json.loads(r.stdout) if parse and r.stdout.strip() else r.stdout


def runs(wf):
    return gh("api", f"repos/{REPO}/actions/workflows/{wf}/runs?per_page=10")["workflow_runs"]


def main():
    mem = json.loads(MEM.read_text(encoding="utf-8")) if MEM.exists() else {"emailed": []}
    actions = []
    ep = runs("episode.yml")
    busy = any(r["status"] != "completed" for r in ep)
    last = next((r for r in ep if r["status"] == "completed"), None)
    if not busy and last and last["conclusion"] in ("failure", "cancelled", "timed_out") \
            and now() - datetime.fromisoformat(last["updated_at"].replace("Z", "+00:00")) > timedelta(minutes=12):
        if last.get("run_attempt", 1) < MAX_ATTEMPTS:
            try:
                gh("run", "rerun", str(last["id"]), "-R", REPO, "--failed", parse=False)
            except RuntimeError:
                gh("run", "rerun", str(last["id"]), "-R", REPO, parse=False)
            actions.append(f"reran episode run {last['id']}")
            busy = True
        elif f"run-{last['id']}" not in mem["emailed"]:
            subprocess.run([sys.executable, str(ROOT / "scripts" / "notify.py"),
                            f"Episode build {last['id']} failed {last.get('run_attempt')} times",
                            f"https://github.com/{REPO}/actions/runs/{last['id']}"])
            mem["emailed"].append(f"run-{last['id']}")
    rows = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []
    ahead = [r for r in rows if r.get("state") in ("built", "awaiting_approval", "awaiting_turn")
             or (r.get("state") == "scheduled" and r.get("publish_at", "") > now().strftime("%Y-%m-%dT%H:%M:%SZ"))]
    keys = os.environ.get("KEYS_READY") == "true"
    failed_run = [r for r in rows[-2:] if r.get("state") == "qa-failed"]
    if len(ahead) < AHEAD and not busy:
        if not keys:
            # without the text and vision keys every build would fail QA and use up a topic
            actions.append("waiting for GROQ_API_KEY_AR and GEMINI_API_KEY_AR; no build started")
        elif len(failed_run) >= 2:
            if "two-qa-failed" not in mem["emailed"]:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "notify.py"),
                                "Two episodes in a row failed the QA gate; builds paused",
                                ", ".join(f"{r['slug']}: {r.get('qa_failed')}" for r in failed_run)])
                mem["emailed"].append("two-qa-failed")
            actions.append("two QA failures in a row; builds paused until a passing one")
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
