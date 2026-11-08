"""Set the prefilter's thresholds on Gemini's own answers (owner 08.11).

    python scripts/calibrate_prefilter.py <vision_log dir> [...] [--write]

Each directory is a build's "vision-log-<slug>" artifact (pipeline/vision.py): log.jsonl, one
row per frame Gemini answered (its line, the era, ok or not), and a small copy of the frame.
SigLIP scores every pair (pipeline/prefilter.py), and the thresholds are the ones where

    accept: of the frames scoring at or above it, at least ACCEPT_PRECISION were ok for Gemini
            (the CPU takes a frame Gemini would have turned down at most 1 time in 20);
    reject: of the frames scoring at or below it, at most REJECT_OK_RATE were ok for Gemini,
            and at most REJECT_LOST of all the ok frames are lost there.

--write keeps them in config/channel.json "prefilter" with what they were set on.
"""
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import prefilter  # noqa: E402

ACCEPT_PRECISION = 0.95
REJECT_OK_RATE = 0.10
REJECT_LOST = 0.10
MIN_SIDE = 15                    # a threshold needs at least this many frames on its side


def rows_of(dirs):
    out = []
    for d in map(Path, dirs):
        for line in (d / "log.jsonl").read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            img = d / f"{r['key']}.jpg"
            if r["line"].startswith("A thumbnail that says") or not img.exists():
                continue
            r["img"] = str(img)
            out.append(r)
    # the same frame under the same line answered twice: once
    return list({r["key"]: r for r in out}.values())


def choose(sc, ok):
    pairs = sorted(zip(sc, ok), key=lambda x: -x[0])
    n_ok = sum(ok)
    # accept: the lowest score whose frames at or above it (the ones the CPU would take) keep the
    # precision; the frames between it and the top are exactly that set
    accept, good = None, 0
    for k, (s, o) in enumerate(pairs, 1):
        good += o
        if k >= MIN_SIDE and good / k >= ACCEPT_PRECISION:
            accept = s
    asc = pairs[::-1]
    reject, lost, k = None, 0, 0
    for s, o in asc:
        k += 1
        lost += o
        if k >= MIN_SIDE and lost / k <= REJECT_OK_RATE and lost <= REJECT_LOST * max(1, n_ok):
            reject = s
    return accept, reject


def report(sc, ok, accept, reject):
    n = len(sc)
    a = [o for s, o in zip(sc, ok) if accept is not None and s >= accept]
    r = [o for s, o in zip(sc, ok) if reject is not None and s <= reject]
    mid = n - len(a) - len(r)
    return {"frames": n, "gemini_ok": sum(ok), "accept": accept, "reject": reject,
            "taken_on_cpu": len(a), "their_gemini_ok_share": round(sum(a) / len(a), 3) if a else None,
            "dropped_on_cpu": len(r), "good_frames_lost": sum(r),
            "still_for_gemini": mid, "share_decided_on_cpu": round((len(a) + len(r)) / n, 3) if n else 0}


def main(args):
    write = "--write" in args
    dirs = [a for a in args if not a.startswith("--")]
    rows = rows_of(dirs)
    sc = prefilter.scores([(r["line"], r.get("era_hint") or "", r["img"]) for r in rows])
    ok = [bool(r["ok"]) for r in rows]
    accept, reject = choose(sc, ok)
    if accept is not None and reject is not None and reject >= accept:
        reject = None
    rep = report(sc, ok, accept, reject)
    # by Gemini's reason: what the CPU cannot see (clean, era, place) among the frames it would take
    why = {}
    for r, s in zip(rows, sc):
        if accept is not None and s >= accept and not r["ok"]:
            for f in ("match", "era", "place_us", "clean"):
                if r.get(f) is False:
                    why[f] = why.get(f, 0) + 1
    rep["taken_on_cpu_but_gemini_said_no_because"] = why
    print(json.dumps(rep, indent=1))
    if write:
        p = ROOT / "config" / "channel.json"
        text = p.read_text(encoding="utf-8")
        cfg = json.loads(text)
        cfg["prefilter"] = {**(cfg.get("prefilter") or {}), "arch": prefilter.ARCH, "pretrained": prefilter.PRETRAINED,
                            "accept": round(accept, 4) if accept is not None else None,
                            "reject": round(reject, 4) if reject is not None else None,
                            "_set_on": f"{rep['frames']} Gemini answers ({', '.join(Path(d).name for d in dirs)}), "
                                       f"{datetime.now(timezone.utc):%Y-%m-%d}: {rep['taken_on_cpu']} taken at "
                                       f"{rep['their_gemini_ok_share']} Gemini agreement, {rep['dropped_on_cpu']} "
                                       f"dropped losing {rep['good_frames_lost']} good"}
        p.write_text(json.dumps(cfg, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("written to", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
