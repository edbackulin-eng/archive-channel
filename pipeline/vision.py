"""Does this frame belong under this line? Gemini vision, the channel's own key.

Key: GEMINI_API_KEY_AR only (config/channel.json "vision"). For every candidate
shot the keyframe goes to Gemini with the narration line it would sit under,
and four questions:
  match      does the picture show what the line is about (or a fitting,
             honest illustration of it)?
  era        does it look like the period the line is about (clothes, cars,
             rooms), give or take ten years?
  place      is it plausibly the United States?
  clean      no watermark, no channel or broadcaster logo, no burned-in caption
             or stock-agency mark, no modern overlay.
A shot is usable when all four hold. Frames are sent in batches of 12 lines and
paced; a frame that could not be checked is "unchecked", never "ok" - the QA
gate counts unchecked frames as unmatched.

When the key's day is spent the check does not go on unchecked: it raises QuotaSpent
(owner 07.11). 1950s-suburb (06.11) went on with 227 of 243 shots unchecked, rendered for
an hour and a half and failed QA on match_share (0.016 of 0.85) and on the thumbnail (no
checked shot to put under its words). Now the stage stops, and the watchdog reruns it
after the free tier's day turns (08:00 UTC), on the artifacts of the jobs that passed.
"""
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from core import llm
from core.logger import get_logger

log = get_logger("vision")
ROOT = Path(__file__).resolve().parent.parent
MODEL = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["vision"]["model"]
URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
BATCH = 24                  # 08.11: 24 frames a request (was 12) - the free day counts requests
MIN_GAP_S = 4.0
_last = [0.0]
REQUESTS = [0]              # requests sent this process (the prefilter's budget, owner 08.11)
# 08.11: 1950s-suburb spent the day at slot 96 of 223 and the rerun would have asked the same
# frames again. Every checked answer is kept (key: the line, the era, the frame's bytes) and the
# workflow carries the file from run to run (actions/cache), so each day goes on where the last stopped.
CACHE = Path(os.environ.get("VISION_CACHE") or ROOT / ".vision_cache.json")
_cache = None


def _store():
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _flush():
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(_cache or {}, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        log.info("vision cache not written (%s)", e)


def _key(line, era, kf):
    h = hashlib.sha1(f"{MODEL}|{era}|{line}|".encode("utf-8"))
    h.update(Path(kf).read_bytes())
    return h.hexdigest()

SYSTEM = """You check archive film frames for a documentary about everyday American life
in the 20th century. For each numbered pair you get the narration LINE and a FRAME that
would be on screen while it is spoken. Judge strictly but fairly: archive film rarely shows
the exact event, so a frame that honestly illustrates the line (same kind of place, people,
objects, activity, period) counts as a match; a frame about something else does not.
Return JSON only: {"results": [{"i": <n>, "match": true|false, "era": true|false,
"place_us": true|false, "clean": true|false, "note": "<few words>"}]}
- era: the frame looks like the period named in the line, within about ten years.
- clean: no watermark, no TV/channel/stock logo, no burned-in modern text or timecode.
  A title card of the film itself is NOT clean for our use."""


QUOTA_MARK = "VISION_QUOTA_SPENT"           # the watchdog looks for this in a failed job's log


class QuotaSpent(RuntimeError):
    pass


def key():
    return os.environ.get("GEMINI_API_KEY_AR", "").strip()


def _img(path, side=640):
    im = Image.open(path).convert("RGB")
    s = min(1.0, side / max(im.size))
    im = im.resize((round(im.width * s), round(im.height * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def check(pairs):
    """pairs: [(line_text, era_hint, keyframe_path)] -> [verdict dict] in the same order."""
    if not key():
        return [{"ok": False, "status": "unchecked", "note": "GEMINI_API_KEY_AR not set"} for _ in pairs]
    store = _store()
    keys = [_key(*p) for p in pairs]
    todo = [i for i, k in enumerate(keys) if k not in store]
    if len(todo) < len(pairs):
        log.info("vision: %d of %d frames answered before (cache)", len(pairs) - len(todo), len(pairs))
    fresh = _check([pairs[i] for i in todo])
    for i, v in zip(todo, fresh):
        if v.get("status") == "checked":
            store[keys[i]] = v
            _record(pairs[i], keys[i], v)
    if todo:
        _flush()
    got = dict(zip(todo, fresh))
    return [got[i] if i in got else store[keys[i]] for i in range(len(pairs))]


def _check(pairs):
    out = []
    for k in range(0, len(pairs), BATCH):
        part = pairs[k:k + BATCH]
        parts = []
        for i, (line, era, kf) in enumerate(part):
            parts.append({"text": f"PAIR {i}. LINE ({era}): {line}"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _img(kf)}})
        payload = {"contents": [{"role": "user", "parts": parts}],
                   "systemInstruction": {"parts": [{"text": SYSTEM}]},
                   "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                                        "maxOutputTokens": 4096}}
        res = None
        for attempt in range(5):
            wait = MIN_GAP_S - (time.monotonic() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
            try:
                req = urllib.request.Request(URL.format(model=MODEL), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key()})
                REQUESTS[0] += 1
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                res = {int(x["i"]): x for x in llm.extract_json(text).get("results", [])}
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and "PerDay" in e.read().decode("utf-8", "replace"):
                    log.warning("%s: Gemini's day for %s is spent; the stage stops and is rerun after "
                                "08:00 UTC", QUOTA_MARK, MODEL)
                    _keep(pairs[:k], out)
                    raise QuotaSpent(f"{QUOTA_MARK}: the day's Gemini vision requests are spent")
                time.sleep(6 * (attempt + 1))
            except Exception as e:                                   # noqa: BLE001
                log.info("vision retry (%s)", e)
                time.sleep(6 * (attempt + 1))
        for i in range(len(part)):
            r = (res or {}).get(i)
            if r is None:
                out.append({"ok": False, "status": "unchecked", "note": "no answer"})
            else:
                ok = all(bool(r.get(f)) for f in ("match", "era", "place_us", "clean"))
                out.append({"ok": ok, "status": "checked", **{f: r.get(f) for f in ("match", "era", "place_us", "clean")},
                            "note": r.get("note", "")})
    return out


def _keep(pairs, verdicts):
    """The answers got before the day ran out, kept for the next run."""
    store = _store()
    for p, v in zip(pairs, verdicts):
        if v.get("status") == "checked":
            k = _key(*p)
            store[k] = v
            _record(p, k, v)
    _flush()


# 08.11 (owner): a local CLIP/SigLIP prefilter is to take the clear cases off Gemini; its
# thresholds are set on Gemini's own answers, so every answer is logged with a small copy of
# its frame (work/vision_log, the run's "vision-log-<slug>" artifact)
LOG_DIR = ROOT / "work" / "vision_log"


def _record(pair, k, v):
    try:
        line, era, kf = pair
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        im = Image.open(kf).convert("RGB")
        im.thumbnail((384, 384))
        im.save(LOG_DIR / f"{k}.jpg", quality=85)
        with open(LOG_DIR / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": k, "line": line, "era_hint": era, "kf": str(kf), "ok": bool(v.get("ok")),
                                **{x: v.get(x) for x in ("match", "era", "place_us", "clean", "note")}},
                               ensure_ascii=False) + "\n")
    except Exception as e:                                           # noqa: BLE001
        log.info("vision log not written (%s)", e)
