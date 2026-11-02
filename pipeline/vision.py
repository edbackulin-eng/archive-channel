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
"""
import base64
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
BATCH = 12
MIN_GAP_S = 4.0
_last = [0.0]

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
                                        "maxOutputTokens": 2048}}
        res = None
        for attempt in range(5):
            wait = MIN_GAP_S - (time.monotonic() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.monotonic()
            try:
                req = urllib.request.Request(URL.format(model=MODEL), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": key()})
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                res = {int(x["i"]): x for x in llm.extract_json(text).get("results", [])}
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and "PerDay" in e.read().decode("utf-8", "replace"):
                    log.warning("Gemini daily quota spent; the rest stays unchecked")
                    break
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
