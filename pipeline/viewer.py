"""The last look before upload, a viewer's (owner 10.11): with no approve any more, an episode goes
up and is scheduled on its own only after it passed QA and this gate.

  check(video, timeline, work)   the finished video two minutes at a time - a contact sheet (a
                                 frame every 5 s, its time on it) and the narration of those
                                 minutes - and Gemini marks what would make a viewer switch off:
                                 a picture plainly against the words, the same shot twice, a logo
                                 or watermark, black or broken frames, something obviously modern.
                                 More than VIEWER_MAX marks and the episode does not go up.
  judge(paths, words, subject)   of the (up to three) thumbnails, the one a viewer of the niche
                                 would click, showing the subject with its words readable.

Both use the channel's vision key and model (config "vision"), flash after an overloaded model.
A spent day raises vision.QuotaSpent: the stage stops and runs again after the reset, so an
episode never goes up unlooked at.
"""
import base64
import io
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from core import llm
from core.logger import get_logger
from pipeline import vision

log = get_logger("viewer")
VIEWER_MAX = 3
EVERY_S, SPAN_S = 5, 120
MODELS = (vision.MODEL, "gemini-flash-latest")

VIEWER = """You are a critical viewer of an English YouTube documentary about everyday American life in
the mid-20th century, told over restored public-domain archive film. You see two minutes of it: a
contact sheet (one frame every 5 seconds, the time on each) and the narration of those minutes.

Mark only what would make a viewer switch off or stop trusting the film:
- a picture that plainly contradicts what is being said (archive film that only loosely
  illustrates the words is normal and fine);
- the same shot shown twice within these minutes;
- a watermark, TV or stock-agency logo, burned-in modern text or a timecode;
- black, blank, frozen or broken frames;
- anything obviously from after 1975 (cars, phones, clothes, signs);
- a title card or credits of another film.
Chapter cards and price cards made for this video are fine.

Return JSON only: {{"marks": [{{"t": "m:ss", "kind": "<one word>", "why": "<a few words>"}}]}} -
an empty list when nothing would.

NARRATION:
{text}"""

JUDGE = """These are {n} thumbnail candidates for an English YouTube video about everyday American life in
the 1950s (archive colour film). Its words: "{words}". The picture must show: {subject}.
Pick the one a viewer of nostalgic 1950s-America videos would most likely click: the subject clear
and large, the words readable at phone size, a vivid restored colour frame, nothing ugly or confusing.
Return JSON only: {{"best": <number of the thumbnail, from 0>, "why": "<a few words>"}}"""


def _b64(p, side=1280):
    im = Image.open(p).convert("RGB")
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode()


def _ask(prompt, images):
    """Gemini's JSON answer for a prompt and images; None if no model could answer."""
    parts = [{"text": prompt}] + [{"inline_data": {"mime_type": "image/jpeg", "data": _b64(p)}} for p in images]
    payload = {"contents": [{"role": "user", "parts": parts}],
               "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json",
                                    "maxOutputTokens": 2048}}
    spent = 0
    for model in MODELS:
        for attempt in range(3):
            try:
                vision.REQUESTS[0] += 1
                req = urllib.request.Request(vision.URL.format(model=model), data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json", "x-goog-api-key": vision.key()})
                with urllib.request.urlopen(req, timeout=180) as r:
                    d = json.loads(r.read())
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                return llm.extract_json(text)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                if e.code == 429 and "PerDay" in body:
                    spent += 1
                    break                                   # this model's day: the next model
                time.sleep(20 * (attempt + 1))
            except Exception as e:                          # noqa: BLE001
                log.info("viewer: %s (%s)", model, e)
                time.sleep(10 * (attempt + 1))
    if spent == len(MODELS):
        raise vision.QuotaSpent(f"{vision.QUOTA_MARK}: the day's Gemini requests are spent (viewer gate)")
    return None


def _text_between(timeline, t0, t1):
    return " ".join(l["text"] for l in timeline if l["end"] > t0 and l["start"] < t1)


def check(video, timeline, work):
    """{"pass", "marks", "detail", "sheets"}: the viewer's marks, two minutes at a time."""
    work = Path(work)
    tmp = work / "viewer"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps=1/{EVERY_S},scale=320:-2", str(tmp / "f%04d.jpg")], check=False)
    frames = sorted(tmp.glob("f*.jpg"))
    per = SPAN_S // EVERY_S
    marks, unanswered, sheets = [], 0, 0
    for k in range(0, len(frames), per):
        chunk = frames[k:k + per]
        t0 = k * EVERY_S
        sheet = Image.new("RGB", (6 * 320, ((len(chunk) + 5) // 6) * 196), (10, 10, 10))
        d = ImageDraw.Draw(sheet)
        for j, f in enumerate(chunk):
            im = Image.open(f).convert("RGB").resize((320, 180))
            x, y = (j % 6) * 320, (j // 6) * 196
            sheet.paste(im, (x, y + 16))
            t = t0 + j * EVERY_S
            d.text((x + 4, y + 2), f"{t // 60}:{t % 60:02d}", fill=(255, 214, 0))
        p = tmp / f"sheet_{k:03d}.jpg"
        sheet.save(p, quality=80)
        sheets += 1
        a = _ask(VIEWER.format(text=_text_between(timeline, t0, t0 + SPAN_S)[:2500]), [p])
        if not isinstance(a, dict):
            unanswered += 1
            continue
        for m in a.get("marks") or []:
            if isinstance(m, dict):
                marks.append({"t": m.get("t"), "kind": m.get("kind"), "why": str(m.get("why", ""))[:120],
                              "minute": t0 // 60})
    # a minute nobody looked at is not passed: the gate stays shut until every sheet was seen
    ok = unanswered == 0 and len(marks) <= VIEWER_MAX
    detail = (f"{len(marks)} places a viewer would switch off (max {VIEWER_MAX})"
              + (f", {unanswered} of {sheets} sheets unanswered" if unanswered else "") + ": "
              + (str([(m["t"], m["kind"], m["why"][:50]) for m in marks[:8]]) if marks else "none"))
    log.info("viewer gate: %s", detail)
    return {"pass": ok, "marks": marks, "detail": detail, "sheets": sheets}


def judge(paths, words, subject):
    """The index of the thumbnail a viewer would click, or None when the judge could not answer."""
    paths = [p for p in paths if Path(p).exists()]
    if len(paths) < 2:
        return 0 if paths else None
    a = _ask(JUDGE.format(n=len(paths), words=words, subject=subject), paths)
    try:
        i = int((a or {}).get("best"))
    except (TypeError, ValueError):
        return None
    log.info("thumbnail judge: %s of %d (%s)", i, len(paths), (a or {}).get("why", ""))
    return i if 0 <= i < len(paths) else None
