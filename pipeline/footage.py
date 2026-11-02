"""The shot plan: which archive shot is on screen under every moment of the narration.

Rhythm (config/channel.json "edit"): a cut every 4-6 s, on line boundaries where
possible; no shot of the same film stretch twice within 3 minutes.

Per slot:
  1. words: the line's content words, plus the chapter's years;
  2. films: archive.search for those words in the chapter's years (licence
     checked per item), colour first;
  3. shots: each film's cuts (clips.shots on its proxy), keyframes;
  4. ranking: words shared with the film's title, description and subjects;
  5. vision: the best candidates go to Gemini (pipeline/vision.py) with the
     line - match, era, place, no watermark or foreign logo - and the first that
     passes takes the slot. Without the key every pick stays "unchecked" and the
     QA gate will not release the episode.
Film breaks (narrator silent, pipeline/breathing.py) take the most vivid checked
shot of their chapter: the highest colourfulness of its keyframe. A chapter
opens on its map card (pipeline/cards.py) for the first 3.5 s of its break.
"""
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from core.logger import get_logger
from pipeline import archive, clips, vision

log = get_logger("footage")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["edit"]
CARD_S = 3.5
STOP = set("""a an the and or but of to in on at for with that this it is was were be been are as by from his her
their they them he she we you i our your its into over than then there here what who why how when where which
one two three four five six seven eight nine ten more most less many much very just only also even still so not
no all any every each some such had has have do did does would could should will can may might about after before
up down out off again new old year years day days time times way like really back first last now""".split())


def words(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3 and w not in STOP]


def colourfulness(path):
    a = np.asarray(Image.open(path).convert("RGB").resize((160, 120)), np.float32)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def slots(timeline, total):
    """Shots on the narration clock: voiced spans cut every 4-6 s, film breaks whole."""
    out, t = [], 0.0
    chapter = ""
    for line in timeline:
        if line.get("chapter"):
            chapter = line["chapter"]
        fb = line.get("film_before", 0)
        if fb:
            b0 = line["start"] - fb - 0.12
            b0 = max(t, b0)
            if line.get("chapter"):
                out.append({"start": round(b0, 3), "end": round(b0 + CARD_S, 3), "kind": "card",
                            "chapter": line["chapter"], "film_break": True})
                b0 += CARD_S
            if line["start"] - b0 > 0.5:
                out.append({"start": round(b0, 3), "end": round(line["start"], 3), "kind": "clip",
                            "film_break": True, "line": line["text"], "chapter_ctx": chapter})
            t = line["start"]
        # the voiced line, in 4-6 s pieces
        s, e = max(t, line["start"]), line["end"]
        n = max(1, round((e - s) / 5.0))
        for k in range(n):
            a = s + (e - s) * k / n
            b = s + (e - s) * (k + 1) / n
            out.append({"start": round(a, 3), "end": round(b, 3), "kind": "clip", "film_break": False,
                        "line": line["text"], "chapter_ctx": chapter})
        t = e
    if out and total - out[-1]["end"] > 0.05:
        out[-1]["end"] = round(total, 3)
    # glue very short pieces onto their neighbour, so no cut is under 2.5 s
    merged = []
    for sl in out:
        if merged and sl["kind"] == "clip" and merged[-1]["kind"] == "clip" and sl["end"] - sl["start"] < 2.5 \
                and sl["film_break"] == merged[-1]["film_break"]:
            merged[-1]["end"] = sl["end"]
        else:
            merged.append(sl)
    return merged


def _years(chapter, default):
    m = re.findall(r"(19\d\d)", chapter or "")
    if m:
        a, b = int(m[0]), int(m[-1])
        return max(1930, a - 4), min(1989, b + 8)
    return default


class Pool:
    """Candidate shots per film, scanned once and cached on disk."""

    def __init__(self, work):
        self.work = Path(work)
        self.films, self.shots = {}, {}

    def film_shots(self, item):
        fid = item["id"]
        if fid not in self.shots:
            try:
                f = archive.files(fid)
                found = clips.shots(f["proxy"], self.work / "kf" / fid, max_shots=40)
                self.shots[fid] = [{"film": fid, "start": a, "end": b, "kf": str(kf), "best": f["best"],
                                    "item": item} for a, b, kf in found]
            except Exception as e:                                   # noqa: BLE001
                log.info("%s: no shots (%s)", fid, e)
                self.shots[fid] = []
        return self.shots[fid]


def _score(item, ws):
    text = " ".join(str(item.get(k, "")) for k in ("title", "description", "subject")).lower()
    return sum(text.count(w) > 0 for w in ws) + (0.5 if item["colour"] else 0)


def plan(timeline, total, topic_years, work, picks=None, films_per_slot=3, check_top=3):
    """The shot plan. `picks` ({slot index: {film, start}}) overrides the search for a slot."""
    work = Path(work)
    pool = Pool(work)
    sl = slots(timeline, total)
    used = []                       # (film, start, end, at) for the 3-minute rule
    for k, s in enumerate(sl):
        if s["kind"] != "clip":
            continue
        d = s["end"] - s["start"]
        if picks and k in picks:
            p = picks[k]
            it = next((x for x in archive.search(f"identifier:{p['film']}", rows=3) if x["id"] == p["film"]), None) \
                or {"id": p["film"], "title": p["film"], "year": p.get("year"), "licence": "Public domain",
                    "page": f"https://archive.org/details/{p['film']}", "colour": True}
            s.update({"film": p["film"], "src_start": p["start"], "best": archive.files(p["film"])["best"],
                      "item": it, "vision": {"status": p.get("checked_by", "picked by hand")}})
            used.append((p["film"], p["start"], p["start"] + d, s["start"]))
            continue
        ws = words(s["line"] + " " + s.get("chapter_ctx", ""))
        y0, y1 = _years(s.get("chapter_ctx"), topic_years)
        items = archive.search(" OR ".join(ws[:6]) or "america", y0, y1, rows=30)
        items = sorted(items, key=lambda it: -_score(it, ws))[:films_per_slot]
        cands = []
        for it in items:
            for sh in pool.film_shots(it):
                if sh["end"] - sh["start"] < min(d, 3.0):
                    continue
                if any(f == sh["film"] and not (sh["end"] < a or sh["start"] > b) and abs(s["start"] - at) < CFG["no_repeat_within_s"]
                       for f, a, b, at in used):
                    continue
                cands.append(sh)
        cands.sort(key=lambda c: -(_score(c["item"], ws) + colourfulness(c["kf"]) / 60 * (2 if s["film_break"] else 1)))
        top = cands[:check_top]
        verdicts = vision.check([(s["line"], f"{y0}-{y1}", c["kf"]) for c in top]) if top else []
        chosen = next((c for c, v in zip(top, verdicts) if v["ok"]), None)
        v = next((v for c, v in zip(top, verdicts) if c is chosen), None) if chosen else (verdicts[0] if verdicts else None)
        chosen = chosen or (top[0] if top else None)
        if not chosen:
            s["missing"] = True
            continue
        s.update({"film": chosen["film"], "src_start": chosen["start"], "best": chosen["best"], "item": chosen["item"],
                  "kf": chosen["kf"], "vision": v or {"status": "unchecked"}})
        used.append((chosen["film"], chosen["start"], chosen["start"] + d, s["start"]))
    return {"shots": sl}
