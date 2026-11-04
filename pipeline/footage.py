"""The shot plan: which archive shot is on screen under every moment of the narration.

Rhythm (config/channel.json "edit"): a cut every 4-6 s, on line boundaries where
possible; no shot of the same film stretch twice within 3 minutes.

Once per episode (owner 04.11: a search and a scan per slot took over five
hours and the job timed out):
  1. films: the episode's 20-25 films (FILMS), chosen once - archive.search for
     the topic and for every chapter in its years (licence checked per item),
     ranked by the words they share with the whole script, colour, and a real
     HD scan; every chapter gets at least two of its own;
  2. shots: each of those films scanned once (clips.shots on its proxy), four
     at a time.
Per slot:
  3. ranking: the pool's shots by the words their film shares with the line and
     its chapter, colour, a real HD scan (720 lines or more) over an SD transfer;
  4. vision: the best candidates go to Gemini (pipeline/vision.py) with the
     line - match, era, place, no watermark or foreign logo - four slots a
     request, and the first that passes takes the slot. Without the key every
     pick stays "unchecked" and the QA gate will not release the episode.
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


FILMS = (20, 25)
SCAN_THREADS = 4
SLOTS_A_REQUEST = 4
_COLOUR = {}


def colourfulness(path):
    if path not in _COLOUR:
        a = np.asarray(Image.open(path).convert("RGB").resize((160, 120)), np.float32)
        rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
        _COLOUR[path] = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return _COLOUR[path]


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
                        "line": line["text"], "chapter_ctx": chapter,
                        **({"price": line["price"]} if k == 0 and line.get("price") else {})})
        t = e
    # the picture is continuous: every shot runs to the start of the next, so the
    # gaps between spoken lines are covered and picture and voice never drift
    if out:
        out[0]["start"] = 0.0
        for a, b in zip(out, out[1:]):
            a["end"] = b["start"]
        out[-1]["end"] = round(max(total, out[-1]["end"]), 3)
    # glue very short pieces onto their neighbour, so no cut is under 2.5 s
    merged = []
    for sl in out:
        if merged and sl["kind"] == "clip" and merged[-1]["kind"] == "clip" and sl["end"] - sl["start"] < 2.5 \
                and sl["film_break"] == merged[-1]["film_break"]:
            merged[-1]["end"] = sl["end"]
            if sl.get("price") and not merged[-1].get("price"):
                merged[-1]["price"] = sl["price"]
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
                                    "res": f["best_size"], "item": item} for a, b, kf in found]
            except Exception as e:                                   # noqa: BLE001
                log.info("%s: no shots (%s)", fid, e)
                self.shots[fid] = []
        return self.shots[fid]


def _score(item, ws):
    text = " ".join(str(item.get(k, "")) for k in ("title", "description", "subject")).lower()
    return sum(text.count(w) > 0 for w in ws) + (0.5 if item["colour"] else 0)


def episode_films(timeline, topic_years, query=None, n=FILMS):
    """The episode's films, chosen once: [(item, chapter titles it was found for)]."""
    from collections import Counter
    script = Counter(w for line in timeline for w in words(line["text"]))
    chapters = []
    for line in timeline:
        if line.get("chapter"):
            chapters.append({"chapter": line["chapter"], "words": Counter()})
        if chapters:
            chapters[-1]["words"].update(words(line["text"]))
    found = {}

    def add(items, tag):
        for it in items:
            found.setdefault(it["id"], [it, set()])[1].add(tag)

    if query:
        add(archive.search(query, *topic_years, rows=60), "topic")
    for ch in chapters:
        y0, y1 = _years(ch["chapter"], topic_years)
        top = [w for w, _ in ch["words"].most_common(8)]
        title = words(ch["chapter"].split("|")[0])
        add(archive.search(" OR ".join(dict.fromkeys(title + top))[:400] or "america", y0, y1, rows=40), ch["chapter"])

    def rank(entry):
        it, tags = entry
        text = " ".join(str(it.get(k, "")) for k in ("title", "description", "subject")).lower()
        # colour first (the channel's rule); a film found for many chapters is often only generic
        return (sum(min(c, 5) for w, c in script.items() if w in text) / 10 + 0.3 * len(tags)
                + (4.0 if it["colour"] else 0) + (2.0 if "topic" in tags else 0))

    ranked = sorted(found.values(), key=lambda e: -rank(e))
    chosen, seen = [], set()
    for ch in chapters:                                  # two of its own for every chapter first
        own = [e for e in ranked if ch["chapter"] in e[1] and e[0]["id"] not in seen][:2]
        for e in own:
            chosen.append(e)
            seen.add(e[0]["id"])
    for e in ranked:
        if len(chosen) >= n[1]:
            break
        if e[0]["id"] not in seen:
            chosen.append(e)
            seen.add(e[0]["id"])
    log.info("episode films: %d of %d found (%s)", len(chosen), len(found), ", ".join(e[0]["id"] for e in chosen))
    return chosen[:n[1]]


def plan(timeline, total, topic_years, work, picks=None, check_top=3, query=None):
    """The shot plan. `picks` ({slot index: {film, start}}) overrides the search for a slot."""
    from concurrent.futures import ThreadPoolExecutor
    work = Path(work)
    pool = Pool(work)
    sl = slots(timeline, total)
    films = episode_films(timeline, topic_years, query)
    with ThreadPoolExecutor(SCAN_THREADS) as ex:
        list(ex.map(lambda e: pool.film_shots(e[0]), films))
    shots_all = [sh for e in films for sh in pool.shots.get(e[0]["id"], [])]
    log.info("pool: %d shots from %d films", len(shots_all), len(films))
    used = []                       # (film, start, end, at) for the 3-minute rule

    def free(sh, s):
        return not any(f == sh["film"] and not (sh["end"] < a or sh["start"] > b)
                       and abs(s["start"] - at) < CFG["no_repeat_within_s"] for f, a, b, at in used)

    def ranked(s):
        d = s["end"] - s["start"]
        ws = words(s["line"] + " " + s.get("chapter_ctx", ""))
        y0, y1 = _years(s.get("chapter_ctx"), topic_years)
        cands = [sh for sh in shots_all if sh["end"] - sh["start"] >= min(d, 3.0) and free(sh, s)
                 and (not sh["item"].get("year") or y0 - 6 <= int(sh["item"]["year"]) <= y1 + 6)]
        cands.sort(key=lambda c: -(_score(c["item"], ws) + colourfulness(c["kf"]) / 60 * (2 if s["film_break"] else 1)
                                   + (0.7 if (c.get("res") or (0, 0))[1] >= 720 else 0)))
        return cands, f"{y0}-{y1}"

    todo = []
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
        todo.append(s)

    # a few slots a request: their best candidates checked together, then taken in order
    for i in range(0, len(todo), SLOTS_A_REQUEST):
        block = todo[i:i + SLOTS_A_REQUEST]
        tops = []
        for s in block:
            cands, era = ranked(s)
            tops.append((s, cands[:check_top + 2], era))
        pairs = [(s["line"], era, c["kf"]) for s, cs, era in tops for c in cs[:check_top]]
        verdicts = vision.check(pairs) if pairs else []
        vi = 0
        for s, cs, era in tops:
            vs = verdicts[vi:vi + len(cs[:check_top])]
            vi += len(cs[:check_top])
            d = s["end"] - s["start"]
            options = [(c, v) for c, v in zip(cs, vs) if free(c, s)]
            chosen = next(((c, v) for c, v in options if v.get("ok")), None) or \
                (options[0] if options else next(((c, {"status": "unchecked"}) for c in cs if free(c, s)), None))
            if not chosen:
                s["missing"] = True
                continue
            c, v = chosen
            s.update({"film": c["film"], "src_start": c["start"], "best": c["best"], "item": c["item"],
                      "kf": c["kf"], "vision": v or {"status": "unchecked"}})
            used.append((c["film"], c["start"], c["start"] + d, s["start"]))
        log.info("planned %d of %d slots", min(i + SLOTS_A_REQUEST, len(todo)), len(todo))
    return {"shots": sl, "films": [e[0]["id"] for e in films]}
