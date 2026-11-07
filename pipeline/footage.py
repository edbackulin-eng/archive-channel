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

Owner 07.11, after 1950s-suburb's match_share of 0.306 (cartoons, castles, lab glass under
lines about Levittown):
  - films are judged for the topic BEFORE they are scanned: a model reads each found
    film's title, description and archive.org subjects and scores it 0-3 for the topic;
    under RELEVANT_MIN it is not taken. The search is wider: the topic's query, its name,
    its Wikipedia titles and every chapter, more rows each;
  - CHECK_TOP candidates a slot (6, was 3) go to the vision check;
  - a slot no candidate matches is never filled with one that does not: it takes another
    stretch of a shot already confirmed (its own chapter first), or the shot before it
    runs on; nothing unchecked goes on screen;
  - under MIN_CONFIRMED_MIN minutes of confirmed shots the topic is deferred (Deferred):
    the build takes the next topic.
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


FILMS = (20, 30)
CHECK_TOP = 6
RELEVANT_MIN = 2                 # of 0-3: "mostly about the topic" or better
MIN_CONFIRMED_MIN = 15.0


class Deferred(RuntimeError):
    """Too little confirmed film for the topic: it waits, the next topic is built."""


RELEVANCE = """You choose archive films for a documentary episode: "{name}" ({y0}-{y1}), about everyday life in
the United States. For each numbered film (title, year, archive.org subjects, description) score how much of it
is likely to show what this episode is about - its places, people, objects and daily life:
3 = about the topic itself; 2 = largely shows it (the same kind of places, homes, shops, families, period);
1 = touches it in passing; 0 = something else (cartoons, fairy tales, theatre, science, industry or war films
unrelated to the topic, instruction films about other subjects).
Return JSON only: {{"results": [{{"i": <n>, "score": 0-3, "why": "<few words>"}}]}}"""


def relevant(entries, name, years):
    """The found films a model scores RELEVANT_MIN or more for the topic, before any is scanned.
    entries: [[item, tags]]; each item gets its "relevance" score."""
    from core import llm
    keep = []
    for k in range(0, len(entries), 20):
        part = entries[k:k + 20]
        listing = "\n".join(
            f"{i}. {e[0].get('title', '')[:120]} ({e[0].get('year') or 'year?'}); subjects: "
            f"{str(e[0].get('subject') or '')[:160]}; {str(e[0].get('description') or '')[:350]}"
            for i, e in enumerate(part))
        try:
            ans = llm.complete_json(RELEVANCE.format(name=name, y0=years[0], y1=years[1]) + "\n\nFILMS:\n" + listing,
                                    max_tokens=3000, temperature=0.0)
            scores = {int(r["i"]): int(r.get("score", 0)) for r in ans.get("results", [])}
        except Exception as e:                                       # noqa: BLE001
            log.warning("relevance not judged for %d films (%s); they are not taken", len(part), e)
            scores = {}
        for i, e in enumerate(part):
            e[0]["relevance"] = scores.get(i)
            if (scores.get(i) or 0) >= RELEVANT_MIN:
                keep.append(e)
    log.info("relevance: %d of %d films are about '%s'", len(keep), len(entries), name)
    return keep
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


def episode_films(timeline, topic_years, query=None, n=FILMS, name=None, extra_queries=()):
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
        add(archive.search(query, *topic_years, rows=120), "topic")
    for q in extra_queries:                              # the topic's name and its Wikipedia titles
        ws = words(q)
        if ws:
            try:
                add(archive.search(" OR ".join(dict.fromkeys(ws))[:400], *topic_years, rows=60), "topic")
            except Exception as e:                                   # noqa: BLE001
                log.info("search '%s' failed: %s", q, e)
    for ch in chapters:
        y0, y1 = _years(ch["chapter"], topic_years)
        top = [w for w, _ in ch["words"].most_common(8)]
        title = words(ch["chapter"].split("|")[0])
        add(archive.search(" OR ".join(dict.fromkeys(title + top))[:400] or "america", y0, y1, rows=60), ch["chapter"])
    if name:                                             # not about the topic: not taken (owner 07.11)
        kept = relevant(list(found.values()), name, topic_years)
        found = {e[0]["id"]: e for e in kept}

    def rank(entry):
        it, tags = entry
        text = " ".join(str(it.get(k, "")) for k in ("title", "description", "subject")).lower()
        # colour first (the channel's rule); a film found for many chapters is often only generic
        return (sum(min(c, 5) for w, c in script.items() if w in text) / 10 + 0.3 * len(tags)
                + (4.0 if it["colour"] else 0) + (2.0 if "topic" in tags else 0)
                + 1.5 * (it.get("relevance") or 0))

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


def plan(timeline, total, topic_years, work, picks=None, check_top=CHECK_TOP, query=None, name=None,
         extra_queries=()):
    """The shot plan. `picks` ({slot index: {film, start}}) overrides the search for a slot."""
    from concurrent.futures import ThreadPoolExecutor
    work = Path(work)
    pool = Pool(work)
    sl = slots(timeline, total)
    films = episode_films(timeline, topic_years, query, name=name, extra_queries=extra_queries)
    with ThreadPoolExecutor(SCAN_THREADS) as ex:
        list(ex.map(lambda e: pool.film_shots(e[0]), films))
    shots_all = [sh for e in films for sh in pool.shots.get(e[0]["id"], [])]
    log.info("pool: %d shots from %d films", len(shots_all), len(films))
    used = []                       # (film, start, end, at) for the 3-minute rule

    def free(sh, s):
        return not any(f == sh["film"] and not (sh["end"] < a or sh["start"] > b)
                       and abs(s["start"] - at) < CFG["no_repeat_within_s"] for f, a, b, at in used)

    def free_range(film, a0, b0, s):
        return not any(f == film and not (b0 < a or a0 > b) and abs(s["start"] - at) < CFG["no_repeat_within_s"]
                       for f, a, b, at in used)

    confirmed = []                  # (pool shot, its verdict, chapter): shots the vision check passed
    rejected = {}                   # id(pool shot) -> times the check turned it down

    def reuse(s):
        """Another stretch of a shot already confirmed, its own chapter first; None if none is free."""
        d = s["end"] - s["start"]
        own = [c for c in confirmed if c[2] == s.get("chapter_ctx")]
        for sh, v, _ch in own + [c for c in confirmed if c not in own]:
            if sh["end"] - sh["start"] < min(d, 3.0):
                continue
            off = sh["start"]
            while off + min(d, sh["end"] - sh["start"]) <= sh["end"] + 1e-6:
                if free_range(sh["film"], off, off + d, s):
                    return sh, off, v
                off += 1.0
        return None

    def ranked(s):
        d = s["end"] - s["start"]
        ws = words(s["line"] + " " + s.get("chapter_ctx", ""))
        y0, y1 = _years(s.get("chapter_ctx"), topic_years)
        cands = [sh for sh in shots_all if sh["end"] - sh["start"] >= min(d, 3.0) and free(sh, s)
                 and (not sh["item"].get("year") or y0 - 6 <= int(sh["item"]["year"]) <= y1 + 6)]
        # a shot turned down twice is not asked again, and one confirmed already is asked last
        # (it can still take a slot as a confirmed stretch): the checks spread over the pool
        cands = [c for c in cands if rejected.get(id(c), 0) < 2]
        cands.sort(key=lambda c: (any(x[0] is c for x in confirmed),
                                  -(_score(c["item"], ws) + colourfulness(c["kf"]) / 60 * (2 if s["film_break"] else 1)
                                    + (0.7 if (c.get("res") or (0, 0))[1] >= 720 else 0))))
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
            for c, v in zip(cs, vs):
                if v.get("ok") and not any(x[0] is c for x in confirmed):
                    confirmed.append((c, v, s.get("chapter_ctx")))
                elif v.get("status") == "checked" and not v.get("ok"):
                    rejected[id(c)] = rejected.get(id(c), 0) + 1
            options = [(c, v) for c, v in zip(cs, vs) if free(c, s)]
            chosen = next(((c, v) for c, v in options if v.get("ok")), None)
            if chosen:
                c, v = chosen
                s.update({"film": c["film"], "src_start": c["start"], "src_end": c["end"], "best": c["best"],
                          "item": c["item"], "kf": c["kf"], "vision": v})
                used.append((c["film"], c["start"], c["start"] + d, s["start"]))
                continue
            # no candidate matches: never one that does not (owner 07.11) - another stretch of a
            # confirmed shot, else the shot before runs on (merged afterwards)
            got = reuse(s)
            if got:
                sh, off, v = got
                s.update({"film": sh["film"], "src_start": off, "src_end": sh["end"], "best": sh["best"],
                          "item": sh["item"], "kf": sh["kf"],
                          "vision": {**v, "note": f"another stretch of a confirmed shot ({v.get('note', '')})",
                                     "reused": True}})
                used.append((sh["film"], off, off + d, s["start"]))
            else:
                s["missing"] = True
        log.info("planned %d of %d slots", min(i + SLOTS_A_REQUEST, len(todo)), len(todo))
    sl = _run_on(sl)
    seen, conf_s = set(), 0.0
    for sh, _v, _ch in confirmed:
        k = (sh["film"], sh["start"])
        if k not in seen:
            seen.add(k)
            conf_s += sh["end"] - sh["start"]
    log.info("confirmed film: %.1f min in %d shots; %d slots run on from the shot before", conf_s / 60, len(seen),
             sum(1 for s in sl if s.get("ran_on")))
    if conf_s / 60 < MIN_CONFIRMED_MIN:
        raise Deferred(f"DEFERRED: {conf_s / 60:.1f} min of confirmed film for the topic, under "
                       f"{MIN_CONFIRMED_MIN:.0f}")
    return {"shots": sl, "films": [e[0]["id"] for e in films], "confirmed_min": round(conf_s / 60, 1)}


def _run_on(sl):
    """A slot left without a matching shot is taken by the shot before it, which runs on."""
    out = []
    for s in sl:
        prev = out[-1] if out else None
        if s.get("missing") and prev is not None and prev["kind"] == "clip" and prev.get("film"):
            prev["end"] = s["end"]
            prev["ran_on"] = True
            if s.get("price"):
                if not prev.get("price"):
                    prev["price"] = s["price"]
                else:                                        # the next shot carries it
                    s["_carry_price"] = s["price"]
            if s.get("_carry_price"):
                prev.setdefault("_carry", s["_carry_price"])
            continue
        if prev is not None and prev.get("_carry") and not s.get("price") and s["kind"] == "clip":
            s["price"] = prev.pop("_carry")
        out.append(s)
    # one with no clip before it (after a chapter card, or the first): the shot after it starts early
    final = []
    for k, s in enumerate(out):
        if s.get("missing"):
            nxt = next((x for x in out[k + 1:] if x["kind"] == "clip" and x.get("film") and not x.get("missing")), None)
            if nxt is not None and out.index(nxt) == k + 1:
                nxt["start"] = s["start"]
                nxt["ran_on"] = True
                if s.get("price") and not nxt.get("price"):
                    nxt["price"] = s["price"]
                continue
        final.append(s)
    for s in final:
        s.pop("_carry", None)
        s.pop("_carry_price", None)
    return final
