"""Smoke test: the whole chain on a few seconds, in a few minutes (smoke.yml).

Only a passing run moves dev to main. Checks:
  1. every module imports;
  2. delivery and breathing: episode 1's script parses, its film breaks give 20-30 %
     without voice (or the length cap's share), all 3-8 s, the hook unbroken;
  3. sources: an archive.org search returns only items with a free licence, and
     a forbidden source (British Pathe) is filtered out;
  4. footage: a real public-domain shot is cut over HTTP and restored to
     1920x1080 with no black border;
  5. voice, map card, edit and sound: two narrated lines, a film break with the
     film's own sound, a chapter map - mixed and checked by the QA gate
     (loudness, breaks);
  6. thumbnail: composed with the corner empty and the words readable small;
  7. prices: BLS CPI-U sums, episode 1's tags (in the hook, 6 or more, never 6
     minutes without), a tag rendered and seen by QA, the BLS source;
  8. footage at its highest real resolution, interlaced transfers inverse-telecined;
  9. thumbnail chain: Real-ESRGAN x4 on the CPU, and a blurred picture turned down.
"""
import importlib
import json
import os
import pkgutil
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
W = ROOT / "work" / "smoke"
failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def plain_line(delivery, text):
    return delivery.plain_text(text)


def main():
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    for pkg in ("core", "pipeline", "automation"):
        for m in pkgutil.iter_modules([str(ROOT / pkg)]):
            if m.name in ("niche", "niche_refine", "saturation", "saturation_refine"):
                continue
            try:
                importlib.import_module(f"{pkg}.{m.name}")
            except Exception as e:                                   # noqa: BLE001
                check(f"import {pkg}.{m.name}", False, repr(e)[:200])
    check("imports", not failures)

    from core import delivery, narrate
    from core.voice import SAMPLE_RATE
    from pipeline import archive, breathing, cards, clips, edit, qa, thumbnail
    cfg = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
    script = (ROOT / "episodes" / "1950s-suburb" / "script.txt").read_text(encoding="utf-8")
    lines = delivery.parse(script)
    share = breathing.share_of(script)
    br = [l.film_before for l in lines if l.film_before]
    first_ch = next(i for i, l in enumerate(lines) if l.chapter)
    check("breathing", cfg["breathing"]["min_share"] - 0.005 <= share <= cfg["breathing"]["max_share"]
          and all(3 <= b <= 8 for b in br) and not any(l.film_before for l in lines[:first_ch]),
          f"{share:.1%} without voice, {len(br)} breaks, hook unbroken")
    # A build may be rerun - the watchdog reruns a failed one three times a day - and
    # stage_narrate places the breaks and writes the script back every time. Placing
    # twice used to double every [film N]: 19.9 % went to 33.2 %, and the episode
    # stretched until two stretches ran over six minutes with no price on screen.
    once = breathing.place(script)
    check("breathing: placing twice is placing once", breathing.place(once) == once
          and breathing.share_of(once) >= cfg["breathing"]["min_share"] - 0.005,
          f"{once.count('[film')} breaks either way, {breathing.share_of(once):.1%}")

    items = archive.search("kitchen", 1950, 1960, rows=30)
    check("sources: free licences only", items and all(archive.licence_ok(i["licence_url"]) for i in items),
          f"{len(items)} items")
    check("sources: forbidden names filtered", bool(archive.FORBIDDEN.search("a British Pathe newsreel")))

    f = archive.files("American1958")
    clip = W / "clip.mp4"
    clips.cut(f["best"], 120.0, 3.0, clip, W / "cut")
    info = clips.probe_info(clip)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    import subprocess
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "1.5", "-i", str(clip), "-frames:v", "1",
                    str(W / "f.png")], check=True)
    a = np.asarray(Image.open(W / "f.png").convert("L"), np.float32)
    edges = min(a[:12].mean(), a[-12:].mean(), a[:, :12].mean(), a[:, -12:].mean())
    check("footage: cut + restore", (v["width"], v["height"]) == (1920, 1080) and edges > 20,
          f"{v['width']}x{v['height']}, darkest edge {edges:.0f}")

    text = "> In 1958, a new kind of kitchen arrived.\n[film 4]\n## Test | Long Island, New York | 1958\nIt came with a promise."
    audio, tl = narrate.synthesize(narrate.load_kokoro(), text)
    sf.write(W / "n.wav", audio, SAMPLE_RATE)
    total = len(audio) / SAMPLE_RATE
    t1 = tl[1]["start"]
    plan = {"shots": [
        {"start": 0.0, "end": round(t1 - 4.0, 3), "kind": "clip", "clip": str(clip), "film": "American1958",
         "src_start": 120.0, "year": 1958, "place": "United States", "item": {"licence": "Public domain"},
         "vision": {"ok": True, "status": "smoke"}},
        {"start": round(t1 - 4.0, 3), "end": round(t1 - 1.5, 3), "kind": "card",
         "chapter": "Test | Long Island, New York | 1958", "film_break": True},
        {"start": round(t1 - 1.5, 3), "end": round(total, 3), "kind": "clip", "clip": str(clip), "film": "American1958",
         "src_start": 121.0, "year": 1958, "place": "United States", "item": {"licence": "Public domain"},
         "vision": {"ok": True, "status": "smoke"}, "film_break": False}]}
    out = W / "smoke.mp4"
    edit.render(plan, W / "n.wav", ROOT / "assets" / "music" / "lobby-time.mp3", out, W / "edit")
    q = qa.check(plan, tl, out, full_episode=False)
    check("edit + sound + QA", q["loudness"]["pass"] and abs(clips.duration(out) - total) < 0.6,
          f"{q['loudness']['lufs']} LUFS, {clips.duration(out):.1f}/{total:.1f} s")
    check("map card", cards.map_frame("Test | Long Island, New York | 1958").size == (1920, 1080))
    r = thumbnail.compose(Image.open(W / "f.png"), "1958", "THE NEW KITCHEN", W / "t.jpg", side="left")
    check("thumbnail", r["corner_empty"] and r["readable_small"], json.dumps(r)[:160])

    # prices and daily life: BLS CPI-U sums, the tags in the script, on screen and in QA
    from pipeline import prices, scriptwriter, sources, upscale
    lev = prices.in_today(7990, 1949)
    check("prices: CPI-U, $7,990 of 1949 in today's dollars", 100_000 < lev < 115_000 and prices.today_year() >= 2025,
          f"{prices.nice(lev)} in {prices.today_year()} dollars")
    check("prices: [today] is spoken, never written by the model",
          prices.spoken_today("[today 400 1947]").startswith("about $") and "[" not in plain_line(delivery, "a [today 400 1947] b"))
    n_tags, missing = scriptwriter.price_gaps(script)
    check("prices: episode 1 - in the hook, 6 or more, never 6 minutes without", not missing, f"{n_tags} tags {missing}")
    check("prices: BLS consumer spending source", "butter" in sources.bls_spending((1947, 1958))[0]["text"])
    plan["shots"][0]["price"] = {"amount": "7990", "year": 1949, "label": "A new Levitt house"}
    out2 = W / "smoke_price.mp4"
    edit.render(plan, W / "n.wav", ROOT / "assets" / "music" / "lobby-time.mp3", out2, W / "edit2")
    q2 = qa.check(plan, tl, out2, full_episode=False)
    check("prices: a tag rendered, checked by QA", q2["prices"]["rendered"] and q2["prices"]["tags"] == 1, str(q2["prices"]))
    # the film at its highest real resolution; an upscaled derivative never beats its original
    fake = [{"name": "a_512kb.mp4", "format": "512Kb MPEG4", "source": "derivative", "width": "320", "height": "240", "size": "9"},
            {"name": "a.mp4", "format": "h.264", "source": "derivative", "width": "640", "height": "480", "size": "50"},
            {"name": "a.mpg", "format": "MPEG1", "source": "original", "width": "320", "height": "240", "size": "90"}]
    best, real = archive.pick_best(fake, "x/", probe=lambda u: (0, 0, 0))
    check("footage: highest real resolution, original over its upscale", best["name"] == "a.mpg" and real == (320, 240),
          f"{best['name']} {real}")
    check("footage: interlaced transfers get an inverse telecine",
          "fieldmatch" in clips.field_filter(archive.files("Wonderfu1960")["best"]))
    # the thumbnail chain: Real-ESRGAN on the CPU, the sharpness gate
    small = Image.open(W / "f.png").convert("RGB").resize((96, 54))
    big = upscale.x4(small)
    check("thumbnail: Real-ESRGAN x4 on the CPU", big.size == (384, 216), str(big.size))
    blur = Image.open(W / "t.jpg").filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(6))
    check("thumbnail: QA turns a blurred picture down",
          thumbnail.detail(blur, (336, 189), (0.5, 0, 1, 1)) < thumbnail.SHARP_MIN, "")
    # release: the first two wait for the owner, the rest wait their turn, then follow in order
    import tempfile
    from automation import release
    from pipeline import youtube
    tmp = Path(tempfile.mkdtemp())
    st = tmp / "episodes.json"
    st.write_text(json.dumps([{"slug": s, "state": "built"} for s in ("a", "b", "c")]))
    for s in ("a", "b", "c"):
        (tmp / "episodes" / s / "out").mkdir(parents=True)
        (tmp / "episodes" / s / "out" / "metadata.json").write_text(json.dumps({"title": s, "description": "", "tags": []}))
    saved = (release.ROOT, release.rows, release.save, youtube.upload, youtube.set_thumbnail, youtube.schedule)
    release.ROOT, release.rows, release.save = tmp, (lambda: json.loads(st.read_text())), (lambda r: st.write_text(json.dumps(r)))
    youtube.upload, youtube.set_thumbnail, youtube.schedule = (lambda *a: "v" + a[1]), (lambda *a: None), (lambda *a: None)
    try:
        release.upload()
        s2 = [(r["state"], r.get("publish_at")) for r in json.loads(st.read_text())]
    finally:
        release.ROOT, release.rows, release.save, youtube.upload, youtube.set_thumbnail, youtube.schedule = saved
    # owner 10.11: no approve (qa.approve_first_episodes 0) - each goes up scheduled, in order
    check("release: no approve - every episode scheduled at once on the next free slots, in order",
          release.CFG["qa"]["approve_first_episodes"] == 0
          and all(x[0] == "scheduled" for x in s2) and s2[0][1] < s2[1][1] < s2[2][1], f"{s2}")
    # owner 07.11: after a QA failure no build starts again until the code or the plan changes
    from automation import pause
    saved_p = pause.PAUSE, pause.STATE
    try:
        pause.PAUSE, pause.STATE = W / "pause.json", W / "episodes.json"
        pause.STATE.write_text(json.dumps([{"slug": "a", "state": "qa-failed", "qa_failed": ["length"]}]))
        fp = pause.fingerprint()
        pause.from_state("a")
        held = pause.active(fp)
        kept = pause.lift_if_changed(fp) is False and pause.active(fp)
        lifted = pause.lift_if_changed(fp + "x") and pause.active(fp) is None and not pause.PAUSE.exists()
    finally:
        pause.PAUSE, pause.STATE = saved_p
    check("pause: a QA failure pauses builds; the same code keeps the pause, a change lifts it",
          bool(held) and "length" in held["why"] and bool(kept) and lifted and len(fp) == 16, str(held))
    # owner 07.11: Gemini's spent day stops the stage (rerun after 08:05 UTC); a long script is cut
    from datetime import datetime as _dt, timezone as _tz
    from automation import watchdog
    at = _dt(2026, 11, 6, 18, 50, tzinfo=_tz.utc)
    check("watchdog: a quota stop waits for 08:05 UTC, then reruns",
          not watchdog.after_reset(at, _dt(2026, 11, 7, 7, 0, tzinfo=_tz.utc))
          and watchdog.after_reset(at, _dt(2026, 11, 7, 8, 10, tzinfo=_tz.utc)))
    # 08.11: a spent Gemini day holds every build until 08:05 UTC, then the stopped topics go first
    from automation import quota_hold
    import tempfile as _tf0
    saved_h = quota_hold.HOLD
    try:
        quota_hold.HOLD = Path(_tf0.mkdtemp()) / "quota_hold.json"
        h1 = quota_hold.set_("a", _dt(2026, 11, 7, 23, 14, tzinfo=_tz.utc))
        h2 = quota_hold.set_("b", _dt(2026, 11, 7, 23, 20, tzinfo=_tz.utc))
        held_h = quota_hold.active(_dt(2026, 11, 8, 7, 0, tzinfo=_tz.utc))
        free_h = quota_hold.active(_dt(2026, 11, 8, 8, 10, tzinfo=_tz.utc))
    finally:
        quota_hold.HOLD = saved_h
    check("quota hold: until 08:05 UTC, the stopped topics in order; a spent text day counts too",
          h1["until"] == "2026-11-08T08:05:00Z" and h2["slugs"] == ["a", "b"] and bool(held_h) and free_h is None
          and quota_hold.spent("no provider answered: groq: HTTP 413; gemini: HTTP 429; gemini: HTTP 429")
          and quota_hold.spent("VISION_QUOTA_SPENT: x") and not quota_hold.spent("groq: HTTP 500"), str(h2))
    from pipeline import scriptwriter, vision
    # 08.11: a short draft is expanded up to five times, the claims of the new lines kept with the old
    saved_cj = scriptwriter.llm.complete_json
    calls_x = []

    def _grow(prompt, **k):
        calls_x.append(1)
        n = 700 + 300 * len(calls_x)
        return {"script": "## One | Place | 1950\n" + "\n".join(["word " * 20] * (n // 20)),
                "claims": [{"claim": f"c{len(calls_x)}"}]}
    try:
        scriptwriter.llm.complete_json = _grow
        grown = scriptwriter.expand({"script": "## One | Place | 1950\n" + "word " * 700, "claims": [{"claim": "c0"}]},
                                    "P")
    finally:
        scriptwriter.llm.complete_json = saved_cj
    check("script: a short draft is expanded up to five rounds, claims kept",
          len(calls_x) == 4 and len(grown["claims"]) == 5 and scriptwriter.words(grown["script"]) >= 1800
          and scriptwriter.est_minutes(grown["script"]) >= scriptwriter.MIN_EST_MIN,
          f"{len(calls_x)} rounds, {scriptwriter.words(grown['script'])} words")
    check("vision: a spent day is QuotaSpent, not unchecked", issubclass(vision.QuotaSpent, RuntimeError)
          and vision.QUOTA_MARK == "VISION_QUOTA_SPENT")
    # 08.11: answers are kept between runs, and the ones got before a spent day too
    import tempfile as _tf
    _d = Path(_tf.mkdtemp())
    for _n in range(30):
        (_d / f"k{_n}.jpg").write_bytes(b"frame %d" % _n)
    _pairs = [(f"line {n}", "1950-1955", str(_d / f"k{n}.jpg")) for n in range(30)]
    saved_v = (vision.CACHE, vision._cache, vision._check, os.environ.get("GEMINI_API_KEY_AR"))
    asked = []
    try:
        vision.CACHE, vision._cache = _d / "cache.json", None
        os.environ["GEMINI_API_KEY_AR"] = "smoke"
        vision._check = lambda ps: (asked.append(len(ps)) or
                                    [{"ok": True, "status": "checked", "note": p[0]} for p in ps])
        first = vision.check(_pairs[:20])
        vision._cache = None                                   # a new run: the file only
        second = vision.check(_pairs)
    finally:
        vision.CACHE, vision._cache, vision._check = saved_v[:3]
        if saved_v[3] is None:
            os.environ.pop("GEMINI_API_KEY_AR", None)
        else:
            os.environ["GEMINI_API_KEY_AR"] = saved_v[3]
    check("vision: answered frames are kept between runs; only new ones are asked",
          asked == [20, 10] and [v["note"] for v in second] == [p[0] for p in _pairs] and len(first) == 20,
          str(asked))
    body = ["## One | Place | 1950", "[price 7990 1949 | A new house]"] + \
        [f"line {i} " + "word " * 18 for i in range(124)] + ["The Times reported it."]      # ~2500 words
    long_s = "\n".join(["Hook line about a price."] + body)
    hook_only = "\n".join(["word " * 20] * 125)                                              # nothing deletable
    saved_llm = scriptwriter.llm.complete_json
    try:
        scriptwriter.llm.complete_json = lambda *a, **k: {"delete": [5, 6, 7]}
        cut = scriptwriter.shorten(long_s)
        try:
            scriptwriter.shorten(hook_only)
            refused = False
        except RuntimeError as e:
            refused = "LENGTH" in str(e)
    finally:
        scriptwriter.llm.complete_json = saved_llm
    kept = cut.splitlines()
    check("script: over WORDS_MAX is cut by whole lines to fit (hook, tags, attribution kept), or stops with LENGTH",
          scriptwriter.WORDS_MIN <= scriptwriter.words(cut) <= scriptwriter.WORDS_MAX and refused
          and kept[0] == "Hook line about a price." and "[price 7990 1949 | A new house]" in kept
          and "line 0 " + "word " * 18 in kept and "The Times reported it." in kept
          and ("line 122 " + "word " * 18) in kept and "line 4 " + "word " * 18 not in kept,
          str(scriptwriter.words(cut)))
    # owner 07.11: a script is expanded up to the full 15-minute minimum, not only under 90 % of its words
    saved_llm = scriptwriter.llm.complete_json
    short = "## One | Place | 1950\n" + "\n".join(f"Line {i} " + "word " * 15 for i in range(85))    # ~1450 words, ~14 min at the voice's pace
    longer = "## One | Place | 1950\n" + "\n".join(f"Line {i} " + "word " * 15 for i in range(128))  # ~2170 words
    asked = []
    try:
        def fake(prompt, **k):
            asked.append(prompt)
            return {"script": longer if asked[1:] else short, "claims": []}
        scriptwriter.llm.complete_json = fake
        out = scriptwriter.write({"name": "x", "years": [1950, 1955]}, [{"title": "s", "url": "u", "text": "t"}])
    finally:
        scriptwriter.llm.complete_json = saved_llm
    check("script: under the 15-minute minimum it is sent back for more",
          scriptwriter.est_minutes(short) < scriptwriter.MIN_EST_MIN and len(asked) >= 2
          and scriptwriter.words(out["script"]) > scriptwriter.words(short),
          f"{scriptwriter.words(short)} words ~{scriptwriter.est_minutes(short):.1f} min -> "
          f"{scriptwriter.words(out['script'])} words")
    # owner 07.11: films judged for the topic before the scan; never a shot that does not match;
    # under 15 min of confirmed film the topic is deferred
    from pipeline import footage, archive as _ar, clips as _cl
    from core import llm as _llm
    saved_f = (_ar.search, _ar.files, _cl.shots, _llm.complete_json, footage.vision.check)
    fk = W / "fk"
    films = {f"good{i}": "Suburban homes and families" for i in range(16)}
    films.update({f"bad{i}": "Cartoon castle fairy tale" for i in range(3)})

    def _shots(proxy, d, max_shots=40):
        out = []
        for i in range(30):
            f = Path(d) / f"k{i}.jpg"
            f.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (120, 80, 40)).save(f)
            out.append((i * 8.0, i * 8.0 + 8.0, f))
        return out
    try:
        _ar.search = lambda *a, **k: [{"id": i, "title": v, "year": 1952, "colour": True, "description": v,
                                       "subject": "", "licence": "PD", "page": ""} for i, v in films.items()]
        _ar.files = lambda fid: {"proxy": "x", "best": "b", "best_size": (640, 480)}
        _cl.shots = _shots
        _llm.complete_json = lambda prompt, **k: {"results": [
            {"i": i, "score": 0 if "Cartoon" in l else 3}
            for i, l in enumerate(l for l in prompt.split("FILMS:\n")[1].splitlines() if l.strip())]}
        footage.vision.check = lambda pairs: [{"ok": sum(map(ord, str(kf))) % 3 == 0, "status": "checked", "note": "x"}
                                              for _l, _e, kf in pairs]
        tl = [{"start": i * 5.0, "end": i * 5.0 + 5.0, "text": f"suburb family house line {i}",
               **({"chapter": "## Homes | Long Island | 1950"} if i == 0 else {})} for i in range(230)]
        pl = footage.plan(tl, 1150.0, (1946, 1962), fk, name="Life in a Brand-New 1950s Suburb")
        cs = [s for s in pl["shots"] if s["kind"] == "clip"]
        films.clear()
        films["good0"] = "Suburban homes"
        try:
            footage.plan(tl, 1150.0, (1946, 1962), fk / "b", name="x")
            deferred = False
        except footage.ShortOfFilm as e:      # under 15 minutes of unique film: build.py defers it
            deferred = e.fits_min < 15.2
        except footage.Deferred:
            deferred = True
    finally:
        _ar.search, _ar.files, _cl.shots, _llm.complete_json, footage.vision.check = saved_f
    check("footage: films not about the topic are not taken; every shot on screen matches and is unique enough; "
          "too little is deferred",
          not any(f.startswith("bad") for f in pl["films"]) and all((s.get("vision") or {}).get("ok") for s in cs)
          and all(abs(a["end"] - b["start"]) < 1e-6 for a, b in zip(pl["shots"], pl["shots"][1:])) and deferred,
          f"{len(cs)} shots, {pl.get('confirmed_min')} min confirmed")
    # owner 08.11: the CPU prefilter takes the clear matches, drops the clear misses, and Gemini sees
    # only the doubtful frames, 8 slots a request; a frame the CPU turned down is never on screen
    from pipeline import prefilter as _pf
    import zlib as _z
    saved_pf = (_pf.thresholds, _pf.available, _pf.scores)
    asked_px = []
    sc_of = lambda line, kf: (_z.crc32(f"{line}|{Path(kf).parent.name}/{Path(kf).name}".encode()) % 100) / 100
    try:
        films.clear()
        films.update({f"good{i}": "Suburban homes and families" for i in range(16)})
        _ar.search = lambda *a, **k: [{"id": i, "title": v, "year": 1952, "colour": True, "description": v,
                                       "subject": "", "licence": "PD", "page": ""} for i, v in films.items()]
        _ar.files = lambda fid: {"proxy": "x", "best": "b", "best_size": (640, 480)}
        _cl.shots = _shots
        _llm.complete_json = lambda prompt, **k: {"results": [
            {"i": i, "score": 3} for i, l in enumerate(l for l in prompt.split("FILMS:\n")[1].splitlines() if l.strip())]}
        footage.vision.check = lambda pairs: (asked_px.append(len(pairs)) or
                                              [{"ok": sc_of(_l, kf) > 0.45, "status": "checked", "note": "x"}
                                               for _l, _e, kf in pairs])
        _pf.thresholds, _pf.available = (lambda: (0.99, 0.2)), (lambda: True)
        _pf.scores = lambda pairs: [sc_of(_l, kf) for _l, _e, kf in pairs]
        pl2 = footage.plan(tl, 1150.0, (1946, 1962), fk / "pf", name="Life in a Brand-New 1950s Suburb")
    finally:
        _ar.search, _ar.files, _cl.shots, _llm.complete_json, footage.vision.check = saved_f
        _pf.thresholds, _pf.available, _pf.scores = saved_pf
    cs2 = [s for s in pl2["shots"] if s["kind"] == "clip" and not s.get("ran_on")]
    st = pl2["vision_stats"]
    check("prefilter: clear matches taken on the CPU, clear misses never shown, Gemini only for the doubtful, "
          "24 frames a request",
          st["local_accept"] > 0 and st["local_reject"] > 0 and all(n <= 24 for n in asked_px)
          and all((s.get("vision") or {}).get("ok") for s in cs2)
          and all(sc_of(s["line"], s["kf"]) > 0.2 for s in cs2 if s.get("kf") and not (s.get("vision") or {}).get("reused"))
          and len(asked_px) < len(tl) // 4,
          f"{st}, {len(asked_px)} requests")
    sys.path.insert(0, str(ROOT / "scripts"))
    import calibrate_prefilter as _cal
    _sc = [k / 100 for k in range(100)]
    _ok = [x > 0.5 or (0.3 < x <= 0.5 and k % 2 == 0) for k, x in enumerate(_sc)]
    _a, _r = _cal.choose(_sc, _ok)
    _rep = _cal.report(_sc, _ok, _a, _r)
    check("prefilter calibration: accept keeps 95% agreement with Gemini, reject loses at most 10% of the good",
          _a is not None and _r is not None and _r < _a and _rep["their_gemini_ok_share"] >= 0.95
          and _rep["good_frames_lost"] <= 0.10 * sum(_ok), f"accept {_a}, reject {_r}, {_rep}")
    # owner 10.11: voiced shots of 8-10 s over several short lines, a price tag starting its own shot
    _tl = [{"start": i * 3.2, "end": i * 3.2 + 3.0, "text": f"short line {i}",
            **({"chapter": "## Homes | Long Island | 1950"} if i == 0 else {}),
            **({"price": {"amount": 7990, "year": 1949}} if i == 40 else {})} for i in range(120)]
    _sl = [s for s in footage.slots(_tl, 384.0) if s["kind"] == "clip"]
    _d = sorted(s["end"] - s["start"] for s in _sl)
    check("footage: voiced shots of 8-10 s over several lines (at most 13 with the pauses), the price tag kept",
          _d[len(_d) // 2] >= 8.0 and _d[-1] <= 13.0 and sum(1 for s in _sl if s.get("price")) == 1
          and any("short line 1 " in s["line"] for s in _sl), f"median {_d[len(_d) // 2]:.1f}, max {_d[-1]:.1f}")
    # a slot without a frame for its words takes one for its chapter's subject; off-topic stays out
    asked_ch = []
    try:
        films.clear()
        films.update({f"good{i}": "Suburban homes and families" for i in range(16)})
        _ar.search = lambda *a, **k: [{"id": i, "title": v, "year": 1952, "colour": True, "description": v,
                                       "subject": "", "licence": "PD", "page": ""} for i, v in films.items()]
        _ar.files = lambda fid: {"proxy": "x", "best": "b", "best_size": (640, 480)}
        _cl.shots = _shots
        _llm.complete_json = lambda prompt, **k: {"results": [
            {"i": i, "score": 3} for i, l in enumerate(l for l in prompt.split("FILMS:\n")[1].splitlines() if l.strip())]}
        footage.vision.check = lambda pairs: (asked_ch.append(sum(1 for l, _e, _k in pairs if l.startswith("(what"))) or
                                              [{"ok": l.startswith("(what"),
                                                "status": "checked", "note": "x"} for l, _e, kf in pairs])
        pl3 = footage.plan(tl, 1150.0, (1946, 1962), fk / "ch", name="Life in a Brand-New 1950s Suburb")
    finally:
        _ar.search, _ar.files, _cl.shots, _llm.complete_json, footage.vision.check = saved_f
    cs3 = [s for s in pl3["shots"] if s["kind"] == "clip" and not s.get("ran_on")]
    check("footage: with no frame for the words, a frame for the chapter's subject (asked as such), never an "
          "unchecked one", sum(asked_ch) > 0 and all((s.get("vision") or {}).get("ok") for s in cs3)
          and all((s.get("vision") or {}).get("level") == "chapter" or (s.get("vision") or {}).get("reused")
                  for s in cs3), f"{len(cs3)} shots, {sum(asked_ch)} chapter-level frames asked")
    # owner 10.11: no approve - a viewer's gate two minutes at a time, a judge of the thumbnails
    from pipeline import viewer
    _vid = W / "viewer.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc=size=320x180:rate=5:duration=250", "-pix_fmt", "yuv420p", str(_vid)], check=True)
    _tlv = [{"start": i * 10.0, "end": i * 10.0 + 9.0, "text": f"line {i}"} for i in range(25)]
    saved_ask = viewer._ask
    seen_v = []
    try:
        viewer._ask = lambda prompt, images: (seen_v.append(prompt) or
                                              {"marks": [{"t": "0:10", "kind": "logo", "why": "x"}] * 2})
        v_bad = viewer.check(_vid, _tlv, W / "v1")
        viewer._ask = lambda prompt, images: {"marks": []}
        v_ok = viewer.check(_vid, _tlv, W / "v2")
        viewer._ask = lambda prompt, images: None
        try:
            viewer.check(_vid, _tlv, W / "v3")
            v_none = {"pass": True}
        except RuntimeError as e:
            v_none = {"pass": False, "stopped": "VIEWER_UNAVAILABLE" in str(e)}
        viewer._ask = lambda prompt, images: {"best": 1, "why": "x"}
        _th = []
        for i in range(3):
            Image.new("RGB", (64, 36), (i * 60, 0, 0)).save(W / f"jt{i}.jpg")
            _th.append(W / f"jt{i}.jpg")
        j = viewer.judge(_th, "THE $7,990 DREAM", "a new house")
    finally:
        viewer._ask = saved_ask
    check("viewer gate: more than 3 marks stops the upload, none passes, an unanswered minute stops the stage; "
          "the judge picks the thumbnail",
          not v_bad["pass"] and v_bad["sheets"] == 3 and v_ok["pass"] and v_none.get("stopped") and j == 1
          and "line 3" in seen_v[0], f"{v_bad['detail'][:60]} | {v_ok['detail'][:40]} | judge {j}")
    nt = subprocess.run([sys.executable, str(ROOT / "automation" / "next_topic.py")], capture_output=True, text=True)
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    check("next topic: never one without its Wikipedia articles",
          nt.stdout.strip() == "" or next(t for t in q["topics"] if t["slug"] == nt.stdout.strip()).get("wikipedia"),
          nt.stdout.strip())
    print("\nSMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
