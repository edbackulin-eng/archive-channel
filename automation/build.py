"""Build one episode, stage by stage (each stage one job of episode.yml).

    python automation/build.py <slug> script        # Groq -> Gemini, fact-checked (write_episode.py)
    python automation/build.py <slug> narrate       # film breaks placed, Kokoro on the CPU
    python automation/build.py <slug> plan          # the shot plan, every shot checked by Gemini
    python automation/build.py <slug> cut <k> <n>   # restore part k of n of the shots (matrix job)
    python automation/build.py <slug> render        # edit, thumbnails, QA, description, credits

Everything an episode needs lives in episodes/<slug>/: script.txt, meta.json and
fact_check.json in the repository; build/ and out/ only on the runner and in
artifacts.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import soundfile as sf  # noqa: E402

from core import narrate  # noqa: E402
from core.logger import get_logger  # noqa: E402
from core.voice import SAMPLE_RATE  # noqa: E402
from pipeline import archive, breathing, clips, edit, footage, prices, qa, thumbnail, vision  # noqa: E402

log = get_logger("build")
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))


def paths(slug):
    e = ROOT / "episodes" / slug
    return e, e / "build", e / "out"


def stage_script(slug):
    e, _, _ = paths(slug)
    if (e / "script.txt").exists():
        from pipeline import scriptwriter
        s = (e / "script.txt").read_text(encoding="utf-8")
        if scriptwriter.est_minutes(s) < scriptwriter.MIN_EST_MIN or scriptwriter.words(s) < scriptwriter.WORDS_MIN:
            # written before the full minimum was kept: written again from its sources
            log.info("%s: the script in the repository runs ~%.1f min; it is written again", slug,
                     scriptwriter.est_minutes(s))
            (e / "script.txt").unlink()
            return subprocess.run([sys.executable, str(ROOT / "automation" / "write_episode.py"), slug]).returncode
        if scriptwriter.words(s) > scriptwriter.WORDS_MAX or scriptwriter.est_minutes(s) > scriptwriter.MAX_EST_MIN:
            cut = scriptwriter.shorten(s) if scriptwriter.words(s) > scriptwriter.WORDS_MAX else s
            if scriptwriter.est_minutes(cut) > scriptwriter.MAX_EST_MIN:
                cut = scriptwriter.shorten_to(cut, scriptwriter.MAX_EST_MIN)
            (e / "script.txt").write_text(cut.strip() + "\n", encoding="utf-8")
            m = json.loads((e / "meta.json").read_text(encoding="utf-8"))
            m.update({"words": scriptwriter.words(cut), "minutes_at_150wpm": round(scriptwriter.words(cut) / 150, 1)})
            (e / "meta.json").write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")
        return 0
    prices.refresh()                     # a new year's CPI-U average, at most monthly; the file stays otherwise
    return subprocess.run([sys.executable, str(ROOT / "automation" / "write_episode.py"), slug]).returncode


def stage_narrate(slug):
    e, b, _ = paths(slug)
    b.mkdir(parents=True, exist_ok=True)
    script = (e / "script.txt").read_text(encoding="utf-8")
    placed = breathing.place(script).rstrip() + "\n"
    if placed != script:                 # a rerun changes nothing, so it commits nothing
        (e / "script.txt").write_text(placed, encoding="utf-8")
    script = placed
    voice = narrate.load_kokoro()
    audio, tl = narrate.synthesize(voice, script)
    secs = len(audio) / SAMPLE_RATE
    ep = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["episode"]
    if secs > ep["minutes_max"] * 60:
        # 09.11: a script that narrates past the ceiling is cut by whole lines to the time and narrated
        # once more, instead of failing the build (1950s-groceries: 20.7 min, three runs)
        from pipeline import scriptwriter
        goal = scriptwriter.est_minutes(script) * (ep["minutes_max"] * 60 - 30) / secs
        log.warning("%s narrates %.1f min; cut to ~%.1f estimated minutes and narrated again", slug, secs / 60, goal)
        script = breathing.place(scriptwriter.shorten_to(script, goal)).rstrip() + "\n"
        (e / "script.txt").write_text(script, encoding="utf-8")
        audio, tl = narrate.synthesize(voice, script)
        secs = len(audio) / SAMPLE_RATE
    if not ep["minutes_min"] * 60 <= secs <= ep["minutes_max"] * 60:
        # QA would fail on length after hours of cutting and rendering: stop here
        raise RuntimeError(f"LENGTH: the narration runs {secs / 60:.1f} min, the episode must be "
                           f"{ep['minutes_min']}-{ep['minutes_max']}")
    sf.write(b / "narration.wav", audio, SAMPLE_RATE)
    (b / "timeline.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


def stage_plan(slug):
    e, b, _ = paths(slug)
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    tl = json.loads((b / "timeline.json").read_text(encoding="utf-8"))
    total = sf.info(str(b / "narration.wav")).duration
    try:
        plan = footage.plan(tl, total, tuple(topic["years"]), b / "footage", query=topic.get("query"),
                            name=topic["name"], extra_queries=[topic["name"]] + list(topic.get("wikipedia") or [])
                            + list(topic.get("related") or []))
    except footage.ShortOfFilm as e:
        # owner 07.11: never the same pictures again - the episode is cut to the unique film there
        # is (not under 15 minutes), narrated and planned again; under 15 it is deferred
        from pipeline import scriptwriter
        if e.fits_min < 15.2:
            topic.update({"status": "deferred", "deferred_why": str(e)})
            (ROOT / "topics" / "queue.json").write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
            log.warning("%s: %s; under 15 minutes of unique film, the topic is deferred", slug, e)
            return 3
        s = (e_dir := paths(slug)[0] / "script.txt").read_text(encoding="utf-8")
        cut = scriptwriter.shorten_to(s, e.fits_min)
        e_dir.write_text(cut.strip() + "\n", encoding="utf-8")
        log.warning("%s: %s; the script is cut to ~%.1f min and narrated again", slug, e,
                    scriptwriter.est_minutes(cut))
        if stage_narrate(slug):
            return 1
        tl = json.loads((b / "timeline.json").read_text(encoding="utf-8"))
        total = sf.info(str(b / "narration.wav")).duration
        try:
            plan = footage.plan(tl, total, tuple(topic["years"]), b / "footage", query=topic.get("query"),
                                name=topic["name"], extra_queries=[topic["name"]] + list(topic.get("wikipedia") or [])
                                + list(topic.get("related") or []))
        except (footage.ShortOfFilm, footage.Deferred) as e2:
            topic.update({"status": "deferred", "deferred_why": str(e2)})
            (ROOT / "topics" / "queue.json").write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n",
                                                       encoding="utf-8")
            log.warning("%s: %s after the cut; the topic is deferred", slug, e2)
            return 3
    except footage.Deferred as e:
        # too little confirmed film (owner 07.11): the topic waits, the next one is built
        topic.update({"status": "deferred", "deferred_why": str(e)})
        (ROOT / "topics" / "queue.json").write_text(json.dumps(q, indent=2, ensure_ascii=False) + "\n",
                                                   encoding="utf-8")
        log.warning("%s: %s; the topic is deferred", slug, e)
        return 3
    for s in plan["shots"]:
        if s["kind"] == "clip" and s.get("item"):
            s["year"] = s["item"].get("year")
            s["place"] = "United States"
    (b / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


def stage_cut(slug, k, n):
    _, b, _ = paths(slug)
    plan = json.loads((b / "plan.json").read_text(encoding="utf-8"))
    todo = [i for i, s in enumerate(plan["shots"]) if s["kind"] == "clip" and s.get("film")]
    for i in todo[k::n]:
        s = plan["shots"][i]
        out = b / "clips" / f"clip_{i:04d}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            clips.cut(s["best"], s["src_start"], s["end"] - s["start"] + 0.2, out, b / f"cutwork_{k}")
    return 0


def _description(meta, plan, tl, music_title):
    chapters, seen = [], set()
    for line in tl:
        if line.get("chapter"):
            t = max(0, int(line["start"] - line.get("film_before", 0)))
            chapters.append(f"{t // 60}:{t % 60:02d} {line['chapter'].split('|')[0].strip()}")
    if not chapters or not chapters[0].startswith("0:00"):
        chapters.insert(0, "0:00 Intro")
    films = []
    for s in plan["shots"]:
        it = s.get("item") or {}
        if s.get("film") and s["film"] not in seen:
            seen.add(s["film"])
            films.append(f"- {archive.credit(it)}" if it.get("title") else f"- https://archive.org/details/{s['film']}")
    lib = json.loads((ROOT / "assets" / "music" / "library.json").read_text(encoding="utf-8"))
    return (f"{meta['title']}\n\nCHAPTERS\n" + "\n".join(chapters) +
            "\n\nSOURCES\n" + "\n".join(f"- {u}" for u in meta["sources"]) +
            "\n\nARCHIVE FILM (public domain, US Government or CC0; each item's licence checked)\n" + "\n".join(films) +
            f"\n\nMUSIC\n{lib['attribution'].format(title=music_title)}\n\n"
            "PRICES IN TODAY'S DOLLARS\nEvery \"today\" figure is the old price times the U.S. Bureau of Labor "
            f"Statistics Consumer Price Index (CPI-U, series CUUR0000SA0), {prices.today_year()} annual average over "
            "the year's own: https://data.bls.gov/timeseries/CUUR0000SA0\n\n"
            "HOW THIS IS MADE\nNarration is a synthetic voice. All footage is real archive film, restored "
            "(film frames recovered from the video transfer, denoised, stabilised, upscaled); captions give "
            "the year and place of the film itself.")


def stage_render(slug):
    e, b, out = paths(slug)
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((b / "plan.json").read_text(encoding="utf-8"))
    for i, s in enumerate(plan["shots"]):
        if s["kind"] == "clip":
            s["clip"] = str(b / "clips" / f"clip_{i:04d}.mp4")
    tl = json.loads((b / "timeline.json").read_text(encoding="utf-8"))
    meta = json.loads((e / "meta.json").read_text(encoding="utf-8"))
    lib = json.loads((ROOT / "assets" / "music" / "library.json").read_text(encoding="utf-8"))
    track = lib["tracks"][sum(map(ord, slug)) % len(lib["tracks"])]
    rep = edit.render(plan, b / "narration.wav", ROOT / "assets" / "music" / track["file"], out / "video.mp4", b / "edit")
    # thumbnails: the picture shows what the words name (owner 04.11: a house's price over a house,
    # a grocery price over a store); of the checked shots, the most vivid that Gemini finds showing
    # meta["thumbnail_subject"], in the manner of the top 30
    words, subject = meta["thumbnail_words"], meta.get("thumbnail_subject") or meta["thumbnail_words"]
    # one candidate per keyframe (07.11: the 12 best were 3 kitchens of one film, repeated by the
    # reused stretches, and none showed the house the words name), up to 36, 12 a request,
    # until three show the subject
    seen_kf, uniq = set(), []
    for s in sorted((s for s in plan["shots"] if s["kind"] == "clip" and s.get("kf") and (s.get("vision") or {}).get("ok")),
                    key=lambda s: -thumbnail.score(thumbnail.Image.open(s["kf"]))[0]):
        if s["kf"] not in seen_kf and (s.get("film"), round(s.get("src_start") or 0)) not in seen_kf:
            seen_kf.update({s["kf"], (s.get("film"), round(s.get("src_start") or 0))})
            uniq.append(s)
    asked = f"A thumbnail that says '{words}'. The picture must show: {subject}"
    # owner 08.11: SigLIP on the CPU ranks the frames by the subject first and drops the clear
    # misses; Gemini sees only the best of the rest, 24 in one request (was up to 36, 12 a request)
    step, upto = 12, 36
    from pipeline import prefilter
    th = prefilter.thresholds() if prefilter.thresholds() and prefilter.available() else None
    if th and uniq:
        sc = prefilter.scores([(subject, f"{s.get('year') or ''}", s["kf"]) for s in uniq])
        ranked_ = sorted(zip(uniq, sc), key=lambda x: -x[1])
        kept = [s for s, x in ranked_ if x > th[1]]
        log.info("thumbnail: %d of %d frames kept by the CPU check of '%s'", len(kept), len(uniq), subject)
        uniq, step, upto = kept or [s for s, _x in ranked_], 24, 48
    cands = []
    for k in range(0, min(upto, len(uniq)), step):
        pool = uniq[k:k + step]
        verdicts = vision.check([(asked, f"{s.get('year') or ''}", s["kf"]) for s in pool])
        cands += [s for s, v in zip(pool, verdicts) if v.get("ok")]
        if len(cands) >= 3:
            break
    cands = cands[:3]
    year = f"{meta.get('decade') or str(min(s.get('year') or 1950 for s in cands) // 10 * 10) + 's'}" if cands else "1950s"
    reports = []
    for k, s in enumerate(cands):
        base = thumbnail.hd_frame(s["best"], s["src_start"] + 1.0, b / f"thumb_base_{k}.png")
        r = thumbnail.compose_best(thumbnail.Image.open(base), year, words, out / f"thumb_{k}.jpg")
        r.update({"matches_words": True, "subject": subject, "film": s.get("film"), "t": s.get("src_start")})
        reports.append(r)
    good = [r for r in reports if r["corner_empty"] and r["readable_small"] and not r["text_over_face"]
            and r["sharp_ok"]]
    best = good[0] if good else (reports[0] if reports else {"matches_words": False, "subject": subject,
                                                               "note": "no shot shows the thumbnail's subject"})
    # owner 10.11: no approve - the judge picks the thumbnail of the ones that pass the checks
    from pipeline import viewer
    if len(good) >= 2:
        k = viewer.judge([out / r["file"] for r in good], words, subject)
        if k is not None:
            best = good[k]
            best["judged"] = True
    if best.get("file"):
        (out / "thumbnail.jpg").write_bytes((out / best["file"]).read_bytes())
    result = qa.check(plan, tl, out / "video.mp4", best)
    result["render"] = rep
    # and the last look, a viewer's - owner 12.11: a score, not a gate
    result["viewer"] = {**viewer.check(out / "video.mp4", tl, b), "blocking": False}
    scored = [v for v in result.values() if isinstance(v, dict) and "pass" in v and not v.get("blocking")]
    result["score"] = round(sum(bool(v["pass"]) for v in scored) / max(1, len(scored)), 2)
    (out / "qa.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "metadata.json").write_text(json.dumps({
        "title": meta["title"], "tags": ["1950s", "vintage america", "archive footage", "american history",
                                         "nostalgia", "life in the 1950s", "restored footage"],
        "description": _description(meta, plan, tl, track["title"])}, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s rendered; QA %s", slug, "passed" if result["passed"] else "FAILED")
    return 0


def main():
    slug, stage = sys.argv[1], sys.argv[2]
    if stage == "cut":
        return stage_cut(slug, int(sys.argv[3]), int(sys.argv[4]))
    try:
        return {"script": stage_script, "narrate": stage_narrate, "plan": stage_plan, "render": stage_render}[stage](slug)
    except Exception as e:                                           # noqa: BLE001
        # the Gemini day is spent: no build starts until it turns (automation/quota_hold.py)
        from automation import quota_hold
        if quota_hold.spent(e):
            h = quota_hold.set_(slug)
            log.warning("GEMINI_DAY_SPENT: %s stopped in its %s stage; builds wait until %s", slug, stage, h["until"])
        raise


if __name__ == "__main__":
    sys.exit(main())
