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
        s1 = [r["state"] for r in json.loads(st.read_text())]
        release.approve("a")
        release.approve("b")
        s2 = [(r["state"], r.get("publish_at")) for r in json.loads(st.read_text())]
    finally:
        release.ROOT, release.rows, release.save, youtube.upload, youtube.set_thumbnail, youtube.schedule = saved
    check("release: first two await approval, the third its turn, then all in order",
          s1 == ["awaiting_approval", "awaiting_approval", "awaiting_turn"]
          and all(x[0] == "scheduled" for x in s2) and s2[0][1] < s2[1][1] < s2[2][1], f"{s1} {s2}")
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
    from pipeline import scriptwriter, vision
    check("vision: a spent day is QuotaSpent, not unchecked", issubclass(vision.QuotaSpent, RuntimeError)
          and vision.QUOTA_MARK == "VISION_QUOTA_SPENT")
    long_s = "\n".join(["[price 7990 1949 | A new house]"] + ["word " * 20] * 125)          # 2500 words
    short_s = "\n".join(["[price 7990 1949 | A new house]"] + ["word " * 20] * 105)         # 2100 words
    saved_llm = scriptwriter.llm.complete_json
    try:
        scriptwriter.llm.complete_json = lambda *a, **k: {"script": short_s}
        cut = scriptwriter.shorten(long_s)
        scriptwriter.llm.complete_json = lambda *a, **k: {"script": long_s}
        try:
            scriptwriter.shorten(long_s)
            refused = False
        except RuntimeError as e:
            refused = "LENGTH" in str(e)
    finally:
        scriptwriter.llm.complete_json = saved_llm
    check("script: over WORDS_MAX is cut to fit, or stops with LENGTH",
          scriptwriter.words(cut) <= scriptwriter.WORDS_MAX and refused, str(scriptwriter.words(cut)))
    nt = subprocess.run([sys.executable, str(ROOT / "automation" / "next_topic.py")], capture_output=True, text=True)
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    check("next topic: never one without its Wikipedia articles",
          nt.stdout.strip() == "" or next(t for t in q["topics"] if t["slug"] == nt.stdout.strip()).get("wikipedia"),
          nt.stdout.strip())
    print("\nSMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
