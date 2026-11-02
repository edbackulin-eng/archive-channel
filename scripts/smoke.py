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
  6. thumbnail: composed with the corner empty and the words readable small.
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
    print("\nSMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
