"""Drawn cards: the chapter map, and the year-and-place caption on archive shots.

chapter_card(): every chapter opens on a map of the United States (Natural
Earth 1:50m states, public domain) with the chapter's state lit and a dot on
its place, the chapter title above and "place - years" below. It runs under the
film break that opens the chapter, with a slow push-in.

caption(): a small lower-left label, "1957 - United States", taken from the
film's own metadata: the year the footage was shot and where, never the year
the narration talks about - a 1957 film shown over a sentence about 1947 says
1957.
"""
import json
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core import fonts

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "geo"
STATES_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/geojson/"
              "ne_50m_admin_1_states_provinces.geojson")
W, H, FPS = 1920, 1080, 30
INK, PAPER, LIT, DOT = (40, 36, 30), (238, 228, 205), (205, 92, 60), (176, 28, 28)

# places the chapters name -> (lat, lon, state)
PLACES = {
    "long island": (40.73, -73.52, "New York"), "levittown, new york": (40.73, -73.52, "New York"),
    "new york": (40.71, -74.0, "New York"), "levittown, pennsylvania": (40.16, -74.83, "Pennsylvania"),
    "pennsylvania": (40.16, -74.83, "Pennsylvania"), "detroit": (42.33, -83.05, "Michigan"),
    "chicago": (41.88, -87.63, "Illinois"), "los angeles": (34.05, -118.24, "California"),
    "norfolk, virginia": (36.85, -76.29, "Virginia"), "washington": (38.9, -77.04, "District of Columbia"),
}


def _states():
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / "states50.geojson"
    if not p.exists():
        req = urllib.request.Request(STATES_URL, headers={"User-Agent": "archive-channel/1.0"})
        p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    feats = json.loads(p.read_text(encoding="utf-8"))["features"]
    return [f for f in feats if f["properties"].get("adm0_a3") == "USA"
            and f["properties"].get("name") not in ("Alaska", "Hawaii")]


def _project(lon, lat, box):
    # a simple equirectangular view of the lower 48, scaled by cos(lat) so it does not look stretched
    x0, y0, x1, y1 = box
    lon0, lon1, lat0, lat1 = -125.0, -66.5, 24.0, 49.6
    k = np.cos(np.radians(37.5))
    w = (lon1 - lon0) * k
    h = lat1 - lat0
    s = min((x1 - x0) / w, (y1 - y0) / h)
    ox = x0 + ((x1 - x0) - w * s) / 2
    oy = y0 + ((y1 - y0) - h * s) / 2
    return ox + (lon - lon0) * k * s, oy + (lat1 - lat) * s


def _place(text):
    t = text.lower()
    for k in sorted(PLACES, key=len, reverse=True):
        if k in t:
            return PLACES[k]
    return None


def map_frame(chapter):
    """One 1920x1080 frame: title, map with the place lit, place and years."""
    title, place, years = ([p.strip() for p in chapter.split("|")] + ["", ""])[:3]
    im = Image.new("RGB", (W, H), PAPER)
    # paper grain, so it reads as an old printed map
    rng = np.random.default_rng(len(chapter))
    grain = (rng.normal(0, 5, (H, W, 1)) + np.array(PAPER, np.float32)).clip(0, 255).astype(np.uint8)
    im = Image.fromarray(np.repeat(grain, 1, axis=2) if grain.shape[2] == 3 else np.concatenate([grain] * 3, 2))
    d = ImageDraw.Draw(im)
    box = (260, 230, W - 260, H - 170)
    hit = _place(place)
    for f in _states():
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        lit = hit and f["properties"].get("name") == hit[2]
        for poly in polys:
            pts = [_project(lon, lat, box) for lon, lat in poly[0]]
            if len(pts) > 2:
                d.polygon(pts, fill=LIT if lit else (226, 214, 186), outline=INK)
    if hit:
        x, y = _project(hit[1], hit[0], box)
        for r, a in ((34, 70), (22, 140)):
            ring = Image.new("L", (W, H), 0)
            ImageDraw.Draw(ring).ellipse((x - r, y - r, x + r, y + r), outline=a, width=4)
            im.paste(DOT, (0, 0), ring)
        d.ellipse((x - 11, y - 11, x + 11, y + 11), fill=DOT, outline=(255, 255, 255), width=3)
    ts, _ = fonts.fit(title.upper(), "title", W - 300, 140, 120)
    tf = fonts.font("title", ts)
    tw = tf.getlength(title.upper())
    d.text(((W - tw) / 2, 60), title.upper(), font=tf, fill=INK)
    sub = " - ".join(x for x in (place, years) if x)
    sf = fonts.font("label", 64)
    sw = sf.getlength(sub)
    d.text(((W - sw) / 2, H - 140), sub, font=sf, fill=(110, 60, 40))
    return im.filter(ImageFilter.SMOOTH)


def chapter_card(chapter, seconds, out, work):
    """The map frame as a clip with a slow push-in."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    still = work / (Path(out).stem + ".png")
    map_frame(chapter).save(still)
    d = max(seconds, 0.5)
    e = f"(st(0,clip(t/{d:.3f},0,1));ld(0)*ld(0)*(3-2*ld(0)))"
    vf = (f"scale=w='trunc(3840*(1+0.05*{e})/2)*2':h='trunc(2160*(1+0.05*{e})/2)*2':eval=frame,"
          f"crop=3840:2160,scale={W}:{H}:flags=lanczos,fps={FPS},format=yuv420p")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS),
                    "-t", f"{d:.3f}", "-i", str(still), "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", "17", str(out)], check=True)
    return out


def caption_png(year, place, out):
    """Transparent 1920x1080 overlay with '1957 - United States' lower left."""
    text = " - ".join(str(x) for x in (year, place) if x)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = fonts.font("label", 54)
    tw = f.getlength(text)
    x, y = 64, H - 112
    pad = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(pad).rounded_rectangle((x - 18, y - 8, x + tw + 18, y + 60), 10, fill=(0, 0, 0, 120))
    im = Image.alpha_composite(im, pad.filter(ImageFilter.GaussianBlur(2)))
    d = ImageDraw.Draw(im)
    d.text((x, y), text, font=f, fill=(250, 240, 220))
    im.save(out)
    return out
