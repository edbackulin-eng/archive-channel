"""Thumbnails in the manner of the niche's top 30 (analysis/conclusion.md).

What the most viewed thumbnails do: bright, saturated colour; a person or a
living scene of the era filling the frame; the year or decade in big numbers;
a hook of two to four words; nothing in the bottom-right 25% x 20%, where
YouTube prints the running time.

Bases are real frames of the episode's own public-domain films. Frames are
mined from each film's proxy (every 12 s) and scored: colourfulness, sharpness,
brightness, and faces (YuNet) - a person reads at thumbnail size, an empty room
does not. The chosen moment is then pulled again, sharp (owner 03.11: the
first thumbnails were soft):
  1. from the film's highest real resolution (pipeline/archive.files), never
     a 512Kb derivative;
  2. of every frame within 1.5 s of the moment, the sharpest: the highest
     variance of the Laplacian (a frame in motion blur or soft focus loses);
  3. black borders trimmed, then Real-ESRGAN x4 on the CPU (pipeline/upscale.py,
     this frame only);
  4. colour, contrast and brightness moved into the range of the niche's top 30
     (TOP30, measured on analysis/thumbs_top30.jpg at 336x189), then sharpened.
The words go where no face is, and never into the reserved corner; every line
must read at 336x189, and the QA gate turns down a picture whose photo half is
softer than SHARP_MIN at 336x189 or SHARP_MIN_FULL at 1280x720.
"""
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from core import fonts
from core.logger import get_logger

log = get_logger("thumbnail")
SIZE = (1280, 720)
CORNER = (0.25, 0.20)
YELLOW, WHITE, RED = (255, 212, 0), (255, 255, 255), (230, 32, 32)
# the top 30 thumbnails of the niche at 336x189 (analysis/thumbs_top30.jpg, 2026-11-03):
# median and upper quartile of HSV saturation, grey-level std (contrast) and mean (brightness)
TOP30 = {"sat": (59, 77), "contrast": (63, 71), "bright": (109, 123)}
# the photo half (no words): variance of the Laplacian after a light denoise, so film grain
# and blocking do not pass for detail. Calibrated 2026-11-03 on episode 1: the first, soft
# thumbnails measured 1187-1368 at 336x189 and 16-29 at 1280x720 (one combed frame 1201);
# through the sharp chain the same moments measure 3562-8434 and 542-669.
SHARP_MIN, SHARP_MIN_FULL = 2000, 200
# OpenCV 5 has no Haar cascades; YuNet (opencv_zoo, Apache-2.0) is its face detector
YUNET = Path(__file__).resolve().parent.parent / "assets" / "models" / "face_detection_yunet_2023mar.onnx"


def _frames(url, every_s, work, total):
    work.mkdir(parents=True, exist_ok=True)
    pat = work / "m_%04d.jpg"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", url, "-vf",
                    f"fps=1/{every_s},scale=iw*sar:ih,setsar=1,scale=480:-2", "-q:v", "3", str(pat)],
                   timeout=1800, check=False)
    return sorted(work.glob("m_*.jpg"))


def faces(im):
    """[(x, y, w, h)] of faces, YuNet, confidence >= 0.8."""
    bgr = cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    det = cv2.FaceDetectorYN.create(str(YUNET), "", (w, h), 0.8, 0.3, 50)
    _, found = det.detect(bgr)
    return [tuple(int(v) for v in f[:4]) for f in (found if found is not None else [])]


def score(im):
    a = np.asarray(im.convert("RGB").resize((320, 240)), np.float32)
    rg, yb = a[..., 0] - a[..., 1], 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    colour = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    g = cv2.cvtColor(a.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright = float(g.mean())
    fs = faces(im)
    face_area = sum(w * h for _, _, w, h in fs) / (im.width * im.height)
    if bright < 60 or bright > 215 or g.std() < 30:
        return -1, {}
    s = colour * 1.0 + min(sharp, 600) / 20 + (25 if fs else 0) + min(face_area * 400, 30)
    return s, {"colour": round(colour, 1), "sharp": round(sharp), "bright": round(bright), "faces": len(fs)}


def mine(film, work, every_s=12, keep=6):
    """The best thumbnail frames of one film: [(score, seconds, metrics)]."""
    frames = _frames(film["proxy"], every_s, Path(work) / film["id"], None)
    rows = []
    for k, f in enumerate(frames):
        s, m = score(Image.open(f))
        if s > 0:
            rows.append((round(s, 1), k * every_s + every_s / 2, m, f))
    rows.sort(key=lambda r: -r[0])
    return rows[:keep]


def laplacian_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detail(im, size=None, box=None):
    """Sharpness that grain and blocking cannot fake: the variance of the Laplacian,
    after a light non-local-means pass, of the box (fractions) of im at size."""
    a = np.asarray(im.convert("RGB").resize(size, Image.LANCZOS) if size else im.convert("RGB"))
    if box:
        h, w = a.shape[:2]
        a = a[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
    g = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)
    g = cv2.fastNlMeansDenoising(g, None, 7, 5, 15)
    return laplacian_var(g)


def trim_borders(im, dark=24):
    """Cut the black (or nearly black) frame some transfers carry."""
    g = np.asarray(im.convert("L"), np.float32)
    rows = np.where(np.percentile(g, 90, axis=1) > dark)[0]
    cols = np.where(np.percentile(g, 90, axis=0) > dark)[0]
    if len(rows) < g.shape[0] * 0.5 or len(cols) < g.shape[1] * 0.5:
        return im
    m = 2                                                        # the soft edge of the border
    return im.crop((cols[0] + m, rows[0] + m, cols[-1] + 1 - m, rows[-1] + 1 - m))


def sharpest_frame(url, t, work, window=1.5):
    """Of every frame within `window` s of t, the sharpest: (PIL image, its second, all scores)."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("s_*.png"):
        old.unlink()
    from pipeline import clips
    a = max(0.0, t - window)
    fields = clips.field_filter(url)                             # whole film frames, no combing
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{a:.2f}", "-i", url,
                    "-t", f"{2 * window:.2f}", "-vf", ",".join(x for x in (fields, "scale=iw*sar:ih,setsar=1") if x),
                    str(work / "s_%03d.png")],
                   timeout=900, check=True)
    frames = sorted(work.glob("s_*.png"))
    if not frames:
        raise RuntimeError(f"no frames near {t:.1f} s of {url}")
    scored = []
    for f in frames:
        g = np.asarray(trim_borders(Image.open(f)).convert("L"))
        h, w = g.shape
        scored.append((laplacian_var(g[h // 10:h - h // 10, w // 10:w - w // 10]), f))
    _, best = max(scored, key=lambda r: r[0])
    sec = a + frames.index(best) * (2 * window) / len(frames)
    return Image.open(best).convert("RGB"), sec, [round(v) for v, _ in scored]


def hd_frame(url, t, out, upscale_x4=True):
    """The thumbnail base: the sharpest frame near t of the best file, borders trimmed,
    Real-ESRGAN x4 (CPU). Writes out (PNG) and a report beside it; returns out."""
    out = Path(out)
    im, sec, scores = sharpest_frame(url, t, out.parent / (out.stem + "_frames"))
    im = trim_borders(im)
    src = im.size
    if upscale_x4:
        from pipeline import upscale
        im = upscale.x4(im, denoise=0.5)
    im.save(out)
    rep = {"source": url, "asked_s": round(t, 2), "frame_s": round(sec, 2), "frame_size": src,
           "frames_scored": len(scores), "laplacian_best": max(scores), "laplacian_median": int(np.median(scores)),
           "upscaled": im.size}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return out


def cover(im, size=SIZE):
    s = max(size[0] / im.width, size[1] / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    x, y = (im.width - size[0]) // 2, (im.height - size[1]) // 2
    return im.crop((x, y, x + size[0], y + size[1]))


def frame_subject(base, side, size=SIZE, max_zoom=1.2):
    """Cover 16:9 with the people in the third away from the words.

    A 4:3 frame filled to 16:9 can only lose its top and bottom, so to move a
    centred face out of the text column the picture is zoomed a little (at most
    max_zoom) and shifted until the faces sit around the 68 % line (or 32 % when
    the words go right)."""
    fs = faces(base)
    s0 = max(size[0] / base.width, size[1] / base.height)
    if not fs or side not in ("left", "right"):
        return cover(base, size)
    x0 = min(x for x, _, _, _ in fs)
    x1 = max(x + w for x, _, w, _ in fs)
    y0 = min(y for _, y, _, _ in fs)
    y1 = max(y + h for _, y, _, h in fs)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    want = 0.68 if side == "left" else 0.32
    best = None
    for z in [1.0 + 0.05 * k for k in range(int((max_zoom - 1) / 0.05) + 1)]:
        s = s0 * z
        bw, bh = base.width * s, base.height * s
        left = cx * s - want * size[0]
        left = min(max(0, left), bw - size[0])
        top = min(max(0, cy * s - 0.46 * size[1]), bh - size[1])
        err = abs((cx * s - left) / size[0] - want)
        if best is None or err < best[0] - 0.01:
            best = (err, s, left, top)
        if err < 0.04:
            break
    _, s, left, top = best
    im = base.resize((round(base.width * s), round(base.height * s)), Image.LANCZOS)
    return im.crop((round(left), round(top), round(left) + size[0], round(top) + size[1]))


def stats(im):
    a = np.asarray(im.convert("RGB").resize((336, 189), Image.LANCZOS))
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return {"sat": float(cv2.cvtColor(a, cv2.COLOR_RGB2HSV)[..., 1].mean()), "contrast": float(g.std()),
            "bright": float(g.mean())}


def style(im):
    """Colour, contrast and brightness moved into the top 30's range (TOP30, from the
    median to the upper quartile; a value already inside is left), then sharpened."""
    out = im
    for _ in range(4):
        s = stats(out)
        moved = False
        for key, enh in (("sat", ImageEnhance.Color), ("contrast", ImageEnhance.Contrast),
                         ("bright", ImageEnhance.Brightness)):
            lo, hi = TOP30[key]
            if lo <= s[key] <= hi:
                continue
            k = max(0.6, min(1.6, (lo + hi) / 2 / max(1.0, s[key])))
            out, moved = enh(out).enhance(k), True
            s = stats(out)
        if not moved:
            break
    return out.filter(ImageFilter.UnsharpMask(radius=1.6, percent=110, threshold=3))


vivid = style


def _outlined(d, xy, text, f, fill, stroke):
    d.text(xy, text, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def compose(base, year, hook, out, side="left"):
    """base: a PIL image. year: '1950s'. hook: 2-4 words. Writes out; returns a report."""
    photo = style(frame_subject(base.convert("RGB"), side))
    im = photo.copy()
    W, H = im.size
    fs = faces(im)
    # the text column is the half with less face in it, unless told
    if side == "auto":
        left = sum(w * h for x, _, w, h in fs if x + w / 2 < W / 2)
        right = sum(w * h for x, _, w, h in fs if x + w / 2 >= W / 2)
        side = "left" if left <= right else "right"
    col_x0, col_w = (36, int(W * 0.46)) if side == "left" else (int(W * 0.52), int(W * 0.46))
    shade = Image.new("L", (W, H), 0)
    sd = ImageDraw.Draw(shade)
    sx0, sx1 = (0, col_x0 + col_w + 60) if side == "left" else (col_x0 - 60, W)
    sd.rectangle((sx0, 0, sx1, int(H * 0.78)), fill=150)
    shade = shade.filter(ImageFilter.GaussianBlur(70))
    im = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), im, shade.point(lambda v: v * 0.55))
    d = ImageDraw.Draw(im)
    # the year, huge
    ysize, _ = fonts.fit(year, "title", col_w, H * 0.34, 260, spacing=1.0)
    yf = fonts.font("title", ysize)
    _outlined(d, (col_x0, 26), year, yf, YELLOW, max(6, ysize // 16))
    yb = d.textbbox((col_x0, 26), year, font=yf)
    # the hook, two to four words, one in red
    words = hook.upper().split()
    hsize, lines = fonts.fit(" ".join(words), "title", col_w, H * 0.42, 130, spacing=1.05)
    hf = fonts.font("title", hsize)
    y = yb[3] + 18
    red = words[-1]
    right = yb[2]
    for line in lines:
        x = col_x0
        for w in line.split():
            _outlined(d, (x, y), w, hf, RED if w == red else WHITE, max(5, hsize // 14))
            x += hf.getlength(w + " ")
        right = max(right, x - hf.getlength(" "))
        y += int(hsize * 1.05)
    # the words as drawn, not the whole column
    text_box = (col_x0, 26, right, y)
    corner = (int(W * (1 - CORNER[0])), int(H * (1 - CORNER[1])), W, H)
    overlap_corner = not (text_box[2] <= corner[0] or text_box[3] <= corner[1])
    # a face the viewer reads: 7 % of the height or more (the crowd behind the shade is not one)
    covers_face = any(not (x + w < text_box[0] or x > text_box[2] or yy + h < text_box[1] or yy > text_box[3])
                      for x, yy, w, h in fs if h >= 0.07 * H)
    im.save(out, quality=92)
    small = im.resize((336, 189), Image.LANCZOS)
    cap_px = hsize * 0.72 * 189 / H
    half = (0.5, 0.0, 1.0, 1.0) if side == "left" else (0.0, 0.0, 0.5, 1.0)
    sharp_336, sharp_full = detail(photo, (336, 189), half), detail(photo, None, half)
    report = {"file": Path(out).name, "faces": len(fs), "text_side": side, "year_px": ysize, "hook_px": hsize,
              "hook_cap_at_336": round(cap_px, 1), "readable_small": cap_px >= 10,
              "corner_empty": not overlap_corner, "text_over_face": covers_face,
              "sharp_336": round(sharp_336), "sharp_1280": round(sharp_full),
              "sharp_ok": sharp_336 >= SHARP_MIN and sharp_full >= SHARP_MIN_FULL,
              "style": {k: round(v) for k, v in stats(im).items()}, "top30": TOP30}
    small.save(Path(out).with_name(Path(out).stem + "_336.jpg"), quality=90)
    return report


def compose_best(base, year, hook, out, sides=("left", "right")):
    """compose() on the first side whose words cover no face (left first, the top 30's habit)."""
    tried = []
    for side in sides:
        rep = compose(base, year, hook, out, side)
        if rep["corner_empty"] and not rep["text_over_face"]:
            return rep
        tried.append((side, rep))
    # none is clean: the one with the fewest faults, written again
    side, rep = min(tried, key=lambda r: (not r[1]["corner_empty"]) + r[1]["text_over_face"])
    return compose(base, year, hook, out, side) if side != sides[-1] else rep
