"""Thumbnails in the manner of the niche's top 30 (analysis/conclusion.md).

What the most viewed thumbnails do: bright, saturated colour; a person or a
living scene of the era filling the frame; the year or decade in big numbers;
a hook of two to four words; nothing in the bottom-right 25% x 20%, where
YouTube prints the running time.

Bases are real frames of the episode's own public-domain films. Frames are
mined from each film's proxy (every 12 s) and scored: colourfulness, sharpness,
brightness, and faces (OpenCV's frontal-face detector) - a person reads at
thumbnail size, an empty room does not. The best frames are pulled again from
the best-quality file, cleaned (non-local-means denoise), enlarged to 1280x720
and made vivid. The words go where no face is, and never into the reserved
corner; every line must read at 336x189.
"""
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


def hd_frame(url, t, out):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", url,
                    "-frames:v", "1", "-vf", "scale=iw*sar:ih,setsar=1", str(out)], timeout=600, check=True)
    im = cv2.imread(str(out))
    im = cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 7, 21)
    cv2.imwrite(str(out), im)
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


def vivid(im):
    im = ImageEnhance.Color(im).enhance(1.45)
    im = ImageEnhance.Contrast(im).enhance(1.15)
    im = ImageEnhance.Brightness(im).enhance(1.06)
    return im.filter(ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))


def _outlined(d, xy, text, f, fill, stroke):
    d.text(xy, text, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))


def compose(base, year, hook, out, side="left"):
    """base: a PIL image. year: '1950s'. hook: 2-4 words. Writes out; returns a report."""
    im = vivid(frame_subject(base.convert("RGB"), side))
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
    for line in lines:
        x = col_x0
        for w in line.split():
            _outlined(d, (x, y), w, hf, RED if w == red else WHITE, max(5, hsize // 14))
            x += hf.getlength(w + " ")
        y += int(hsize * 1.05)
    text_box = (col_x0, 26, col_x0 + col_w, y)
    corner = (int(W * (1 - CORNER[0])), int(H * (1 - CORNER[1])), W, H)
    overlap_corner = not (text_box[2] <= corner[0] or text_box[3] <= corner[1])
    covers_face = any(not (x + w < text_box[0] or x > text_box[2] or yy + h < text_box[1] or yy > text_box[3])
                      for x, yy, w, h in fs)
    im.save(out, quality=92)
    small = im.resize((336, 189), Image.LANCZOS)
    cap_px = hsize * 0.72 * 189 / H
    report = {"file": Path(out).name, "faces": len(fs), "text_side": side, "year_px": ysize, "hook_px": hsize,
              "hook_cap_at_336": round(cap_px, 1), "readable_small": cap_px >= 10,
              "corner_empty": not overlap_corner, "text_over_face": covers_face}
    small.save(Path(out).with_name(Path(out).stem + "_336.jpg"), quality=90)
    return report
