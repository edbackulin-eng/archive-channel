"""Shots out of archive films, restored on the CPU.

1. shots(): the film's small proxy file is scanned for cuts (ffmpeg scdet) and
   split into candidate shots; each gets a keyframe from its middle for the
   vision check. Black, blank and title-card stretches are left out.
2. cut(): the chosen 4-6 s are read straight from the best-quality file over
   HTTP (archive.org serves byte ranges, so the whole film is never
   downloaded) and restored:
     deflicker    old film breathes in brightness from frame to frame
     hqdn3d       grain and dust, gently - archive film should still look like film
     vidstab      two passes, smoothing gate weave and handheld shake
     scale        up to 1920x1080 with Lanczos, filling 16:9 (a 4:3 frame loses a
                  sliver top and bottom, never letterboxed or pillarboxed)
     unsharp      a light sharpen after the upscale
     fps          30
   The original sound is kept at its own level, to sit quietly under the voice.
"""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from core.logger import get_logger

log = get_logger("clips")
W, H, FPS = 1920, 1080, 30
SCENE_TH = 0.28
MIN_SHOT_S = 2.5


def _ff(args, timeout=900):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return r.stderr


def duration(src):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(src)], capture_output=True, text=True, timeout=120)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _blank(path):
    a = np.asarray(Image.open(path).convert("L").resize((160, 120)), np.float32)
    return a.std() < 10 or a.mean() < 18 or a.mean() > 238


def shots(proxy, work, max_shots=80):
    """[(start, end, keyframe_path)] of usable shots in a film."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    total = duration(proxy)
    err = _ff(["-i", str(proxy), "-vf", f"scale=320:-2,scdet=threshold={SCENE_TH * 100:.0f}",
               "-an", "-f", "null", "-"], timeout=1800)
    cuts = sorted({0.0, *[float(x) for x in re.findall(r"lavfi\.scd\.time:\s*([\d.]+)", err)], total})
    spans = [(a, b) for a, b in zip(cuts, cuts[1:]) if b - a >= MIN_SHOT_S]
    # very long takes are split, so a 40 s pan gives several choices
    out = []
    for a, b in spans:
        t = a
        while b - t >= MIN_SHOT_S and len(out) < max_shots * 3:
            e = min(b, t + 8.0)
            out.append((round(t, 2), round(e, 2)))
            t = e
    keep = []
    step = max(1, len(out) // max_shots)
    for a, b in out[::step][:max_shots]:
        kf = work / f"kf_{a:08.2f}.jpg"
        try:
            _ff(["-loglevel", "error", "-y", "-ss", f"{(a + b) / 2:.2f}", "-i", str(proxy), "-frames:v", "1",
                 "-vf", "scale=768:-2", str(kf)], timeout=120)
        except RuntimeError:
            continue
        if kf.exists() and not _blank(kf):
            keep.append((a, b, kf))
    log.info("%s: %d cuts, %d usable shots", Path(str(proxy)).name, len(cuts), len(keep))
    return keep


def cut(source_url, start, seconds, out, work, sound=True):
    """A restored 1920x1080/30 clip of `seconds` from `start` in the source."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    # Read over HTTP, a cut now and then came back truncated (a broken container
    # with exit code 0); it is checked and read again.
    raw = work / (Path(out).stem + "_raw.mp4")
    for attempt in range(3):
        _ff(["-loglevel", "error", "-y", "-ss", f"{start:.3f}", "-i", source_url, "-t", f"{seconds + 0.5:.3f}",
             "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", str(raw)], timeout=900)
        if duration(raw) >= seconds * 0.9:
            break
        if attempt == 2:
            raise RuntimeError(f"cut of {source_url} at {start:.1f}s came back short three times")
    # 1. border, flicker and grain, on the transfer's own geometry
    crop = _cropdetect(raw)
    pre = work / (Path(out).stem + "_pre.mp4")
    chain = ([crop] if crop else ["scale=iw*sar:ih,setsar=1"]) + ["deflicker=size=5:mode=pm", "hqdn3d=2:1.5:4:3"]
    _ff(["-loglevel", "error", "-y", "-i", str(raw), "-vf", ",".join(chain), "-map", "0:v:0", "-map", "0:a?",
         "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8", "-c:a", "copy", str(pre)], timeout=900)
    # 2. stabilise that same picture (detect and transform must see the same frames);
    #    optzoom keeps the moved frame filled, so no black edge creeps in
    trf = work / (Path(out).stem + ".trf")
    final = []
    try:
        _ff(["-loglevel", "error", "-y", "-i", str(pre), "-vf",
             f"vidstabdetect=shakiness=5:accuracy=9:result={trf.as_posix()}", "-f", "null", "-"])
        if trf.exists():
            final.append(f"vidstabtransform=input={trf.as_posix()}:smoothing=12:optzoom=1:interpol=bicubic")
    except RuntimeError:
        pass
    # 3. up to 1080p, filling 16:9
    final += [f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos",
              f"crop={W}:{H}", "unsharp=5:5:0.6:5:5:0.0", f"fps={FPS}", "format=yuv420p"]
    audio = (["-map", "0:a?", "-c:a", "aac", "-b:a", "160k"] if sound else ["-an"])
    _ff(["-loglevel", "error", "-y", "-i", str(pre), "-vf", ",".join(final), "-map", "0:v:0", *audio,
         "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-r", str(FPS),
         str(out)], timeout=1800)
    pre.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    trf.unlink(missing_ok=True)
    return out


def _cropdetect(src):
    """The crop that removes the black border inside a transfer, or None.

    ffmpeg's cropdetect follows the bright content, not the border, on pale
    scenes; this looks for the border itself: edge rows and columns that stay
    near-black and flat across several frames (after squaring the pixels)."""
    tmp = Path(str(src) + "_b")
    tmp.mkdir(exist_ok=True)
    _ff(["-loglevel", "error", "-y", "-i", str(src), "-vf",
         "scale=iw*sar:ih,setsar=1,select='not(mod(n\,12))',scale=640:-2,format=gray",
         "-frames:v", "6", "-vsync", "vfr", str(tmp / "f%02d.png")], timeout=300)
    frames = [np.asarray(Image.open(f), np.float32) for f in sorted(tmp.glob("f*.png"))]
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()
    # a border is dark in every frame; a fade or a night scene is not a border,
    # so frames that are dark all over are left out and the narrowest run wins
    frames = [f for f in frames if f.mean() >= 35]
    if not frames:
        return None
    h, w = frames[0].shape

    def run(lines, cap):
        n = 0
        for line in lines[:cap]:
            if line.mean() < 30 and line.std() < 14:
                n += 1
            else:
                break
        return n
    top = min(run(f, h // 3) for f in frames)
    bottom = min(run(f[::-1], h // 3) for f in frames)
    left = min(run(f.T, w // 4) for f in frames)
    right = min(run(f.T[::-1], w // 4) for f in frames)
    if not any(v >= 2 for v in (top, bottom, left, right)):
        return None
    # in the squared-pixel picture; the chain squares the pixels before cropping
    return (f"scale=iw*sar:ih,setsar=1,crop=iw*{(w - left - right) / w:.4f}:ih*{(h - top - bottom) / h:.4f}:"
            f"iw*{left / w:.4f}:ih*{top / h:.4f}")


def has_sound(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return bool(r.stdout.strip())


def probe_info(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
                        str(path)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")
