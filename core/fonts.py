"""Fonts for everything the code draws, with glyph coverage checked.

Anton and Bebas Neue (SIL Open Font License) ship in assets/fonts for titles
and thumbnails. Interface text uses a system sans and mono: Segoe UI and
Consolas on Windows, DejaVu on the Ubuntu runner, both covering the full Latin
range plus punctuation. A character a font cannot draw is reported rather than
rendered as an empty box.
"""
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets" / "fonts"

CANDIDATES = {
    "title": [ASSETS / "Anton-Regular.ttf"],
    "label": [ASSETS / "BebasNeue-Regular.ttf"],
    "sans": [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
             Path("C:/Windows/Fonts/segoeui.ttf")],
    "sans_bold": [Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
                  Path("C:/Windows/Fonts/segoeuib.ttf")],
    "mono": [Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
             Path("C:/Windows/Fonts/consola.ttf")],
}


@lru_cache(maxsize=None)
def path(kind):
    for p in CANDIDATES[kind]:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"no font for {kind!r}; tried {CANDIDATES[kind]}")


@lru_cache(maxsize=256)
def font(kind, size):
    return ImageFont.truetype(path(kind), int(size))


def missing_glyphs(kind, text):
    """Characters the font draws as nothing or as its 'missing' box."""
    f = font(kind, 48)
    notdef = f.getmask("\uffff").getbbox()
    missing = set()
    for ch in set(text):
        if ch.isspace():
            continue
        box = f.getmask(ch).getbbox()
        if box is None or box == notdef:
            missing.add(ch)
    return missing


def wrap(text, kind, size, max_w):
    f = font(kind, size)
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if f.getlength(trial) <= max_w or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def fit(text, kind, max_w, max_h, max_size, min_size=14, spacing=1.18):
    """The largest size at which `text`, wrapped, fits the box. Never clips:
    at the floor size a word longer than the box is still placed whole."""
    size = max_size
    while size >= min_size:
        lines = wrap(text, kind, size, max_w)
        f = font(kind, size)
        widest = max(f.getlength(l) for l in lines) if lines else 0
        height = len(lines) * size * spacing
        if widest <= max_w and height <= max_h:
            return size, lines
        size -= 2
    return min_size, wrap(text, kind, min_size, max_w)
