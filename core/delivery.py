"""Delivery markup: how a line is spoken, written into the script itself.

    > fast line        quick and confident: hooks, transitions
    < slow line        slower, with a pause before it: reveals, key facts
    plain line         the base pace
    [pause 0.8]        a short silence in the narration, in seconds
    [film 5]           the narrator stops and the film speaks for 3-8 seconds:
                       its own sound, the music and the room come up
                       (pipeline/breathing.py places these; the edit lays
                       the most vivid shot of the passage under them)
    ## Title | place | years   a chapter starts here (never spoken)
    [price 7990 1949 | A new Levitt house]
                       a price tag on screen over the next line's first shot:
                       then, and now in today's dollars (pipeline/prices.py)
    [today 7990 1949]  inside a line: spoken as "about $108,000", worked out
                       from the BLS CPI-U, never written by the text model

The channel's rule (config/channel.json "breathing"): 20-30 % of the running
time has no voice, in film breaks of 3-8 s - on the most vivid shots and
between chapters. Every TTS engine reads the same parse, so swapping the voice
does not change the timing the script was written for.
"""
import re
from dataclasses import dataclass

FAST, BASE, SLOW = 1.10, 1.02, 0.95
SLOW_LEAD_S = 0.75
LINE_GAP_S = 0.12
_PAUSE = re.compile(r"^\[pause\s+([0-9.]+)\]$", re.I)
_FILM = re.compile(r"^\[film\s+([0-9.]+)\]$", re.I)
_CHAPTER = re.compile(r"^##\s*(.+)$")
_PRICE = re.compile(r"^\[price\s+([\d.,]+)\s+(\d{4})\s*(?:\|\s*([^\]]*))?\]$", re.I)


@dataclass
class Line:
    text: str
    kind: str              # "fast" | "base" | "slow"
    speed: float
    pause_before: float    # a narration pause (short silence)
    film_before: float = 0.0   # a film break: no voice, the chronicle speaks
    chapter: str = ""      # "Title | place | years" when a chapter opens on this line
    price: dict = None     # {"amount", "year", "label"}: a price tag over this line


def parse(script):
    from pipeline import prices
    lines, pending, film, chapter, price = [], 0.0, 0.0, "", None
    for raw in script.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        m = _PAUSE.match(raw)
        if m:
            pending += float(m.group(1))
            continue
        m = _FILM.match(raw)
        if m:
            film += float(m.group(1))
            continue
        m = _CHAPTER.match(raw)
        if m:
            chapter = m.group(1).strip()
            continue
        m = _PRICE.match(raw)
        if m:
            price = {"amount": m.group(1), "year": int(m.group(2)), "label": (m.group(3) or "").strip()}
            continue
        raw = prices.spoken_today(raw)
        if raw.startswith(">"):
            kind, speed, text = "fast", FAST, raw[1:].strip()
        elif raw.startswith("<"):
            kind, speed, text = "slow", SLOW, raw[1:].strip()
        else:
            kind, speed, text = "base", BASE, raw
        gap = pending if pending else LINE_GAP_S
        if kind == "slow":
            gap = max(gap, SLOW_LEAD_S)
        lines.append(Line(text, kind, speed, 0.0 if not lines and not pending else gap, film, chapter, price))
        pending, film, chapter, price = 0.0, 0.0, "", None
    return lines


def plain_text(script):
    """The words alone, for subtitles and word counts."""
    return " ".join(l.text for l in parse(script))


def price_tags(script):
    """[(line index, price)] in order."""
    return [(i, l.price) for i, l in enumerate(parse(script)) if l.price]


def chapters(script):
    """[(line index, 'Title | place | years')] in order."""
    return [(i, l.chapter) for i, l in enumerate(parse(script)) if l.chapter]
