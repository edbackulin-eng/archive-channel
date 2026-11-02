"""Where the narrator stops and the film speaks.

The channel's rule (config/channel.json "breathing"): the narrator does not
talk without a break. 20-30 % of the running time has no voice, in film breaks
of 3-8 seconds, where the chronicle's own sound, the music and the atmosphere
carry the picture - above all on the most vivid shots and between chapters.

place() writes [film N] lines into a script that has none (or too few), in
this order of priority until the share reaches the target:
  1. between chapters, 7 s, so every chapter opens on the film;
  2. right after the hook, 4 s;
  3. after each reveal (a "<" line), 4-5 s - the moment the viewer looks;
  4. after the longest paragraphs, 3-4 s, spread through the episode.
The share is computed against the speech time at the channel's speaking pace,
so it holds whatever the voice. measure() gives the real share from a rendered
timeline, which the QA gate checks.
"""
import json
import re
from pathlib import Path

from core import delivery

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
RULE = CFG["breathing"]
WPM = CFG["voice"]["wpm"]


def _speech_s(lines):
    words = sum(len(l.text.split()) for l in lines)
    pauses = sum(l.pause_before for l in lines)
    return words / WPM * 60 + pauses


def share_of(script):
    lines = delivery.parse(script)
    film = sum(l.film_before for l in lines)
    total = _speech_s(lines) + film
    return film / total if total else 0.0


def place(script, target=None):
    """The script with film breaks placed until their share reaches `target`."""
    target = target if target is not None else RULE["target_share"]
    if share_of(script) >= RULE["min_share"]:
        return script
    raw = script.splitlines()
    speech = _speech_s(delivery.parse(script))
    # the episode must still end inside its length: the share is lowered toward
    # the minimum when the speech alone is long, never below it
    max_total = CFG["episode"]["minutes_max"] * 60
    if speech / (1 - target) > max_total:
        target = max(RULE["min_share"], 1 - speech / max_total)
    need = target * speech / (1 - target)          # seconds of film for the target share
    inserts = {}                                   # text line index -> seconds, inserted BEFORE it
    text_lines = [i for i, r in enumerate(raw) if r.strip() and not r.strip().startswith(("[", "##"))]

    def slot(i):
        """A break belongs to the spoken line it precedes, so two never stack."""
        return next((j for j in text_lines if j >= i), None)

    first_chapter = next((i for i, r in enumerate(raw) if r.strip().startswith("##")), 0)

    def add(i, s):
        nonlocal need
        # the hook runs unbroken: its 30 seconds are voice, picture and promise
        if i < first_chapter:
            return
        i = slot(i)
        if need <= 0 or i is None or i in inserts:
            return
        s = max(RULE["min_s"], min(RULE["max_s"], s))
        inserts[i] = s
        need -= s

    text_idx = [i for i, r in enumerate(raw) if r.strip() and not r.strip().startswith(("[", "##"))]
    chapters = [i for i, r in enumerate(raw) if r.strip().startswith("##")]
    # 1. every chapter after the first opens on the film; 2. the hook ends in film
    for c in chapters[1:]:
        add(c, RULE["chapter_s"])
    if chapters:
        add(chapters[0], RULE["after_hook_s"])
    # 3. after reveals
    for i, r in enumerate(raw):
        if r.strip().startswith("<"):
            nxt = next((j for j in text_idx if j > i), None)
            if nxt is not None and nxt not in chapters:
                add(nxt, RULE["reveal_s"])
    # 4. spread the rest at paragraph ends ([pause] lines), evenly
    if need > 0:
        pauses = [i for i, r in enumerate(raw) if r.strip().lower().startswith("[pause")]
        step = max(1, len(pauses) // max(1, int(need // RULE["spread_s"]) + 1))
        for i in pauses[::step]:
            add(i, RULE["spread_s"])
    # 5. still short: lengthen the breaks already placed, up to max_s ...
    if need > 0 and inserts:
        for i in sorted(inserts, key=lambda i: inserts[i]):
            extra = min(RULE["max_s"] - inserts[i], need)
            inserts[i] += extra
            need -= extra
            if need <= 0:
                break
    # ... then add breaks evenly between lines that have none yet
    if need > 0:
        free = [i for i in text_idx if i not in inserts and i - 1 not in inserts and i + 1 not in inserts]
        n = int(need // RULE["spread_s"]) + 1
        step = max(2, len(free) // max(1, n))
        for i in free[step // 2::step]:
            add(i, RULE["spread_s"])
    out = []
    for i, r in enumerate(raw):
        if i in inserts:
            out.append(f"[film {inserts[i]:g}]")
        out.append(r)
    return "\n".join(out)


def measure(timeline, total_s):
    """Share of the running time without voice, from a rendered narration timeline,
    and the film breaks found in it."""
    voiced = sum(t["end"] - t["start"] for t in timeline)
    breaks = [t["film_before"] for t in timeline if t.get("film_before")]
    return {"no_voice_share": round(1 - voiced / total_s, 3) if total_s else 0.0,
            "film_breaks": len(breaks), "film_break_s": [round(b, 1) for b in breaks],
            "breaks_in_range": all(RULE["min_s"] <= b <= RULE["max_s"] for b in breaks)}
