"""Documentary narration with Kokoro on the CPU, read from the delivery markup.

The channel speaks at about 150 words a minute over the whole programme,
pauses included - an archive documentary, unhurried. Each voice has its own
natural rate, so the pace is calibrated per voice: the script is rendered once
at the markup's nominal speeds, every speed is scaled so the total lands on the
target, and it is rendered again.

Voices taken by the owner's other channels (am_michael, bm_lewis, am_fenrir)
are never used here (config/channel.json "voice").

Returns the audio and, for every line, where it starts and ends, so the edit
can cut to the narration without a separate alignment pass.
"""
import json
import re
from pathlib import Path

import numpy as np

from core import delivery
from core.logger import get_logger
from core.voice import SAMPLE_RATE, _trim_silence, peak_normalize

log = get_logger("narrate")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["voice"]
TARGET_WPM = float(CFG["wpm"])
SPEED_MIN, SPEED_MAX = 0.80, 1.15
SLOW_LEAD = (0.6, 0.9)
SENTENCE_TAIL_S = 0.32
QUESTION_EXTRA_S = 0.14
_SENT = re.compile(r"(?<=[.!?])\s+")


def _lang(voice):
    return "en-gb" if voice.startswith("b") else "en-us"


def _lexicon():
    data = json.loads((ROOT / "voice" / "pronunciations.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def respell(text, lexicon):
    for word in sorted(lexicon, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lexicon[word], text)
    return text


def spoken(text):
    """What Kokoro should say: years and prices read the way a narrator says them."""
    text = re.sub(r"\$([\d.,]+) (million|billion)\b", lambda m: f"{m.group(1).replace(',', '')} {m.group(2)} dollars", text)
    text = re.sub(r"\$(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d\d))?",
                  lambda m: f"{m.group(1).replace(',', '')} dollars" + (f" {int(m.group(2))} cents" if m.group(2) and int(m.group(2)) else ""),
                  text)
    text = re.sub(r"\b(\d{2})(\d{2})s\b", lambda m: f"{m.group(1)} {m.group(2)}s" if m.group(1) == "19" else m.group(0), text)
    return text


def _render(kokoro, lines, voice, scale, lexicon):
    chunks, timeline, t = [], [], 0.0
    for line in lines:
        if line.film_before:
            # a film break: no voice; the edit raises the film's own sound here
            chunks.append(np.zeros(int(SAMPLE_RATE * line.film_before), dtype=np.float32))
            t += line.film_before
        pause = line.pause_before
        if line.kind == "slow":
            n = len(line.text.split())
            pause = max(pause, round(SLOW_LEAD[0] + (SLOW_LEAD[1] - SLOW_LEAD[0]) * min(1.0, n / 10), 2))
        if pause:
            chunks.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
            t += pause
        speed = min(SPEED_MAX, max(SPEED_MIN, line.speed * scale))
        start = t
        for sentence in [s for s in _SENT.split(line.text) if s.strip()]:
            audio, _ = kokoro.create(respell(spoken(sentence), lexicon), voice=voice, speed=speed,
                                     lang=_lang(voice))
            audio = _trim_silence(np.asarray(audio, dtype=np.float32))
            chunks.append(audio)
            t += len(audio) / SAMPLE_RATE
            tail = SENTENCE_TAIL_S + (QUESTION_EXTRA_S if sentence.rstrip().endswith("?") else 0)
            chunks.append(np.zeros(int(SAMPLE_RATE * tail), dtype=np.float32))
            t += tail
        timeline.append({"text": line.text, "kind": line.kind, "start": round(start, 3),
                         "end": round(t, 3), "speed": round(speed, 3),
                         "film_before": line.film_before, "chapter": line.chapter, "price": line.price})
    audio = peak_normalize(np.concatenate(chunks)) if chunks else np.zeros(1, np.float32)
    return audio, timeline


def synthesize(kokoro, script, voice=None, wpm=TARGET_WPM):
    voice = voice or CFG["voice"]
    if voice in CFG["taken_elsewhere"]:
        raise ValueError(f"{voice} is used by another channel")
    lexicon = _lexicon()
    lines = delivery.parse(script)
    words = sum(len(l.text.split()) for l in lines)
    scale = 1.0
    for _ in range(3):              # pauses do not scale with speed, so it takes a pass or two
        audio, timeline = _render(kokoro, lines, voice, scale, lexicon)
        film = sum(l.film_before for l in lines)
        # the pace is measured while the narrator speaks; film breaks are not speech
        got = words / ((len(audio) / SAMPLE_RATE - film) / 60)
        if abs(got - wpm) / wpm <= 0.02:
            break
        scale *= wpm / got
    log.info("%s: %d words, %.1f s with %.1f s of film breaks, %.0f wpm speaking (x%.2f)", voice, words,
             len(audio) / SAMPLE_RATE, film, got, scale)
    return audio, timeline


def load_kokoro():
    from kokoro_onnx import Kokoro
    models = ROOT / "assets" / "models"
    return Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
