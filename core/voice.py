"""Kokoro narration: synthesised sentence by sentence, then joined with real pauses.

kokoro-onnx silently truncates anything past ~510 phoneme tokens, so a whole
paragraph handed to it in one call comes back cut off mid-thought. Going sentence
by sentence avoids that and is also what makes the delivery sound alive: each
sentence gets its own pace, and the gap after it depends on how it ended.
"""
import re

import numpy as np

from core.logger import get_logger
from core.text import split_sentences, word_count

log = get_logger("voice")

SAMPLE_RATE = 24000

# Silence after a sentence, by how it ended. A question mark lands differently
# from a full stop, and a colon leans forward into what follows.
PAUSE_S = {".": 0.34, "!": 0.40, "?": 0.44, ":": 0.26, ";": 0.26, ",": 0.18}
DEFAULT_PAUSE_S = 0.30
PARAGRAPH_PAUSE_S = 0.70

# A sentence longer than this is split again at commas before being synthesised.
MAX_WORDS_PER_CHUNK = 30

# Pace drifts slightly from sentence to sentence; a perfectly even tempo is the
# single thing that makes synthetic narration sound synthetic. Deterministic, so
# the same script always renders identically.
SPEED_BASE = 1.0
SPEED_SWING = 0.045
# A short sentence after a long one reads as emphasis, so slow it a little more.
SHORT_SENTENCE_WORDS = 7
SHORT_SENTENCE_SPEED = 0.93


def _split_long(sentence):
    """Break an over-long sentence at commas so Kokoro never truncates it."""
    if word_count(sentence) <= MAX_WORDS_PER_CHUNK:
        return [sentence]
    parts, current = [], []
    for piece in re.split(r"(?<=,)\s+", sentence):
        current.append(piece)
        if word_count(" ".join(current)) >= MAX_WORDS_PER_CHUNK:
            parts.append(" ".join(current))
            current = []
    if current:
        tail = " ".join(current)
        if parts and word_count(tail) < 4:
            parts[-1] = f"{parts[-1]} {tail}"
        else:
            parts.append(tail)
    return parts


def _speed_for(index, words):
    """A gentle, repeatable rise and fall in pace across the narration."""
    if words <= SHORT_SENTENCE_WORDS:
        return SHORT_SENTENCE_SPEED
    return SPEED_BASE + SPEED_SWING * np.sin(index * 1.7)


def _pause_after(sentence):
    ending = sentence.strip()[-1:] if sentence.strip() else ""
    return PAUSE_S.get(ending, DEFAULT_PAUSE_S)


def _trim_silence(samples, threshold=0.004, keep_s=0.02):
    """Drop the dead air Kokoro leaves at both ends, so our own pauses are exact."""
    loud = np.flatnonzero(np.abs(samples) > threshold)
    if loud.size == 0:
        return samples
    keep = int(keep_s * SAMPLE_RATE)
    start = max(0, loud[0] - keep)
    end = min(len(samples), loud[-1] + keep)
    return samples[start:end]


def synthesize(kokoro, text, voice, on_chunk=None):
    """Render `text` with `voice`. Returns (samples float32, sample_rate).

    `text` may contain blank lines; those become longer pauses.
    """
    pieces = []
    index = 0
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]

    for p_i, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        for s_i, sentence in enumerate(sentences):
            for chunk in _split_long(sentence):
                words = word_count(chunk)
                if not words:
                    continue
                speed = _speed_for(index, words)
                samples, rate = kokoro.create(chunk, voice=voice, speed=speed, lang="en-us")
                if rate != SAMPLE_RATE:
                    raise RuntimeError(f"unexpected sample rate {rate}")
                samples = _trim_silence(np.asarray(samples, dtype=np.float32))
                pieces.append(samples)
                if on_chunk:
                    on_chunk(index, chunk, len(samples) / SAMPLE_RATE)
                index += 1
            pieces.append(np.zeros(int(_pause_after(sentence) * SAMPLE_RATE), dtype=np.float32))
        if p_i < len(paragraphs) - 1:
            pieces.append(np.zeros(int(PARAGRAPH_PAUSE_S * SAMPLE_RATE), dtype=np.float32))

    if not pieces:
        raise ValueError("nothing to synthesise")
    audio = np.concatenate(pieces)
    log.info("synthesised %d chunks, %.1fs with %s", index, len(audio) / SAMPLE_RATE, voice)
    return audio, SAMPLE_RATE


def peak_normalize(samples, target_db=-3.0):
    """Bring the loudest point to `target_db`. Final loudness is set later by ffmpeg
    loudnorm; this only stops the WAV from clipping or sitting too quiet."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-6:
        return samples
    return samples * (10 ** (target_db / 20) / peak)
