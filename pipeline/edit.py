"""Put an episode (or its first minute) together from a shot plan and the narration.

A shot plan is a list of shots on the narration's clock:
  {"start", "end", "kind": "clip" | "card",
   "clip": restored 1920x1080 clip (pipeline/clips.cut), "year", "place",
   "film": archive id, "film_break": true when the narrator is silent here,
   "chapter": "Title | place | years" for a card}

Picture: every shot rendered to its exact length and joined; a clip carries its
caption ("1957 - United States") from the film's own metadata; a card is the
chapter map (pipeline/cards.py).

Sound (config/channel.json "sound", "breathing"):
  voice      two-pass loudnorm
  music      22 dB under the voice while it speaks, up 6 dB in film breaks
  film sound the chronicle's own track, low under the voice (-30 dB) and raised
             in film breaks (-12 dB), where the film speaks for itself
  all        two-pass loudnorm to -14 LUFS, true peak -1 dBTP
"""
import json
import subprocess
from pathlib import Path

import soundfile as sf

from core.logger import get_logger
from pipeline import cards

log = get_logger("edit")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))
SND, BR = CFG["sound"], CFG["breathing"]
W, H, FPS, SR = 1920, 1080, 30, 48000


def _ff(args, timeout=1800):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stderr


def _lufs(path):
    err = _ff(["-i", str(path), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"])
    return json.loads(err[err.rfind("{"):err.rfind("}") + 1])


def _norm(src, dest, target, tp=-1.5):
    m = _lufs(src)
    _ff(["-y", "-i", str(src), "-af",
         f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:measured_TP={m['input_tp']}:"
         f"measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true",
         "-ar", str(SR), "-ac", "2", str(dest)])
    return dest


def _shot_video(s, k, work):
    d = s["end"] - s["start"]
    out = work / f"v_{k:03d}.mp4"
    if s["kind"] == "card":
        return cards.chapter_card(s["chapter"], d, out, work)
    cap = work / f"cap_{k:03d}.png"
    cards.caption_png(s.get("year"), s.get("place") or "United States", cap)
    # the caption fades in after half a second and out before the cut
    fade = (f"[1:v]format=rgba,fade=t=in:st=0.4:d=0.4:alpha=1,"
            f"fade=t=out:st={max(0.5, d - 0.6):.3f}:d=0.4:alpha=1[c];[0:v][c]overlay=0:0,format=yuv420p")
    _ff(["-y", "-stream_loop", "-1", "-i", str(s["clip"]), "-loop", "1", "-i", str(cap), "-filter_complex", fade,
         "-t", f"{d:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-r", str(FPS), str(out)])
    return out


def _shot_sound(s, k, work):
    """The shot's own film sound, exactly its length (silence for a card or a mute film)."""
    d = s["end"] - s["start"]
    out = work / f"a_{k:03d}.wav"
    if s["kind"] == "clip":
        try:
            _ff(["-y", "-stream_loop", "-1", "-i", str(s["clip"]), "-map", "0:a:0", "-t", f"{d:.3f}",
                 "-ar", str(SR), "-ac", "2", str(out)])
            return out
        except RuntimeError:
            pass
    _ff(["-y", "-f", "lavfi", "-i", f"anullsrc=r={SR}:cl=stereo", "-t", f"{d:.3f}", str(out)])
    return out


def _windows(shots):
    return [(s["start"], s["end"]) for s in shots if s.get("film_break")]


def _vol_expr(base_db, up_db, windows):
    if not windows:
        return f"{base_db}dB"
    inside = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in windows)
    lin_base, lin_up = 10 ** (base_db / 20), 10 ** ((base_db + up_db) / 20)
    return f"'if({inside},{lin_up:.5f},{lin_base:.5f})'"


def render(plan, narration_wav, music_mp3, out, work):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    shots = plan["shots"]
    total = shots[-1]["end"]
    vids = [_shot_video(s, k, work) for k, s in enumerate(shots)]
    lst = work / "v.txt"
    lst.write_text("".join(f"file '{Path(v).resolve().as_posix()}'\n" for v in vids), encoding="utf-8")
    picture = work / "picture.mp4"
    _ff(["-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(picture)])

    auds = [_shot_sound(s, k, work) for k, s in enumerate(shots)]
    alst = work / "a.txt"
    alst.write_text("".join(f"file '{Path(a).resolve().as_posix()}'\n" for a in auds), encoding="utf-8")
    film = work / "film.wav"
    _ff(["-y", "-f", "concat", "-safe", "0", "-i", str(alst), "-c", "copy", str(film)])

    voice = _norm(narration_wav, work / "voice.wav", -16.0)
    bed_gain = (-16.0 + SND["music_under_voice_db"]) - float(_lufs(music_mp3)["input_i"])
    film_n = _norm(film, work / "film_n.wav", -16.0) if _lufs(film)["input_i"] not in ("-inf", "inf") else film
    win = _windows(shots)
    graph = (f"[1:a]aresample={SR},aformat=channel_layouts=stereo,volume={bed_gain:.2f}dB,atrim=0:{total:.3f},"
             f"volume={_vol_expr(0, BR['music_up_db'], win)}:eval=frame[m];"
             f"[2:a]aresample={SR},volume={_vol_expr(SND['original_sound_db'], BR['film_sound_db'] - SND['original_sound_db'], win)}:eval=frame[f];"
             f"[0:a][m][f]amix=inputs=3:duration=first:normalize=0[pre]")
    pre = work / "premix.wav"
    _ff(["-y", "-i", str(voice), "-stream_loop", "-1", "-i", str(music_mp3), "-i", str(film_n), "-filter_complex", graph,
         "-map", "[pre]", "-t", f"{total:.3f}", "-ar", str(SR), "-ac", "2", str(pre)])
    mix = _norm(pre, work / "mix.wav", SND["target_lufs"], SND["true_peak"] - 0.3)
    _ff(["-y", "-i", str(picture), "-i", str(mix), "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "fast",
         "-b:v", "12M", "-maxrate", "16M", "-bufsize", "24M", "-c:a", "aac", "-b:a", "192k", "-shortest",
         "-movflags", "+faststart", str(out)])
    return {"seconds": round(total, 2), "shots": len(shots), "film_breaks": len(win),
            "music_bed_gain_db": round(bed_gain, 1)}
