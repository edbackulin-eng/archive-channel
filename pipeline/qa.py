"""The QA gate: an episode is released only when every check passes.

  match_share     shots whose frame Gemini found to match the line, era, place and
                  to be clean, over all clip shots; unchecked counts as unmatched
                  (config "qa.match_share_min", 85 %)
  repeats         no film stretch shown twice within 3 minutes
  no_voice_share  20-30 % of the running time without voice, every film break
                  3-8 s (config "breathing")
  loudness        -14 LUFS +- 1, true peak <= -1 dBTP
  hook            the first 30 s ask a question and promise more ("not all", "more")
  length          15-20 minutes
  thumbnail       bottom-right 25 % x 20 % empty, words readable at 336x189, no
                  text on a face, a year in big numbers, 2-4 hook words, and sharp:
                  the photo half over the thresholds at 336x189 and 1280x720
                  (pipeline/thumbnail.py SHARP_MIN, SHARP_MIN_FULL), and its picture shows
                  what the words name (a house's price over a house; Gemini-checked)
  licences        every shot's film has a free licence recorded
  prices          the angle: a price tag in the hook, at least 6 in the episode,
                  never more than 6 minutes without one, each rendered with its
                  today's-dollars figure (config "prices")
The report goes to episodes/<slug>/qa.json; release reads "passed".

Owner 12.11 - two tiers. Only real faults stop an upload ("blocking": true): the wrong thing on
screen for the words (match_share under BLOCK_MATCH), broken sound or silence (sound: clipping,
loudness far off, 3 s of silence), broken picture (continuity: gaps, a video shorter or longer
than its plan), a length outside the range, a source without a licence, a thumbnail whose words
cannot be read or sit on a face. The rest - the 85 % match target, repeats, breathing, the hook's
words, prices, sharpness, the viewer's marks - is a score: reported, aimed at, never a stop.
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))


def loudness(video):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-af",
                        "ebur128=peak=true:framelog=quiet", "-f", "null", "-"], capture_output=True, text=True)
    i = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", r.stderr)
    p = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", r.stderr)
    return (float(i[-1]) if i else None), (float(p[-1]) if p else None)


BLOCK_MATCH = 0.6          # under this share of matching shots the film shows the wrong things


def silences(video, db=-50, min_s=3.0):
    """Stretches of min_s or more with the whole mix under db (ffmpeg silencedetect)."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-af",
                        f"silencedetect=noise={db}dB:d={min_s}", "-f", "null", "-"], capture_output=True, text=True)
    return [round(float(x), 1) for x in re.findall(r"silence_start:\s*(-?[\d.]+)", r.stderr)]


def check(plan, timeline, video, thumb_report=None, full_episode=True):
    shots = [s for s in plan["shots"] if s["kind"] == "clip"]
    total = plan["shots"][-1]["end"]
    res = {}
    ok_v = [s for s in shots if (s.get("vision") or {}).get("ok")]
    res["match_share"] = {"value": round(len(ok_v) / max(1, len(shots)), 3),
                          "unchecked": sum((s.get("vision") or {}).get("status") != "checked" for s in shots),
                          "pass": len(ok_v) / max(1, len(shots)) >= CFG["qa"]["match_share_min"]}
    reps = []
    for i, a in enumerate(shots):
        for b in shots[i + 1:]:
            if a.get("film") and a.get("film") == b.get("film") and abs(a["start"] - b["start"]) < CFG["edit"]["no_repeat_within_s"]:
                da, db = a["end"] - a["start"], b["end"] - b["start"]
                if not (a["src_start"] + da < b["src_start"] or b["src_start"] + db < a["src_start"]):
                    reps.append([round(a["start"], 1), round(b["start"], 1)])
    res["repeats"] = {"value": reps, "pass": not reps}
    voiced = sum(t["end"] - t["start"] for t in timeline)
    breaks = [t["film_before"] for t in timeline if t.get("film_before")]
    share = 1 - voiced / total if total else 0
    lo, hi = CFG["qa"]["no_voice_share"]
    br = CFG["breathing"]
    res["no_voice_share"] = {"value": round(share, 3), "breaks": len(breaks),
                             "break_s": [min(breaks, default=0), max(breaks, default=0)],
                             "pass": (lo <= share <= hi or not full_episode)
                             and all(br["min_s"] <= b <= br["max_s"] for b in breaks)}
    lufs, peak = loudness(video)
    res["loudness"] = {"lufs": lufs, "true_peak": peak,
                       "pass": lufs is not None and abs(lufs - CFG["sound"]["target_lufs"]) <= 1.0
                       and peak <= CFG["sound"]["true_peak"] + 0.2}
    hook = " ".join(t["text"] for t in timeline if t["start"] < CFG["episode"]["hook_seconds"] + 6)
    res["hook"] = {"question": "?" in hook, "promise": bool(re.search(r"not all|more to|wait until|we'll come back", hook, re.I))}
    res["hook"]["pass"] = res["hook"]["question"] and res["hook"]["promise"]
    gaps = [round(b["start"] - a["end"], 3) for a, b in zip(plan["shots"], plan["shots"][1:]) if abs(b["start"] - a["end"]) > 0.01]
    vdur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                                capture_output=True, text=True).stdout.strip() or 0)
    res["continuity"] = {"gaps": gaps[:10], "video_s": round(vdur, 2), "plan_s": round(total, 2),
                         "pass": not gaps and abs(vdur - total) <= 0.3}
    mins = total / 60
    res["length"] = {"minutes": round(mins, 1),
                     "pass": (CFG["episode"]["minutes_min"] <= mins <= CFG["episode"]["minutes_max"]) or not full_episode}
    res["licences"] = {"missing": [s.get("film") for s in shots if not (s.get("item") or {}).get("licence")]}
    res["licences"]["pass"] = not res["licences"]["missing"]
    pc = CFG["prices"]
    tagged = [s for s in plan["shots"] if s.get("price")]
    at = [round(s["start"], 1) for s in tagged]
    gap = max((b - a for a, b in zip([0.0] + at, at + [total])), default=total)
    in_hook = any(a < CFG["episode"]["hook_seconds"] + 6 for a in at)
    rendered = all(s.get("price_tag") for s in tagged)
    res["prices"] = {"tags": len(tagged), "at_s": at, "longest_without_s": round(gap, 1), "in_hook": in_hook,
                     "rendered": rendered,
                     "pass": rendered and in_hook and (not full_episode or (len(tagged) >= pc["per_episode_min"]
                                                                            and gap <= pc["max_gap_s"]))}
    if thumb_report is not None:
        t = thumb_report
        res["thumbnail"] = {"corner_empty": t.get("corner_empty"), "readable_small": t.get("readable_small"),
                            "text_over_face": t.get("text_over_face"),
                            "sharp_336": t.get("sharp_336"), "sharp_1280": t.get("sharp_1280"),
                            "matches_words": t.get("matches_words"), "subject": t.get("subject"),
                            "pass": bool(t.get("corner_empty") and t.get("readable_small") and not t.get("text_over_face")
                                         and t.get("sharp_ok") and t.get("matches_words"))}
    # owner 12.11: what stops an upload, and what is only a score
    quiet = silences(video) if Path(str(video)).exists() else []
    res["sound"] = {"lufs": lufs, "true_peak": peak, "silent_at": quiet[:10],
                    "pass": lufs is not None and abs(lufs - CFG["sound"]["target_lufs"]) <= 3.0
                    and (peak is None or peak <= -0.5) and not quiet, "blocking": True}
    res["wrong_on_screen"] = {"value": res["match_share"]["value"], "min": BLOCK_MATCH,
                              "pass": res["match_share"]["value"] >= BLOCK_MATCH, "blocking": True}
    if thumb_report is not None:
        t = thumb_report
        res["thumbnail_text"] = {"pass": bool(t.get("readable_small") and not t.get("text_over_face")), "blocking": True}
    for k in ("continuity", "length", "licences"):
        res[k]["blocking"] = True
    for k, v in res.items():
        if isinstance(v, dict) and "pass" in v:
            v.setdefault("blocking", False)
    scored = [v for v in res.values() if isinstance(v, dict) and "pass" in v and not v["blocking"]]
    res["score"] = round(sum(bool(v["pass"]) for v in scored) / max(1, len(scored)), 2)
    res["passed"] = all(v["pass"] for v in res.values() if isinstance(v, dict) and v.get("blocking"))
    return res
