"""Second pass over the niche search: only the videos that are this niche.

    python automation/niche_refine.py

YouTube search for "1970s america footage" also returns music videos of the
period, object restorations and concerts. A video counts here when its title
names the era (a decade or a year 1900-1989) AND archive life on film (footage,
life, a city, colour, what it was like ...), and is not music, a live show, a
restoration of an object, or a Short. Rewrites analysis/niche.csv with a
"relevant" column, analysis/niche_relevant.csv, analysis/channels_relevant.csv
and analysis/thumbs_top30.jpg from the relevant videos only.
"""
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation.niche import collage  # noqa: E402

A = ROOT / "analysis"
ERA = re.compile(r"\b(19[0-8]0'?s|19[0-8]\d|(nineteen|the) (thirties|forties|fifties|sixties|seventies|eighties)|"
                 r"\d+ years ago|100 years|century)\b", re.I)
NICHE = re.compile(r"\b(footage|film|films|life|living|lived|america|american|usa|u\.s\.|new york|nyc|los angeles|"
                   r"chicago|city|cities|street|streets|people|colou?r|colori[sz]ed|kodachrome|vintage|archive|"
                   r"newsreel|daily|everyday|was like|were like|suburb\w*|town|family|families|road trip|"
                   r"documentary|restored footage|time travel|then and now|home movies?)\b", re.I)
NOT_NICHE = re.compile(r"(official (music |lyric )?video|music video|\blyrics?\b|\blive (at|in|from)\b|\blive\b.*\b19\d\d\)|"
                       r"\bvevo\b|\bfeat\.|\bft\.|remix|\bcover\b|karaoke|restoration\b(?!.*footage)|i restored|"
                       r"\basmr\b|#shorts|gameplay|minecraft|\breaction\b|tier list|trailer)", re.I)


def relevant(r):
    t = r["title"]
    if r["channel"].lower().endswith("vevo") or NOT_NICHE.search(t):
        return False
    return bool(ERA.search(t)) and bool(NICHE.search(t)) and float(r["dur_min"]) >= 3


def main():
    rows = list(csv.DictReader(open(A / "niche.csv", encoding="utf-8-sig")))
    for r in rows:
        r["relevant"] = "yes" if relevant(r) else ""
    with open(A / "niche.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rel = [r for r in rows if r["relevant"]]
    rel.sort(key=lambda r: -int(r["views"]))
    with open(A / "niche_relevant.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rel)
    chans = {c["channel_id"]: c for c in csv.DictReader(open(A / "channels.csv", encoding="utf-8-sig"))}
    by_ch = {}
    for r in rel:
        by_ch.setdefault(r["channel_id"], []).append(r)
    crow = []
    for cid, vs in by_ch.items():
        c = chans.get(cid, {})
        crow.append({**c, "niche_videos_found": len(vs), "niche_views_found": sum(int(v["views"]) for v in vs),
                     "best_title": vs[0]["title"], "best_views": vs[0]["views"]})
    crow.sort(key=lambda c: -c["niche_views_found"])
    with open(A / "channels_relevant.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0].keys()))
        w.writeheader()
        w.writerows(crow)
    top = rel[:30]
    for r in top:
        r["views"] = int(r["views"])
        r["dur_min"] = float(r["dur_min"])
    collage(top, A / "thumbs_top30.jpg")
    t100 = rel[:100]
    s = {
        "relevant_videos": len(rel), "relevant_channels": len(crow),
        "top100_median_views": int(statistics.median(int(r["views"]) for r in t100)),
        "top100_median_minutes": statistics.median(float(r["dur_min"]) for r in t100),
        "top100_minutes_buckets": {b: sum(lo <= float(r["dur_min"]) < hi for r in t100)
                                   for b, (lo, hi) in {"3-10": (3, 10), "10-20": (10, 20), "20-40": (20, 40),
                                                       "40+": (40, 1e9)}.items()},
        "top100_colour": sum(r["colour"] == "colour" for r in t100),
        "top100_bw": sum(r["colour"] == "bw" for r in t100),
        "newcomers": [{k: c[k] for k in ("channel", "created", "subs", "median_views_recent", "median_dur_min",
                                         "days_between_uploads", "niche_videos_found", "best_title")}
                      for c in crow if c.get("newcomer") == "True"],
        "top_channels": [{k: c.get(k) for k in ("channel", "created", "subs", "median_views_recent",
                                                 "median_dur_min", "days_between_uploads", "niche_videos_found",
                                                 "niche_views_found", "best_title")} for c in crow[:15]],
    }
    (A / "summary_relevant.json").write_text(json.dumps(s, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in s.items() if k not in ("newcomers", "top_channels")}, indent=1))
    for c in s["top_channels"]:
        print(f"{c['channel'][:28]:28} created {c['created']} subs {int(c['subs'] or 0):>9,} median {int(c['median_views_recent'] or 0):>9,} "
              f"dur {c['median_dur_min']} every {c['days_between_uploads']}d | {c['best_title'][:60]}")
    print("NEWCOMERS:")
    for c in s["newcomers"]:
        print(f"  {c['channel'][:28]:28} created {c['created']} subs {int(c['subs'] or 0):>9,} median {int(c['median_views_recent'] or 0):>9,} "
              f"dur {c['median_dur_min']} every {c['days_between_uploads']}d | {c['best_title'][:60]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
