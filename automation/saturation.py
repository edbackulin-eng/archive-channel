"""How crowded the chosen subtopic is, and how its possible sub-niches compare.

    python automation/saturation.py           (YT_API_KEY as for niche.py)

For each segment (the subtopic itself and candidate sub-niches), one YouTube
search per phrase: by view count, published in the last 24 months, English.
Of the hits (3 min or longer, not music), per segment:
  videos         how many were published in 24 months (supply)
  per_month      how many of them in the last 90 days, per month (current pace)
  median_views   of the segment's videos; share_100k: share with >= 100k views
  channels_100k  channels in the segment whose recent long uploads have a median
                 of at least 100k views
  newcomers      channels created <= 24 months ago with >= 100k median, and
                 whether they grow (median of their latest 10 uploads against
                 the 10 before)
Writes analysis/saturation.csv and analysis/saturation.json.
"""
import csv
import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import niche  # noqa: E402
from automation.niche_refine import NOT_NICHE  # noqa: E402

SEGMENTS = {
    "core: everyday life 1950s-60s": ["life in the 1950s america", "1950s america in color",
                                      "what life was like in the 1950s", "1950s housewife", "1960s america life"],
    "one city (NYC / LA / Chicago)": ["new york 1950s", "los angeles 1950s footage", "chicago 1950s footage"],
    "prices and daily life": ["1950s prices", "how much things cost in the 1950s", "cost of living 1950s",
                              "1950s grocery prices"],
    "then and now (same place)": ["then and now 1950s", "same place then and now footage", "1950s vs today"],
    "a decade: the 1970s": ["life in the 1970s america", "1970s america footage", "1970s new york footage"],
    "a profession / workplace": ["1950s jobs", "1950s factory workers footage", "1950s office life"],
}


def main():
    now = datetime.now(timezone.utc)
    after = (now - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%SZ")
    recent = now - timedelta(days=90)
    known = {c["channel_id"]: c for c in csv.DictReader(open(ROOT / "analysis" / "channels.csv", encoding="utf-8-sig"))}
    rows, out = [], {}
    for seg, queries in SEGMENTS.items():
        ids = set()
        for q in queries:
            try:
                ids |= set(niche.search(q, "viewCount", after=after))
            except Exception as e:                               # noqa: BLE001
                print(f"search {q!r} failed: {e}")
        vids = [v for v in niche.videos(ids).values()
                if niche.iso_dur(v["contentDetails"].get("duration")) >= 180
                and not NOT_NICHE.search(v["snippet"]["title"])
                and not v["snippet"]["channelTitle"].lower().endswith("vevo")]
        views = [int(v["statistics"].get("viewCount", 0)) for v in vids]
        pub = [datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00")) for v in vids]
        chan_ids = {v["snippet"]["channelId"] for v in vids}
        chans = niche.channels(chan_ids - set(known)) if chan_ids - set(known) else {}
        c100, newcomers = [], []
        for cid in chan_ids:
            if cid in known:
                k = known[cid]
                med, created = int(k["median_views_recent"] or 0), k["created"]
                ch = None
            else:
                ch = chans.get(cid)
                if not ch:
                    continue
                created = ch["snippet"]["publishedAt"][:10]
                ups = niche.recent_uploads(ch, 30)
                longs = [u for u in ups if niche.iso_dur(u["contentDetails"].get("duration")) >= 180]
                med = int(statistics.median([int(u["statistics"].get("viewCount", 0)) for u in longs])) if longs else 0
            name = next(v["snippet"]["channelTitle"] for v in vids if v["snippet"]["channelId"] == cid)
            if med >= 100_000:
                c100.append(name)
                age = (now - datetime.fromisoformat(created + "T00:00:00+00:00")).days
                if age <= 730:
                    if ch is None:
                        ch = niche.channels([cid]).get(cid)
                    ups = niche.recent_uploads(ch, 30) if ch else []
                    longs = sorted((u for u in ups if niche.iso_dur(u["contentDetails"].get("duration")) >= 180),
                                   key=lambda u: u["snippet"]["publishedAt"], reverse=True)
                    v_ = [int(u["statistics"].get("viewCount", 0)) for u in longs]
                    last10 = statistics.median(v_[:10]) if v_[:10] else 0
                    prev10 = statistics.median(v_[10:20]) if v_[10:20] else 0
                    newcomers.append({"channel": name, "created": created, "median": med,
                                      "last10_median": int(last10), "prev10_median": int(prev10),
                                      "growing": bool(prev10) and last10 > prev10})
        top = sorted(vids, key=lambda v: -int(v["statistics"].get("viewCount", 0)))[:5]
        out[seg] = {
            "videos_24m": len(vids),
            "per_month_now": round(sum(p >= recent for p in pub) / 3, 1),
            "median_views": int(statistics.median(views)) if views else 0,
            "share_100k": round(sum(x >= 100_000 for x in views) / len(views), 2) if views else 0,
            "channels": len(chan_ids), "channels_100k": len(c100), "channels_100k_names": sorted(c100)[:12],
            "newcomers_100k": newcomers,
            "top": [f"{int(v['statistics'].get('viewCount', 0)):,} - {v['snippet']['channelTitle']} - {v['snippet']['title'][:70]}" for v in top],
        }
        o = out[seg]
        rows.append({"segment": seg, **{k: o[k] for k in ("videos_24m", "per_month_now", "median_views", "share_100k",
                                                          "channels", "channels_100k")},
                     "newcomers_100k": len(newcomers), "newcomers_growing": sum(n["growing"] for n in newcomers)})
        print(f"{seg:34} videos {o['videos_24m']:3}  new/mo {o['per_month_now']:4}  median {o['median_views']:>9,}  "
              f">=100k {o['share_100k']:.0%}  ch>=100k {o['channels_100k']:2}  newcomers {len(newcomers)} "
              f"(growing {sum(n['growing'] for n in newcomers)})")
    with open(ROOT / "analysis" / "saturation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (ROOT / "analysis" / "saturation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
