"""Saturation per sub-niche, on the cleaned niche set (analysis/niche_relevant.csv).

    python automation/saturation_refine.py

The raw per-segment searches of saturation.py also return off-topic hits
(Formula 1 karting under "then and now", Okinawa under "1970s"), so the
decision rests on the videos that passed niche_refine.py's filter, sorted into
segments by their titles. No API calls.
"""
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "analysis"
SEG = {
    "core: everyday life 1946-1969": (r"\b(19[45]\d|195\d|196\d|1950'?s|1960'?s|1940'?s|fifties|sixties)\b",
                                      r"\b(life|living|everyday|daily|housewife|home|suburb\w*|kitchen|family|families|"
                                      r"americana|nostalgia|people|town|school|shopping|was like)\b"),
    "one city (NYC/LA/Chicago/SF)": (r".", r"\b(new york|nyc|manhattan|brooklyn|los angeles|hollywood|chicago|san francisco)\b"),
    "prices and daily life": (r".", r"(\bprices?\b|\bcosts?\b|\bcheap\b|\$\d|\bwages?\b|\bsalary\b|\bearn\w*|\bspent\b|\bbudget\b|how much)"),
    "then and now (same place)": (r".", r"(then and now|now and then|\bvs\.? (today|now|20\d\d)|\b19\d\d vs\.? 20\d\d|today\b|"
                                         r"same (place|street)|\bevolution of\b)"),
    "a decade: the 1970s": (r"\b(197\d|1970'?s|seventies)\b", r"."),
    "a profession / workplace": (r".", r"\b(jobs?|workers?|factory|office|secretary|work(ing)? life|career|mill|miners?)\b"),
}


def main():
    now = datetime.now(timezone.utc)
    rows = list(csv.DictReader(open(A / "niche_relevant.csv", encoding="utf-8-sig")))
    chans = {c["channel_id"]: c for c in csv.DictReader(open(A / "channels.csv", encoding="utf-8-sig"))}
    out, table = {}, []
    for seg, (era, kw) in SEG.items():
        vs = [r for r in rows if re.search(era, r["title"], re.I) and re.search(kw, r["title"], re.I)]
        pub = [datetime.fromisoformat(r["published"] + "T00:00:00+00:00") for r in vs]
        last24 = [r for r, p in zip(vs, pub) if p >= now - timedelta(days=730)]
        last12 = [r for r, p in zip(vs, pub) if p >= now - timedelta(days=365)]
        views = [int(r["views"]) for r in vs]
        cids = {r["channel_id"] for r in vs}
        c100 = [chans[c] for c in cids if c in chans and int(chans[c]["median_views_recent"] or 0) >= 100_000]
        new = [c for c in c100 if int(c["age_days"]) <= 730]
        v24 = [int(r["views"]) for r in last24]
        o = {
            "videos": len(vs), "videos_last_24m": len(last24), "videos_last_12m": len(last12),
            "new_per_month_12m": round(len(last12) / 12, 1),
            "median_views": int(statistics.median(views)) if views else 0,
            "median_views_last_24m": int(statistics.median(v24)) if v24 else 0,
            "share_100k": round(sum(v >= 100_000 for v in views) / len(views), 2) if views else 0,
            "channels": len(cids), "channels_100k": len(c100),
            "newcomers_100k": [{"channel": c["channel"], "created": c["created"], "median": int(c["median_views_recent"]),
                                "every_days": c["days_between_uploads"]} for c in new],
            "top": [f"{int(r['views']):,} | {r['channel']} | {r['title'][:70]}" for r in sorted(vs, key=lambda r: -int(r['views']))[:4]],
        }
        out[seg] = o
        table.append({"segment": seg, **{k: o[k] for k in ("videos", "videos_last_24m", "new_per_month_12m",
                                                           "median_views", "median_views_last_24m", "share_100k",
                                                           "channels", "channels_100k")},
                      "newcomers_100k": len(new)})
        print(f"{seg:32} videos {o['videos']:3} (24m {o['videos_last_24m']:3}, {o['new_per_month_12m']}/mo)  "
              f"median {o['median_views']:>9,} (24m {o['median_views_last_24m']:>9,})  >=100k {o['share_100k']:.0%}  "
              f"ch {o['channels']:3} ch>=100k {o['channels_100k']:2} newcomers {len(new)}")
    with open(A / "saturation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    (A / "saturation.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
