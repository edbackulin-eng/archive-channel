"""Niche analysis: what archive-footage documentaries about 20th-century America win.

    python automation/niche.py                 (YT_API_KEY in the environment)

1. Searches YouTube (English) for the niche's own phrases, by view count and,
   separately, by relevance over the last two years.
2. Reads every hit's statistics, length and channel.
3. Per channel: creation date, subscribers, the median views of its latest
   uploads and how often it uploads. A "newcomer" is a channel created at most
   two years ago whose recent uploads have a median of at least 100k views.
4. Per video: colour or black-and-white (the colourfulness of its thumbnail,
   and the title), and whether it is narrated (title and description words).
5. Writes analysis/niche.csv (videos), analysis/channels.csv (channels) and
   analysis/thumbs_top30.jpg - the 30 most viewed thumbnails with their views
   and channel - and prints a summary for the conclusion.
"""
import csv
import io
import json
import os
import re
import statistics
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "analysis"
API = "https://www.googleapis.com/youtube/v3"

QUERIES = [
    "life in 1950s america", "1950s america in color", "1960s america footage",
    "1970s america footage", "new york 1970s", "new york 1980s footage",
    "america in color 1940s", "1940s america footage", "old footage restored",
    "restored footage america", "vintage america documentary", "what life was like in the 1930s",
    "1930s america footage", "great depression footage color", "kodachrome america footage",
    "1950s suburbia", "life in the 1960s america", "1970s los angeles footage",
    "america 1980s footage", "old new york footage color", "1950s diner footage",
    "prelinger archives", "1960s commercials america", "life in america 100 years ago",
]
NARRATION_WORDS = re.compile(r"\b(narrat\w*|documentary|explained|story of|history of|what life was like|"
                             r"how people lived|the truth about)\b", re.I)
SILENT_WORDS = re.compile(r"\b(no narration|no commentary|without narration|ambient|music only|"
                          r"silent footage|raw footage|sound added|asmr)\b", re.I)
COLOR_WORDS = re.compile(r"\b(in colou?r|colori[sz]ed|kodachrome|technicolor|restored colou?r|"
                         r"4k colou?r|colou?r footage|60 ?fps)\b", re.I)
BW_WORDS = re.compile(r"\b(black and white|b&w)\b", re.I)


def key():
    k = os.environ.get("YT_API_KEY", "").strip()
    if not k:
        env = Path(r"C:\prehistoric-channel\.env")
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("YT_API_KEY="):
                k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("YT_API_KEY missing")
    return k


def get(path, **params):
    params["key"] = key()
    url = f"{API}/{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def iso_dur(s):
    m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s or "")
    if not m:
        return 0
    d, h, mi, se = (int(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def search(q, order, after=None, pages=1):
    ids, token = [], None
    for _ in range(pages):
        p = {"part": "id", "q": q, "type": "video", "maxResults": 50, "order": order,
             "relevanceLanguage": "en"}
        if after:
            p["publishedAfter"] = after
        if token:
            p["pageToken"] = token
        r = get("search", **p)
        ids += [i["id"]["videoId"] for i in r.get("items", []) if i["id"].get("videoId")]
        token = r.get("nextPageToken")
        if not token:
            break
    return ids


def chunks(xs, n=50):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def videos(ids):
    out = {}
    for part in chunks(list(ids)):
        r = get("videos", part="snippet,statistics,contentDetails", id=",".join(part), maxResults=50)
        for v in r.get("items", []):
            out[v["id"]] = v
    return out


def channels(ids):
    out = {}
    for part in chunks(list(ids)):
        r = get("channels", part="snippet,statistics,contentDetails", id=",".join(part), maxResults=50)
        for c in r.get("items", []):
            out[c["id"]] = c
    return out


def recent_uploads(ch, n=30):
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    r = get("playlistItems", part="contentDetails", playlistId=pl, maxResults=min(n, 50))
    ids = [i["contentDetails"]["videoId"] for i in r.get("items", [])]
    return list(videos(ids).values())


def thumb(v):
    t = v["snippet"]["thumbnails"]
    return (t.get("maxres") or t.get("high") or t.get("medium") or t["default"])["url"]


def fetch_image(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")


def colourfulness(im):
    """Hasler-Suesstrunk colourfulness; under ~15 reads as black and white."""
    a = np.asarray(im.resize((320, 180)), np.float32)
    rg = a[..., 0] - a[..., 1]
    yb = 0.5 * (a[..., 0] + a[..., 1]) - a[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def collage(rows, path):
    W, H, cols = 384, 216, 6
    nrows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, (H + 46) * nrows), (18, 18, 18))
    d = ImageDraw.Draw(sheet)
    try:
        f = ImageFont.truetype("arial.ttf", 17)
        fb = ImageFont.truetype("arialbd.ttf", 18)
    except OSError:
        f = fb = ImageFont.load_default()
    for k, r in enumerate(rows):
        x, y = (k % cols) * W, (k // cols) * (H + 46)
        try:
            sheet.paste(fetch_image(r["thumb"]).resize((W, H)), (x, y))
        except Exception:                                        # noqa: BLE001
            pass
        d.text((x + 6, y + H + 3), f"#{k + 1}  {r['views'] / 1e6:.1f}M views  {r['dur_min']:.0f} min",
               font=fb, fill=(255, 220, 90))
        d.text((x + 6, y + H + 24), r["channel"][:40], font=f, fill=(220, 220, 220))
    sheet.save(path, quality=88)


def main():
    OUT.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc)
    after = (now - timedelta(days=730)).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = {}
    for q in QUERIES:
        for vid in search(q, "viewCount"):
            found.setdefault(vid, set()).add(q)
        for vid in search(q, "relevance", after=after):
            found.setdefault(vid, set()).add(q + " [2y]")
    vids = videos(found)
    chans = channels({v["snippet"]["channelId"] for v in vids.values()})

    rows = []
    for vid, v in vids.items():
        dur = iso_dur(v["contentDetails"].get("duration"))
        if dur < 180:                          # Shorts and clips are another business
            continue
        sn, st = v["snippet"], v["statistics"]
        text = sn["title"] + " " + sn.get("description", "")[:600]
        rows.append({
            "video_id": vid, "title": sn["title"], "channel": sn["channelTitle"],
            "channel_id": sn["channelId"], "views": int(st.get("viewCount", 0)),
            "likes": int(st.get("likeCount", 0) or 0), "published": sn["publishedAt"][:10],
            "dur_min": round(dur / 60, 1),
            "narrated": "no" if SILENT_WORDS.search(text) else ("yes" if NARRATION_WORDS.search(text) else "?"),
            "colour_title": "colour" if COLOR_WORDS.search(sn["title"]) else ("bw" if BW_WORDS.search(sn["title"]) else ""),
            "lang": sn.get("defaultAudioLanguage") or sn.get("defaultLanguage") or "",
            "found_by": "; ".join(sorted(found[vid])), "thumb": thumb(v),
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    rows = [r for r in rows if not r["lang"] or r["lang"].startswith("en")]
    rows.sort(key=lambda r: -r["views"])

    # colour from the thumbnail itself, for the top 120
    for r in rows[:120]:
        try:
            r["colourfulness"] = round(colourfulness(fetch_image(r["thumb"])), 1)
        except Exception:                                        # noqa: BLE001
            r["colourfulness"] = ""
    for r in rows:
        c = r.get("colourfulness")
        r["colour"] = r["colour_title"] or ("" if c in ("", None) else ("colour" if c >= 15 else "bw"))

    # channels: age, median views of the latest uploads, upload rhythm
    crow = []
    for cid, c in chans.items():
        created = c["snippet"]["publishedAt"][:10]
        age_d = (now - datetime.fromisoformat(created + "T00:00:00+00:00")).days
        try:
            ups = recent_uploads(c)
        except Exception:                                        # noqa: BLE001
            ups = []
        longs = [u for u in ups if iso_dur(u["contentDetails"].get("duration")) >= 180]
        views = [int(u["statistics"].get("viewCount", 0)) for u in longs]
        dates = sorted(datetime.fromisoformat(u["snippet"]["publishedAt"].replace("Z", "+00:00")) for u in longs)
        gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
        crow.append({
            "channel": c["snippet"]["title"], "channel_id": cid, "created": created, "age_days": age_d,
            "subs": int(c["statistics"].get("subscriberCount", 0) or 0),
            "videos": int(c["statistics"].get("videoCount", 0) or 0),
            "total_views": int(c["statistics"].get("viewCount", 0) or 0),
            "recent_long_uploads": len(longs),
            "median_views_recent": int(statistics.median(views)) if views else 0,
            "median_dur_min": round(statistics.median([iso_dur(u["contentDetails"]["duration"]) / 60 for u in longs]), 1) if longs else 0,
            "days_between_uploads": round(statistics.median(gaps), 1) if gaps else "",
            "newcomer": age_d <= 730 and (int(statistics.median(views)) if views else 0) >= 100_000,
            "in_niche_hits": sum(1 for r in rows if r["channel_id"] == cid),
        })
    crow.sort(key=lambda c: -c["median_views_recent"])

    cols = ["video_id", "title", "channel", "channel_id", "views", "likes", "published", "dur_min",
            "colour", "colourfulness", "narrated", "lang", "found_by", "url", "thumb"]
    with open(OUT / "niche.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "channels.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(crow[0].keys()))
        w.writeheader()
        w.writerows(crow)
    collage(rows[:30], OUT / "thumbs_top30.jpg")

    top = rows[:100]
    summary = {
        "videos": len(rows), "channels": len(crow),
        "top100_median_views": int(statistics.median(r["views"] for r in top)),
        "top100_median_minutes": statistics.median(r["dur_min"] for r in top),
        "top100_colour_share": round(sum(r["colour"] == "colour" for r in top) / len(top), 2),
        "top100_narrated": {k: sum(r["narrated"] == k for r in top) for k in ("yes", "no", "?")},
        "newcomers": [c for c in crow if c["newcomer"]][:15],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "newcomers"}, indent=1))
    print(f"{len(summary['newcomers'])} newcomer channel(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
