"""Archive film from sources whose licence allows any use: public domain, CC0, US Government.

Searched:
  - Internet Archive, collection "prelinger" (Prelinger Archives: industrial,
    educational and home films, mostly 1930s-1970s);
  - Internet Archive, collection "FedFlix" (US Government films, including the
    National Archives' own uploads, ids gov.archives.*);
  - Internet Archive, collection "nasa", and the NASA Image and Video Library.
The Library of Congress site sits behind a bot challenge and is not used.

Every item is checked on its own, never by collection: its licenseurl must be
one of config/channel.json "licence_urls_allowed" (public domain mark, CC0,
the old CC public-domain dedication). An item with no licence, or any other
licence (CC BY-NC-SA turns up inside Prelinger), is skipped. Titles and
descriptions that name a forbidden source (British Pathe, Getty, Reuters, AP
...) are skipped too, since a re-upload of their film can carry a free licence
it does not have. Colour film is ranked first.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from core.logger import get_logger

log = get_logger("archive")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["sources"]
UA = {"User-Agent": "archive-channel/1.0 (documentary research; public-domain film only)"}
SEARCH = "https://archive.org/advancedsearch.php"
COLLECTIONS = ("prelinger", "FedFlix", "nasa")
# whole words; the short agency names ("AP") only in capitals, or "happy" would match
FORBIDDEN = re.compile(r"\b(" + "|".join(re.escape(f) for f in CFG["forbidden"] if len(f) > 3) + r")\b", re.I)
FORBIDDEN_CAPS = re.compile(r"\b(" + "|".join(re.escape(f) for f in CFG["forbidden"] if len(f) <= 3) + r")\b")
VIDEO_FORMATS = ("MPEG2", "h.264", "MPEG4", "h.264 HD", "512Kb MPEG4", "Ogg Video")


def _get(url, timeout=60, attempts=3):
    for a in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as e:                                       # noqa: BLE001
            if a == attempts - 1:
                raise
            log.info("retry %s (%s)", url[:80], e)
            time.sleep(3 * (a + 1))


def licence_ok(url):
    u = (url or "").lower().replace("https://", "").replace("http://", "")
    return any(u.startswith(ok) for ok in CFG["licence_urls_allowed"])


def licence_name(url):
    u = (url or "").lower()
    if "zero" in u:
        return "CC0 1.0 (public domain dedication)"
    if "mark" in u:
        return "Public Domain Mark 1.0"
    if "publicdomain" in u:
        return "Public domain"
    return url or "unknown"


def search(words, year_from=None, year_to=None, rows=50, colour_first=True):
    """Items matching `words`, each with a verified free licence."""
    q = f"({words}) AND mediatype:movies AND (" + " OR ".join(f"collection:{c}" for c in COLLECTIONS) + ")"
    if year_from or year_to:
        q += f" AND year:[{year_from or 1900} TO {year_to or 1999}]"
    params = [("q", q), ("rows", rows), ("output", "json"), ("sort[]", "downloads desc")]
    for f in ("identifier", "title", "year", "date", "color", "sound", "licenseurl", "description",
              "subject", "collection", "runtime", "downloads", "creator", "sponsor"):
        params.append(("fl[]", f))
    d = json.loads(_get(f"{SEARCH}?{urllib.parse.urlencode(params)}"))
    out = []
    for x in d["response"]["docs"]:
        text = " ".join(str(x.get(k, "")) for k in ("title", "description", "creator", "sponsor"))
        if not licence_ok(x.get("licenseurl")):
            continue
        if FORBIDDEN.search(text) or FORBIDDEN_CAPS.search(text):
            log.info("skipped %s: names a forbidden source", x["identifier"])
            continue
        colour = str(x.get("color", "")).lower()
        out.append({
            "id": x["identifier"], "title": x.get("title", ""), "year": _year(x),
            "colour": colour.startswith("c") or "color" in colour,
            "sound": str(x.get("sound", "")), "licence_url": x.get("licenseurl"),
            "licence": licence_name(x.get("licenseurl")),
            "collection": x.get("collection"), "downloads": x.get("downloads", 0),
            "description": re.sub("<[^>]+>", " ", str(x.get("description", "")))[:1200],
            "subject": x.get("subject"), "page": f"https://archive.org/details/{x['identifier']}",
            "creator": x.get("creator") or x.get("sponsor") or "",
        })
    if colour_first:
        out.sort(key=lambda r: (not r["colour"], -int(r["downloads"] or 0)))
    return out


def _year(x):
    for k in ("year", "date"):
        m = re.search(r"(19\d\d)", str(x.get(k, "")))
        if m:
            return int(m.group(1))
    m = re.search(r"\b(19[3-8]\d)\b", str(x.get("title", "")))
    return int(m.group(1)) if m else None


def files(identifier):
    """The item's metadata (re-checked licence) and its best video file."""
    d = json.loads(_get(f"https://archive.org/metadata/{identifier}"))
    meta = d.get("metadata", {})
    if not licence_ok(meta.get("licenseurl")):
        raise PermissionError(f"{identifier}: licence {meta.get('licenseurl')!r} is not free")
    vids = [f for f in d.get("files", []) if f.get("format") in VIDEO_FORMATS]
    if not vids:
        raise FileNotFoundError(f"{identifier}: no video file")

    def rank(f):
        return (int(f.get("height") or 0) * int(f.get("width") or 0),
                VIDEO_FORMATS.index(f["format"]) * -1)
    best = max(vids, key=rank)
    small = min(vids, key=lambda f: int(f.get("size") or 1 << 40))
    base = f"https://archive.org/download/{identifier}/"
    return {"meta": meta, "best": base + urllib.parse.quote(best["name"]),
            "best_size": (best.get("width"), best.get("height")),
            "proxy": base + urllib.parse.quote(small["name"])}


def credit(item):
    who = f" ({item['creator']})" if item.get("creator") else ""
    return f"{item['title']}{who}, {item.get('year') or 'n.d.'} - {item['licence']} - {item['page']}"
