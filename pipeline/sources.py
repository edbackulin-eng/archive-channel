"""Open sources an episode is written from, and only from.

Every topic carries its Wikipedia articles (topics/queue.json "wikipedia");
each article's plain text and its references are fetched through the public
MediaWiki API. The script writer sees these texts and nothing else, and the
fact check holds every claim against them.
"""
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from core.logger import get_logger

log = get_logger("sources")
API = "https://en.wikipedia.org/w/api.php"
UA = {"User-Agent": "archive-channel/1.0 (documentary research)"}
MAX_CHARS = 22000
CACHE = Path(__file__).resolve().parent.parent / "cache" / "wikipedia"
MIN_GAP_S = 1.5
_last = [0.0]


def _get(url, attempts=6):
    """Wikipedia answers 429 to quick bursts; requests are spaced and its
    Retry-After is honoured."""
    for a in range(attempts):
        wait = MIN_GAP_S - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code != 429 or a == attempts - 1:
                raise
            delay = int(e.headers.get("Retry-After") or 0) or 10 * (a + 1)
            log.info("Wikipedia 429; waiting %ds", delay)
            time.sleep(min(delay, 120))
        except Exception as e:                                       # noqa: BLE001
            if a == attempts - 1:
                raise
            log.info("retry (%s)", e)
            time.sleep(3 * (a + 1))


def wikipedia(title):
    CACHE.mkdir(parents=True, exist_ok=True)
    cached = CACHE / (hashlib.sha1(title.encode()).hexdigest()[:16] + ".json")
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    q = urllib.parse.urlencode({"action": "query", "titles": title, "redirects": 1, "prop": "extracts|info",
                                "inprop": "url", "explaintext": 1, "format": "json", "formatversion": 2})
    page = _get(f"{API}?{q}")["query"]["pages"][0]
    if page.get("missing"):
        raise LookupError(f"no Wikipedia article {title!r}")
    text = page.get("extract", "")
    # the reference and link sections carry no facts to narrate
    text = re.split(r"\n== (See also|References|Notes|Further reading|External links|Bibliography) ==", text)[0]
    out = {"title": page["title"], "url": page["fullurl"], "text": text[:MAX_CHARS]}
    cached.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def gather(titles):
    out = []
    for t in titles:
        try:
            out.append(wikipedia(t))
        except LookupError as e:
            log.warning("%s", e)
    if not out:
        raise LookupError(f"none of {titles} could be read")
    return out
