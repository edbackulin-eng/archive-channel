"""Open sources an episode is written from, and only from.

Every topic carries its Wikipedia articles (topics/queue.json "wikipedia");
each article's plain text and its references are fetched through the public
MediaWiki API. For the channel's angle, prices and daily life, every episode
also gets the US Bureau of Labor Statistics' "100 Years of U.S. Consumer
Spending" (Report 991, 2006; a US Government work, public domain): its
chapter for the census year nearest the topic - family incomes, what a family
spent, retail food prices, wages (bls_spending). The script writer sees these
texts and nothing else, and the fact check holds every claim against them.
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


BLS_REPORT = "https://www.bls.gov/opub/100-years-of-u-s-consumer-spending.pdf"
BLS_PERIODS = ("1901", "1918-19", "1934-36", "1950", "1960-61", "1972-73", "1984-85", "1996-97", "2002-03")
BLS_DATA = Path(__file__).resolve().parent.parent / "data" / "bls_100_years.json"
UA = {"User-Agent": "Mozilla/5.0 (compatible; archive-channel/1.0; documentary research)"}


def _pdf_text(pdf):
    import shutil
    import subprocess
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, encoding="utf-8",
                              errors="replace", check=True).stdout
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(str(pdf)).pages)


def bls_chapters(pdf):
    """The report's narrative, period by period (tables and charts, which come out as
    columns of bare numbers, left out). Written to data/bls_100_years.json: bls.gov
    turns most scripted requests away, so the repository keeps the text."""
    text = _pdf_text(pdf)
    heads = list(re.finditer(r"\n(" + "|".join(re.escape(p) for p in BLS_PERIODS) + r")\n\nUnited States\n", text))
    chapters = {}
    for a, b in zip(heads, heads[1:] + [None]):
        body = text[a.end():b.start() if b else len(text)]
        lines = [l for l in body.splitlines() if len(l) > 60 and not re.fullmatch(r"[\d\s$.,%-]+", l)]
        chapters.setdefault(a.group(1), "\n".join(lines))
    BLS_DATA.parent.mkdir(exist_ok=True)
    BLS_DATA.write_text(json.dumps({"source": BLS_REPORT, "title": "100 Years of U.S. Consumer Spending: Data for "
                                    "the Nation, New York City, and Boston (BLS Report 991, 2006)",
                                    "licence": "US Government work, public domain", "chapters": chapters},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    return chapters


def bls_spending(years):
    """The report's chapters for the periods that fall within, or nearest to, years (a, b)."""
    chapters = json.loads(BLS_DATA.read_text(encoding="utf-8"))["chapters"]
    y0, y1 = years
    first = lambda p: int(p[:4])                                 # noqa: E731
    inside = [p for p in chapters if y0 <= first(p) <= y1]
    pick = inside or [min(chapters, key=lambda p: min(abs(first(p) - y0), abs(first(p) - y1)))]
    return [{"title": f"100 Years of U.S. Consumer Spending: {p} (U.S. Bureau of Labor Statistics)",
             "url": BLS_REPORT, "text": chapters[p][:MAX_CHARS]} for p in pick]


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
