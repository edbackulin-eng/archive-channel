"""Prices of then in today's dollars: the channel's angle, "prices and daily life".

Every episode puts what things cost in its years next to what that is now
(config/channel.json "prices"). The conversion is done here, never by the text
model: amount x CPI(today) / CPI(year), with the US Bureau of Labor Statistics
Consumer Price Index for All Urban Consumers (CPI-U, U.S. city average, all
items, not seasonally adjusted, 1982-84 = 100; series CUUR0000SA0).

data/cpi_u.json holds the annual averages: BLS's own (period M13) where the
API gives them, otherwise the mean of the year's twelve months, which is how
BLS computes them. "Today" is the last full year. fetch() refreshes the file
(BLS public API v1, ten years a request, no key).

In a script:
  [price 7990 1949 | A new Levitt house]   an on-screen price tag over the
                                           next line's picture
  [today 7990 1949]                        spoken: "about $107,000" in the
                                           last full year's dollars
"""
import json
import re
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "data" / "cpi_u.json"
SERIES = "CUUR0000SA0"
API = "https://api.bls.gov/publicAPI/v1/timeseries/data/"
SOURCE = f"U.S. Bureau of Labor Statistics, CPI-U (series {SERIES}), https://data.bls.gov/timeseries/{SERIES}"
PRICE = re.compile(r"\[price\s+([\d.,]+)\s+(\d{4})\s*(?:\|\s*([^\]]*))?\]")
TODAY = re.compile(r"\[today\s+([\d.,]+)\s+(\d{4})\]")


def fetch(first=1913, last=None):
    last = last or date.today().year
    old = json.loads(FILE.read_text(encoding="utf-8")) if FILE.exists() else {}
    monthly, annual = {}, {int(y): v for y, v in old.get("annual", {}).items()}
    # the ranges give months only; the plain request (last three years) also gives BLS's
    # own annual averages, the only ones for a year with a month not collected (Oct 2025)
    reqs = [urllib.request.Request(API + SERIES)]
    for a in range(first, last + 1, 10):
        body = json.dumps({"seriesid": [SERIES], "startyear": str(a), "endyear": str(min(a + 9, last))}).encode()
        reqs.append(urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"}))
    for req in reqs:
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
        if d.get("status") != "REQUEST_SUCCEEDED":
            raise RuntimeError(f"BLS: {d.get('status')} {d.get('message')}")
        for x in d["Results"]["series"][0]["data"]:
            if not re.fullmatch(r"[\d.]+", x["value"]):        # a month BLS did not collect ("-")
                continue
            y, p, v = int(x["year"]), x["period"], float(x["value"])
            if p == "M13":
                annual[y] = v
            elif p.startswith("M"):
                monthly.setdefault(y, {})[p] = v
    for y, ms in monthly.items():
        if y not in annual and len(ms) == 12:
            annual[y] = round(sum(ms.values()) / 12, 3)
    FILE.parent.mkdir(exist_ok=True)
    FILE.write_text(json.dumps({"series": SERIES, "source": SOURCE, "fetched": date.today().isoformat(),
                                "base": "1982-84=100", "annual": {str(y): annual[y] for y in sorted(annual)},
                                **({"notes": old["notes"]} if old.get("notes") else {})},
                               indent=1), encoding="utf-8")
    return FILE


def refresh(max_age_days=40):
    """A new year's average once a month at most; the file in the repository stays
    in use when BLS is down or its daily limit is spent."""
    try:
        age = (date.today() - date.fromisoformat(json.loads(FILE.read_text(encoding="utf-8"))["fetched"])).days
    except (OSError, ValueError, KeyError):
        age = 1 << 30
    if age < max_age_days:
        return False
    try:
        fetch(date.today().year - 2)
        return True
    except Exception:                                            # noqa: BLE001
        return False


def _cpi():
    return {int(y): v for y, v in json.loads(FILE.read_text(encoding="utf-8"))["annual"].items()}


def today_year():
    return max(_cpi())


def in_today(amount, year):
    cpi = _cpi()
    if int(year) not in cpi:
        raise KeyError(f"no CPI-U annual average for {year}")
    return float(amount) * cpi[today_year()] / cpi[int(year)]


def nice(x):
    """Rounded the way it is said: $13.20, $480, $9,700, $107,000, $1.2 million."""
    if x < 20:
        return f"${x:,.2f}"
    if x < 1000:
        return f"${round(x, -1 if x >= 200 else 0):,.0f}"
    if x < 1e6:
        step = 100 if x < 20000 else 1000
        return f"${round(x / step) * step:,.0f}"
    if x < 1e9:
        m = x / 1e6
        return f"${m:,.1f} million" if m < 10 else f"${round(m, 0 if m < 100 else -1):,.0f} million"
    return f"${x / 1e9:,.1f} billion"


def amount(text):
    return float(text.replace(",", ""))


def money(x):
    return f"${x:,.2f}" if x != int(x) else f"${x:,.0f}"


def spoken_today(text):
    """[today A Y] -> 'about $107,000'."""
    return TODAY.sub(lambda m: "about " + nice(in_today(amount(m.group(1)), m.group(2))), text)


def tag(a, year, label=""):
    """What the on-screen price tag says."""
    a = amount(a) if isinstance(a, str) else a
    return {"then": money(a), "year": str(year), "label": (label or "").strip(),
            "now": nice(in_today(a, year)), "now_year": str(today_year()),
            "note": f"in {today_year()} dollars - US BLS CPI-U"}
