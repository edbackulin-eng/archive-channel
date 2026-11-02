"""Fetch the music library (assets/music/library.json) onto this machine or runner.

    python automation/fetch_music.py
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUSIC = ROOT / "assets" / "music"


def main():
    lib = json.loads((MUSIC / "library.json").read_text(encoding="utf-8"))
    bad = 0
    for t in lib["tracks"]:
        dest = MUSIC / t["file"]
        if dest.exists() and dest.stat().st_size > 200_000:
            continue
        for a in range(3):
            try:
                req = urllib.request.Request(t["url"], headers={"User-Agent": "archive-channel/1.0"})
                dest.write_bytes(urllib.request.urlopen(req, timeout=120).read())
                break
            except Exception as e:                                   # noqa: BLE001
                if a == 2:
                    print(f"{t['title']}: {e}")
                    bad += 1
                time.sleep(3)
        if dest.exists() and dest.stat().st_size < 200_000:
            print(f"{t['title']}: file too small")
            bad += 1
    print(f"{len(lib['tracks']) - bad} of {len(lib['tracks'])} tracks ready")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
