"""Write one episode's script from its sources, fact-check it, and store it as text.

    python automation/write_episode.py 1950s-suburb

episodes/<slug>/script.txt     the narration in delivery markup, chapters included
episodes/<slug>/meta.json      title, thumbnail words, sources, word count
episodes/<slug>/fact_check.json  every claim, its source and the verdict
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import scriptwriter, sources  # noqa: E402

log = get_logger("write-episode")


def main(slug):
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    topic = next(t for t in q["topics"] if t["slug"] == slug)
    found = sources.gather(topic["wikipedia"])
    try:
        found += sources.bls_spending(tuple(topic["years"]))      # prices and daily life, from BLS
    except Exception as e:                                       # noqa: BLE001
        log.warning("BLS consumer spending report not read: %s", e)
    try:
        draft = scriptwriter.write(topic, found)
    except Exception as e:                                       # noqa: BLE001
        from automation import quota_hold
        if quota_hold.spent(e):
            h = quota_hold.set_(slug)
            log.warning("GEMINI_DAY_SPENT: no text provider can write %s today (%s); builds wait until %s",
                        slug, str(e)[:160], h["until"])
        raise
    script, checked = scriptwriter.fact_check(draft, found)
    # 08.11: the fact check cuts lines; a script it left short is expanded and checked again
    for _ in range(2):
        if scriptwriter.est_minutes(script) >= scriptwriter.MIN_EST_MIN                 and scriptwriter.words(script) >= scriptwriter.WORDS_MIN:
            break
        log.info("after the fact check: %d words, ~%.1f min; expanded again", scriptwriter.words(script),
                 scriptwriter.est_minutes(script))
        draft = scriptwriter.expand({**draft, "script": script}, scriptwriter.prompt_of(topic, found), rounds=3)
        script, checked = scriptwriter.fact_check(draft, found)
    # 09.11: expanded past the ceiling (2022 words narrated 20.7 min): cut by whole lines to fit
    if scriptwriter.est_minutes(script) > scriptwriter.MAX_EST_MIN:
        log.info("%d words, ~%.1f min: cut to %.1f", scriptwriter.words(script), scriptwriter.est_minutes(script),
                 scriptwriter.MAX_EST_MIN)
        script = scriptwriter.shorten_to(script, scriptwriter.MAX_EST_MIN)
    out = ROOT / "episodes" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / "script.txt").write_text(script.strip() + "\n", encoding="utf-8")
    (out / "fact_check.json").write_text(json.dumps(checked, indent=1, ensure_ascii=False), encoding="utf-8")
    meta = {"slug": slug, "name": topic["name"], "title": draft.get("title", topic["name"]),
            "thumbnail_words": draft.get("thumbnail_words", ""), "thumbnail_subject": draft.get("thumbnail_subject", ""),
            "words": scriptwriter.words(script),
            "minutes_at_150wpm": round(scriptwriter.words(script) / 150, 1),
            "sources": [{"title": s["title"], "url": s["url"]} for s in found],
            "fact_check": {"claims": len(checked),
                           "supported": sum(c["verdict"] == "supported" for c in checked)}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s: %d words (~%.1f min), %d claims checked", slug, meta["words"], meta["minutes_at_150wpm"],
             len(checked))
    print(json.dumps(meta, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
