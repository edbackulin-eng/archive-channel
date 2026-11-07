"""The episode script, written from its sources and fact-checked against them.

Form (config/channel.json "episode"): 15-20 minutes running, of which 20-30 %
has no voice (film breaks, pipeline/breathing.py), so 1,800-2,250 words spoken
at ~150 words a minute, in English, in the delivery markup
(core/delivery.py). A documentary told through scenes and people: what a day
looked like, what things cost, what people were told and what they did, with
"open loops" - a question raised early and answered later - to carry the
viewer through.

The channel's angle (config/channel.json "niche", "prices"): prices and daily
life. Every chapter puts at least one price of its years on screen, with what
that is today; the writer marks them, pipeline/prices.py does the sums from
the BLS CPI-U, so the model never computes a "today" figure itself.

The first 30 seconds are the hook: the most vivid picture of the story, a
question, and a promise that there is more ("and that's not all").

Fact check: the writer returns, beside the script, every factual claim with the
source it came from. A second pass, by a separate call, holds each claim against
that source's text; a claim the text does not support is cut from the script
or softened, and the check is recorded in fact_check.json.
"""
import json
import re

from core import llm
from core.logger import get_logger

log = get_logger("scriptwriter")
# 12-15 minutes of speech at 150 wpm, so that with 20-30 % film breaks the
# episode lands at 15-20 minutes (config/channel.json "breathing")
WORDS_MIN, WORDS_MAX = 1800, 2250

SYSTEM = """You write narration for an English-language YouTube documentary channel about everyday
life in twentieth-century America, told over public-domain archive film. The viewer is
curious and nostalgic; the tone is warm, precise and unhurried, like a good museum guide.

RULES
- Write ONLY from the SOURCES given. Never add a fact, number, date, name or quote that the
  sources do not state. If a number is not in the sources, do not give one.
- Describe what the viewer would see in the period footage: rooms, streets, clothes, cars,
  shops, work. Short concrete sentences. Avoid lists of statistics.
- THE CHANNEL'S ANGLE IS PRICES AND DAILY LIFE. Prices, wages, rents and costs from the SOURCES
  run through the episode: one in the hook, at least 6 in all, never 6 minutes without one, each
  shown on screen as a price tag with what it is in today's money. Say what it meant for a family (hours of work, share of a
  week's pay, what it bought). Only prices the sources state, with the year they applied to.
- For each such price put a tag line of its own just BEFORE the line that says it:
    [price <amount> <year> | <what it is, 2-5 words>]      e.g. [price 7990 1949 | A new Levitt house]
  and when the narration gives today's equivalent, write [today <amount> <year>] in the line instead
  of a number, e.g. "That is [today 7990 1949] in today's money." NEVER write a today's-dollars
  figure yourself; the production fills it in from the US Bureau of Labor Statistics price index.
- Open loops: raise two or three questions early that are answered later, and say so
  ("we'll come back to that").
- People: no living private individuals by name; public figures and well-documented
  historical people only as the sources describe them.
- HOOK (the first ~30 seconds, 70-80 words): open on the single most vivid image of the story,
  ask the question the episode answers, and end the hook with a line that promises more
  (for example "And that's not all.").
- Length: {words_min}-{words_max} words of narration in total.
- Chapters: 5 to 7, each opening with a line "## Chapter title | place | years"
  (place and years as the chapter is set, e.g. "## The Potato Fields | Long Island, New York | 1947-1951").

DELIVERY MARKUP (one sentence or short thought per line)
  > line      faster: hook, transitions
  < line      slower, with a pause before: reveals and key facts
  [pause 0.6] an explicit pause in seconds
  [price 7990 1949 | A new Levitt house]   a price tag on screen (its own line)
  [today 7990 1949]                        today's equivalent, inside a spoken line
  plain line  normal pace

Return JSON only:
{{"title": "<YouTube title, <= 70 characters, curiosity without clickbait lies>",
  "thumbnail_words": "<2-4 words for the thumbnail, often a price from the script, e.g. $400 TO MOVE IN>",
  "thumbnail_subject": "<what the thumbnail picture must show so it matches those words: a house for a house's price, a grocery store for food prices, a family at home for a household cost>",
  "script": "<the whole script with chapters and markup, lines separated by \\n>",
  "claims": [{{"claim": "<one factual statement made in the script>", "source": <index of SOURCE>}}]}}"""

CHECK = """You are a strict fact checker. For each CLAIM you get the SOURCE text it is said to come from.
Answer for each: "supported" (the source states it), "partly" (the source states something close but
the claim overstates or adds detail), or "unsupported" (the source does not state it). For "partly",
give a corrected wording the source does support. Return JSON only:
{"results": [{"i": <index>, "verdict": "supported|partly|unsupported", "fix": "<corrected claim or empty>"}]}"""

REVISE = """You revise a documentary script after a fact check. For every claim marked unsupported, remove
it from the script (rewrite the sentence around it so the narration still flows). For every claim marked
partly, change the wording to the corrected version. Change nothing else. Keep the delivery markup and the
chapter lines exactly. Return JSON only: {"script": "<the revised script>"}"""


def _sources_block(sources):
    return "\n\n".join(f"SOURCE {i}: {s['title']} ({s['url']})\n{s['text']}" for i, s in enumerate(sources))


def words(script):
    body = [l for l in script.splitlines() if l.strip() and not l.startswith("##") and not l.startswith("[")]
    return sum(len(re.sub(r"^[<>]\s*", "", l).split()) for l in body)


def price_gaps(script, per_episode=6, max_gap_s=360, hook_s=36, wpm=150):
    """(tags, what is missing) for the channel's angle, on the script's estimated
    clock (words at wpm, plus pauses and film breaks): a tag in the hook, per_episode
    tags in all, and no stretch over max_gap_s without one."""
    from core import delivery
    t, at, chapter, where = 0.0, [], "", []
    for line in delivery.parse(script):
        t += line.film_before + line.pause_before
        chapter = line.chapter or chapter
        if line.price:
            at.append(t)
            where.append(chapter.split("|")[0].strip())
        t += len(line.text.split()) / wpm * 60
    missing = []
    if not any(a < hook_s for a in at):
        missing.append("a price in the hook")
    if len(at) < per_episode:
        missing.append(f"{len(at)} of {per_episode} price tags in the episode")
    for a, b in zip([0.0] + at, at + [t]):
        if b - a > max_gap_s:
            missing.append(f"no price from {a / 60:.1f} to {b / 60:.1f} min")
    return len(at), missing


def shorten(script, prompt=""):
    """The script cut to WORDS_MAX or fewer: lines dropped or condensed, nothing added, so
    every claim left is one the fact check already holds; the price tags, chapters and
    markup kept. 1950s-suburb (06.11) was 2,378 words and ran 20.2 minutes (QA: 15-20)."""
    n = words(script)
    if n <= WORDS_MAX:
        return script
    target = WORDS_MAX - 100
    for temp in (0.3, 0.5):
        more = llm.complete_json(
            prompt + f"\n\nYOUR SCRIPT ({n} words, too long):\n{script}\n\nCut it to {target}-{WORDS_MAX} "
            "words. Only delete or condense sentences; add no fact, name, number or line that is not already "
            "there. Keep the hook, every chapter heading, every [price ...] and [today ...] line with the line "
            "it belongs to, and all delivery markup. Return JSON: {\"script\": \"...\"}.",
            system=SYSTEM.format(words_min=WORDS_MIN, words_max=WORDS_MAX), max_tokens=16000, temperature=temp)
        cut = more.get("script") or ""
        if WORDS_MIN * 0.9 <= words(cut) <= WORDS_MAX and len(price_gaps(cut)[1]) <= len(price_gaps(script)[1]):
            log.info("script cut from %d to %d words", n, words(cut))
            return cut
    raise RuntimeError(f"LENGTH: the script is {n} words and could not be cut to {WORDS_MAX}")


def write(topic, sources):
    prompt = (f"EPISODE: {topic['name']}\nYEARS: {topic['years'][0]}-{topic['years'][1]}\n\n"
              f"{_sources_block(sources)}")
    out = llm.complete_json(prompt, system=SYSTEM.format(words_min=WORDS_MIN, words_max=WORDS_MAX),
                            max_tokens=16000, temperature=0.7)
    n = words(out["script"])
    if n < WORDS_MIN * 0.9:
        log.info("script is %d words; asking for the missing length", n)
        more = llm.complete_json(
            prompt + f"\n\nYOUR DRAFT ({n} words, too short):\n{out['script']}\n\nExpand it to "
            f"{WORDS_MIN}-{WORDS_MAX} words with more scenes from the SAME sources; keep the hook, the "
            "chapters and the markup. Return the same JSON shape, claims included.",
            system=SYSTEM.format(words_min=WORDS_MIN, words_max=WORDS_MAX), max_tokens=16000, temperature=0.6)
        if words(more.get("script", "")) > n:
            out = more
    tags, gaps = price_gaps(out["script"])
    if gaps:
        log.info("price tags: %d, missing: %s; asking for them", tags, gaps)
        more = llm.complete_json(
            prompt + f"\n\nYOUR DRAFT:\n{out['script']}\n\nThe channel's angle is prices and daily life. Missing: "
            f"{gaps}. Add, from the SAME sources only, prices, wages, rents or costs where they are missing "
            "(one in the hook, 6 or more in the episode, never 6 minutes without one), each with its "
            "[price <amount> <year> | <label>] line before the line that says it, and [today <amount> <year>] "
            "where the narration gives today's money. Change nothing else. Return the same JSON shape, claims included.",
            system=SYSTEM.format(words_min=WORDS_MIN, words_max=WORDS_MAX), max_tokens=16000, temperature=0.5)
        if more.get("script") and len(price_gaps(more["script"])[1]) < len(gaps):
            out = more
    if words(out["script"]) > WORDS_MAX:
        out["script"] = shorten(out["script"], prompt)
    return out


def fact_check(draft, sources):
    claims = [c for c in draft.get("claims", []) if isinstance(c, dict) and c.get("claim")]
    rows = []
    for k in range(0, len(claims), 20):
        part = claims[k:k + 20]
        ask = "\n\n".join(
            f"CLAIM {k + i}: {c['claim']}\nSOURCE: {sources[int(c.get('source', 0)) % len(sources)]['text'][:9000]}"
            for i, c in enumerate(part))
        res = llm.complete_json(ask, system=CHECK, max_tokens=4000, temperature=0.0)
        by = {int(r["i"]): r for r in res.get("results", [])}
        for i, c in enumerate(part):
            r = by.get(k + i, {"verdict": "unsupported", "fix": ""})
            rows.append({"claim": c["claim"], "source": sources[int(c.get("source", 0)) % len(sources)]["url"],
                         "verdict": r.get("verdict", "unsupported"), "fix": r.get("fix", "")})
    bad = [r for r in rows if r["verdict"] != "supported"]
    script = draft["script"]
    if bad:
        ask = "FACT CHECK:\n" + "\n".join(f"- [{r['verdict']}] {r['claim']}" + (f" -> {r['fix']}" if r["fix"] else "")
                                         for r in bad) + f"\n\nSCRIPT:\n{script}"
        script = llm.complete_json(ask, system=REVISE, max_tokens=16000, temperature=0.2)["script"]
    log.info("fact check: %d claims, %d supported, %d corrected or cut", len(rows),
             len(rows) - len(bad), len(bad))
    return script, rows
