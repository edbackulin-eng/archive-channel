"""Text models for this channel: Groq first, Gemini flash-lite as the fallback.

Groq's model is one no other channel of the owner uses (config/channel.json
"text_llm"): the first of the preference list that Groq currently serves and
that is not taken elsewhere. Keys: GROQ_API_KEY_AR (or GROQ_API_KEY) and
GEMINI_API_KEY_AR, from the environment only - never printed.

On the owner's PC, with neither key present, the local `claude` CLI can stand
in (ARCHIVE_LOCAL_LLM=claude) so review drafts can be made; CI never uses it.
"""
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from core.logger import get_logger

log = get_logger("llm")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["text_llm"]
GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
UA = "archive-channel/1.0"


class LLMError(RuntimeError):
    pass


def _key(*names):
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    return ""


def groq_key():
    return _key("GROQ_API_KEY_AR", "GROQ_API_KEY")


def gemini_key():
    return _key("GEMINI_API_KEY_AR")


_groq_model = None


def groq_model():
    """The first preferred model Groq serves today, never one another channel uses."""
    global _groq_model
    if _groq_model:
        return _groq_model
    req = urllib.request.Request(f"{GROQ_URL}/models", headers={
        "Authorization": f"Bearer {groq_key()}", "User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        served = {m["id"] for m in json.loads(r.read())["data"]}
    taken = set(CFG["groq_taken_elsewhere"])
    for m in CFG["groq_preference"]:
        if m in served and m not in taken:
            _groq_model = m
            return m
    raise LLMError(f"none of {CFG['groq_preference']} is served by Groq")


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    a = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if a < 0:
        raise LLMError(f"no JSON in answer: {text[:200]}")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[a:])
    except json.JSONDecodeError:
        # a backslash that starts no JSON escape (07.11: 1950s-groceries' script, "Invalid \escape")
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text[a:])
        obj, _ = json.JSONDecoder().raw_decode(fixed)
    return obj


def _post(url, payload, headers, timeout=240):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _groq(prompt, system, max_tokens, temperature, as_json):
    model = groq_model()
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
               "messages": ([{"role": "system", "content": system}] if system else [])
               + [{"role": "user", "content": prompt}]}
    if model.startswith("openai/gpt-oss"):
        # a reasoning model: its thinking counts against max_tokens, so it gets room of its own
        payload.update({"reasoning_effort": "medium", "max_tokens": max_tokens + 3000})
    if as_json:
        payload["response_format"] = {"type": "json_object"}
    d = _post(f"{GROQ_URL}/chat/completions", payload, {"Authorization": f"Bearer {groq_key()}"})
    return d["choices"][0]["message"]["content"]


def _gemini(prompt, system, max_tokens, temperature, as_json):
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
               "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}
    if as_json:
        payload["generationConfig"]["responseMimeType"] = "application/json"
    d = _post(GEMINI_URL.format(model=CFG["gemini_fallback"]), payload, {"x-goog-api-key": gemini_key()})
    cand = (d.get("candidates") or [{}])[0]
    return "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))


def _claude_cli(prompt, system, max_tokens, temperature, as_json):
    full = (system + "\n\n" if system else "") + prompt
    if as_json:
        full += "\n\nAnswer with JSON only."
    r = subprocess.run(["claude", "-p"], input=full, capture_output=True, text=True,
                       encoding="utf-8", timeout=900, shell=os.name == "nt")
    if r.returncode != 0:
        raise LLMError(r.stderr[-300:])
    return r.stdout


def providers():
    out = []
    if groq_key():
        out.append(("groq", _groq))
    if gemini_key():
        out.append(("gemini", _gemini))
    if not out and os.environ.get("ARCHIVE_LOCAL_LLM") == "claude":
        out.append(("claude-cli", _claude_cli))
    return out


def complete(prompt, system="", max_tokens=4000, temperature=0.6, as_json=False, retries=4):
    errors = []
    for name, fn in providers():
        for attempt in range(retries):
            try:
                text = fn(prompt, system, max_tokens, temperature, as_json)
                if text and text.strip():
                    log.info("answered by %s", name if name != "groq" else f"groq:{groq_model()}")
                    return text
                errors.append(f"{name}: empty answer")
            except urllib.error.HTTPError as e:
                errors.append(f"{name}: HTTP {e.code}")
                if e.code not in (429, 500, 502, 503, 504):
                    break
                if e.code != 429 and attempt < retries - 1:
                    # 08.11: Gemini answered 503 three times in 24 s and 1950s-groceries stopped in its
                    # script; an overloaded model gets minutes, not seconds
                    time.sleep(30 * (attempt + 1))
                    continue
            except Exception as e:                                   # noqa: BLE001
                errors.append(f"{name}: {e}")
            time.sleep(4 * (attempt + 1))
    raise LLMError("no provider answered: " + "; ".join(errors[-6:]) if errors else "no provider configured")


def complete_json(prompt, system="", max_tokens=4000, temperature=0.4):
    return extract_json(complete(prompt, system, max_tokens, temperature, as_json=True))
