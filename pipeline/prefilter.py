"""Does this frame belong under this line? A first answer on the CPU, before Gemini (owner 08.11).

The Gemini key's free day is the channel's narrowest point: 1950s-suburb spent it at slot 96
of 223. SigLIP (open_clip ViT-B-16-SigLIP, webli; Apache-2.0) scores every candidate frame
against its narration line on the runner's CPU:

    score >= accept   the frame is taken without Gemini ("checked", by "siglip")
    score <= reject   the frame is not taken and not asked about
    in between        doubtful: Gemini decides (pipeline/vision.py)

The two thresholds are set on Gemini's own answers (scripts/calibrate_prefilter.py, on the
"vision-log-<slug>" artifact of a build) and kept in config/channel.json "prefilter". With no
thresholds there (or no model), nothing is decided locally: every frame goes to Gemini, as before.

The model's weights (~800 MB) live in assets/models/open_clip, kept by actions/cache; text and
frame embeddings are cached in memory for the stage.
"""
import json
import os
from pathlib import Path

from core.logger import get_logger

log = get_logger("prefilter")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8")).get("prefilter") or {}
ARCH = CFG.get("arch", "ViT-B-16-SigLIP")
PRETRAINED = CFG.get("pretrained", "webli")
CACHE_DIR = ROOT / "assets" / "models" / "open_clip"
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "hf"))      # the tokenizer too, in the cached folder
PROMPT = "a frame from a {era} American colour film: {line}"
_m = {}
_img_emb, _txt_emb = {}, {}


def thresholds():
    """(accept, reject) or None when they have not been set."""
    a, r = CFG.get("accept"), CFG.get("reject")
    return (float(a), float(r)) if a is not None and r is not None else None


def _model():
    if "m" not in _m:
        import open_clip
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        model, _, pre = open_clip.create_model_and_transforms(ARCH, pretrained=PRETRAINED,
                                                              cache_dir=str(CACHE_DIR))
        model.eval()
        _m.update(m=model, pre=pre, tok=open_clip.get_tokenizer(ARCH), torch=torch)
        log.info("prefilter: %s (%s) loaded", ARCH, PRETRAINED)
    return _m


def available():
    try:
        _model()
        return True
    except Exception as e:                                           # noqa: BLE001
        log.warning("prefilter: no local model (%s); every frame goes to Gemini", e)
        return False


def _embed_images(paths):
    m = _model()
    todo = [p for p in dict.fromkeys(map(str, paths)) if p not in _img_emb]
    from PIL import Image
    for k in range(0, len(todo), 32):
        part = todo[k:k + 32]
        x = m["torch"].stack([m["pre"](Image.open(p).convert("RGB")) for p in part])
        with m["torch"].no_grad():
            e = m["m"].encode_image(x)
            e = e / e.norm(dim=-1, keepdim=True)
        for p, v in zip(part, e):
            _img_emb[p] = v


def _embed_texts(texts):
    m = _model()
    todo = [t for t in dict.fromkeys(texts) if t not in _txt_emb]
    for k in range(0, len(todo), 64):
        part = todo[k:k + 64]
        with m["torch"].no_grad():
            e = m["m"].encode_text(m["tok"](part))
            e = e / e.norm(dim=-1, keepdim=True)
        for t, v in zip(part, e):
            _txt_emb[t] = v


def text_of(line, era):
    return PROMPT.format(era=era or "mid-century", line=line.strip())


def scores(pairs):
    """pairs: [(line, era, keyframe path)] -> [cosine similarity of the frame and the line]."""
    if not pairs:
        return []
    texts = [text_of(l, e) for l, e, _ in pairs]
    _embed_texts(texts)
    _embed_images([kf for _, _, kf in pairs])
    return [float(_img_emb[str(kf)] @ _txt_emb[t]) for t, (_, _, kf) in zip(texts, pairs)]


def decide(score, th=None):
    """'accept', 'reject' or 'ask' for one score."""
    th = th or thresholds()
    if th is None:
        return "ask"
    if score >= th[0]:
        return "accept"
    if score <= th[1]:
        return "reject"
    return "ask"
