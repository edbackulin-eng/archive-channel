"""Real-ESRGAN x4 on the CPU, for the thumbnail frame only (the video is never upscaled with it).

The model is the official realesr-general-x4v3 (SRVGGNetCompact, BSD-3-Clause,
github.com/xinntao/Real-ESRGAN, release v0.2.5.0), blended with its weak-denoise
twin realesr-general-wdn-x4v3 the way Real-ESRGAN's own inference does it
("denoise strength"): 1 removes the film grain hardest, 0 keeps it. A 640x480
frame takes some tens of seconds on a runner's CPU.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from core.logger import get_logger

log = get_logger("upscale")
M = Path(__file__).resolve().parent.parent / "assets" / "models"
GENERAL, WDN = M / "realesr-general-x4v3.pth", M / "realesr-general-wdn-x4v3.pth"
_net = {}


def _model(denoise):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class SRVGGNetCompact(nn.Module):
        """Real-ESRGAN's compact network (realesrgan/archs/srvgg_arch.py)."""

        def __init__(self, num_feat=64, num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            body = [nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            for _ in range(num_conv):
                body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
            body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
            self.body = nn.ModuleList(body)
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

    def params(path):
        d = torch.load(str(path), map_location="cpu", weights_only=True)
        return d.get("params_ema", d.get("params", d))

    a = params(GENERAL)
    if denoise < 1:
        b = params(WDN)
        a = {k: denoise * a[k] + (1 - denoise) * b[k] for k in a}
    net = SRVGGNetCompact()
    net.load_state_dict(a, strict=True)
    return net.eval()


def x4(im, denoise=0.5, tile=320):
    """A PIL image four times larger. Tiled with an overlap, so memory stays small."""
    import torch
    key = round(denoise, 2)
    if key not in _net:
        _net.clear()
        _net[key] = _model(denoise)
        torch.set_num_threads(max(1, torch.get_num_threads()))
    net = _net[key]
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    h, w = a.shape[:2]
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    pad = 12
    with torch.inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                ya, xa, yb, xb = max(0, y0 - pad), max(0, x0 - pad), min(h, y1 + pad), min(w, x1 + pad)
                t = torch.from_numpy(a[ya:yb, xa:xb].transpose(2, 0, 1).copy())[None]
                r = net(t)[0].clamp(0, 1).numpy().transpose(1, 2, 0)
                oy, ox = (y0 - ya) * 4, (x0 - xa) * 4
                out[y0 * 4:y1 * 4, x0 * 4:x1 * 4] = r[oy:oy + (y1 - y0) * 4, ox:ox + (x1 - x0) * 4]
    log.info("Real-ESRGAN x4: %dx%d -> %dx%d (denoise %.2f)", w, h, w * 4, h * 4, denoise)
    return Image.fromarray((out * 255 + 0.5).astype(np.uint8))
