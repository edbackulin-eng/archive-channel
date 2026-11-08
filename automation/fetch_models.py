"""Fetch the models the runner needs: Kokoro (voice), YuNet (faces, for thumbnails) and
Real-ESRGAN's general model and its denoising twin (the thumbnail frame, on the CPU).

    python automation/fetch_models.py

08.11: GitHub's release storage answered 504 for an hour and stopped 1950s-suburb's build
twice. Every file now has its SHA-256 (taken from the official release files downloaded on
02-04.11) and mirrors on Hugging Face after the official address; a download is kept only when
its hash is the official one, so a mirror can never hand in another file. The episode workflow
also keeps assets/models in actions/cache, so most builds download nothing.
"""
import hashlib
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "assets" / "models"
GH_KOKORO = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
HF_KOKORO = "https://huggingface.co/fastrtc/kokoro-onnx/resolve/main/"
GH_ESRGAN = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
FILES = {
    "kokoro-v1.0.onnx": ("7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
                         [GH_KOKORO + "kokoro-v1.0.onnx", HF_KOKORO + "kokoro-v1.0.onnx"]),
    "voices-v1.0.bin": ("bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
                        [GH_KOKORO + "voices-v1.0.bin", HF_KOKORO + "voices-v1.0.bin"]),
    "face_detection_yunet_2023mar.onnx": (
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        ["https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
         "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"]),
    # Real-ESRGAN (BSD-3-Clause), the official release, then mirrors of the same file
    "realesr-general-x4v3.pth": (
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
        [GH_ESRGAN + "realesr-general-x4v3.pth",
         "https://huggingface.co/jhj0517/realesr-general-x4v3/resolve/main/realesr-general-x4v3.pth",
         "https://huggingface.co/clavidk/realesr-general-x4v3/resolve/main/realesr-general-x4v3.pth"]),
    "realesr-general-wdn-x4v3.pth": (
        "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
        [GH_ESRGAN + "realesr-general-wdn-x4v3.pth",
         "https://huggingface.co/Mitroshenkov87/voxprint-mirror-realesr-general-wdn-x4v3/resolve/main/"
         "realesr-general-wdn-x4v3.pth"]),
}
ROUNDS = 4


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name, want, urls):
    p = M / name
    if p.exists() and sha256(p) == want:
        return
    for rnd in range(ROUNDS):
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "archive-channel/1.0"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    data = r.read()
            except (urllib.error.URLError, OSError) as e:
                print(f"{name}: {url}: {e}")
                continue
            if hashlib.sha256(data).hexdigest() != want:
                print(f"{name}: {url}: not the official file (hash differs); not kept")
                continue
            p.write_bytes(data)
            print(name, len(data), "from", url)
            return
        time.sleep(30 * (rnd + 1))
    raise SystemExit(f"{name}: no address gave the official file")


def main():
    M.mkdir(parents=True, exist_ok=True)
    for name, (want, urls) in FILES.items():
        fetch(name, want, urls)
    return 0


if __name__ == "__main__":
    sys.exit(main())
