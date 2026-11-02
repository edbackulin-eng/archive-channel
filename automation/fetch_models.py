"""Fetch the models the runner needs: Kokoro (voice) and YuNet (faces, for thumbnails).

    python automation/fetch_models.py
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "assets" / "models"
FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
}


def main():
    M.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        p = M / name
        if p.exists() and p.stat().st_size > 100_000:
            continue
        req = urllib.request.Request(url, headers={"User-Agent": "archive-channel/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r:
            p.write_bytes(r.read())
        print(name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
