"""Fetch the models the runner needs: Kokoro (voice), YuNet (faces, for thumbnails) and
Real-ESRGAN's general model and its denoising twin (the thumbnail frame, on the CPU).

    python automation/fetch_models.py
"""
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
M = ROOT / "assets" / "models"
FILES = {
    "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # Real-ESRGAN (BSD-3-Clause), the official release
    "realesr-general-x4v3.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "realesr-general-wdn-x4v3.pth":
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
}


def main():
    M.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        p = M / name
        if p.exists() and p.stat().st_size > 100_000:
            continue
        # 08.11: a 504 from GitHub's release storage stopped 1950s-suburb's build; five tries, paced
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "archive-channel/1.0"})
                with urllib.request.urlopen(req, timeout=600) as r:
                    p.write_bytes(r.read())
                break
            except (urllib.error.URLError, OSError) as e:
                if attempt == 4:
                    raise
                print(f"{name}: {e}; again in {30 * (attempt + 1)} s")
                time.sleep(30 * (attempt + 1))
        print(name, p.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
