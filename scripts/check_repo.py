"""Refuse a commit or push that carries a secret or a heavy file.

    python scripts/check_repo.py            # every tracked file (CI)
    python scripts/check_repo.py --staged   # what is about to be committed (pre-commit hook)

This repository is public. The What If channel came within one push of
publishing a Hugging Face token in a Kaggle job file and did publish 311 MB of
video before it was noticed. Both are caught here, in CI on every push and
locally before a commit.
"""
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
PATTERNS = {
    "Groq key": r"gsk_[A-Za-z0-9]{20,}",
    "Google API key": r"AIza[0-9A-Za-z_-]{30,}",
    "Google OAuth token": r"ya29\.[0-9A-Za-z_-]{20,}",
    "Google OAuth refresh token": r"1//0[0-9A-Za-z_-]{20,}",
    "Google client secret": r"GOCSPX-[0-9A-Za-z_-]{20,}",
    "Hugging Face token": r"hf_[A-Za-z0-9]{30,}",
    "GitHub token": r"gh[pousr]_[A-Za-z0-9]{30,}",
    "Anthropic / Claude token": r"sk-ant-[A-Za-z0-9_-]{20,}",
    "Kaggle key": r"\"key\"\s*:\s*\"[0-9a-f]{32}\"",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
SKIP = {"scripts/check_repo.py"}


def files(staged):
    args = (["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"] if staged
            else ["git", "ls-files"])
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and f not in SKIP]


def main():
    staged = "--staged" in sys.argv
    problems = []
    for name in files(staged):
        path = Path(name)
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{name}: {size / 1e6:.1f} MB, over the 5 MB limit (media belongs in .gitignore)")
            continue
        text = path.read_bytes().decode("utf-8", "ignore")
        for label, pattern in PATTERNS.items():
            if re.search(pattern, text):
                problems.append(f"{name}: looks like it contains a {label}")
    if problems:
        print("REFUSED:")
        for p in problems:
            print("  " + p)
        return 1
    print(f"ok: {len(files(staged))} file(s), no secrets, nothing over 5 MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
