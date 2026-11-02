"""Email the owner: a comment mentioning them on the open "run-log" issue.

    python scripts/notify.py "<subject>" "<body>"

GitHub mails every @mention by the Actions bot to the repository owner. One
issue collects them all; it is created on first use.
"""
import json
import os
import subprocess
import sys

REPO = os.environ.get("GITHUB_REPOSITORY", "edbackulin-eng/archive-channel")
OWNER = REPO.split("/")[0]


def gh(*args):
    r = subprocess.run(["gh", *args, "-R", REPO], capture_output=True, text=True)
    return r.stdout.strip()


def main(subject, body=""):
    issues = json.loads(gh("issue", "list", "--label", "run-log", "--state", "open", "--json", "number") or "[]")
    if issues:
        n = str(issues[0]["number"])
    else:
        gh("label", "create", "run-log", "--color", "b60205")
        n = gh("issue", "create", "--title", "Run log", "--label", "run-log",
               "--body", "A comment here mentioning the owner arrives as an email.").rsplit("/", 1)[-1]
    gh("issue", "comment", n, "--body", f"@{OWNER} **{subject}**\n\n{body}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
