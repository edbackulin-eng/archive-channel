"""Authorise the channel once, by hand, and store the token as a GitHub secret.

    python scripts/youtube_auth.py client_secret.json      # consent in the browser, show the channel
    python scripts/youtube_auth.py --check                 # which channel the stored token opens
    python scripts/youtube_auth.py --store                 # put the token in Secrets (YT_TOKEN_JSON)

The OAuth client is closed-tabs' (client_secret.json, git-ignored here). On the
consent screen pick the channel Yesterday in Color; check the name and id this
prints before --store. The token stays in youtube_token.json (git-ignored) and
in the secret, and is never printed.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import youtube  # noqa: E402

REPO = "edbackulin-eng/archive-channel"
TOKEN_FILE = ROOT / "youtube_token.json"


def authorize(client_secret):
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), youtube.SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    TOKEN_FILE.write_text(json.dumps({"refresh_token": creds.refresh_token, "client_id": creds.client_id,
                                      "client_secret": creds.client_secret, "token_uri": creds.token_uri,
                                      "scopes": youtube.SCOPES}), encoding="utf-8")


def show():
    os.environ["YT_TOKEN_JSON"] = TOKEN_FILE.read_text(encoding="utf-8")
    yt = youtube.service()
    ch = yt.channels().list(part="snippet,statistics", mine=True).execute()["items"][0]
    print(f"channel: {ch['snippet']['title']}")
    print(f"id:      {ch['id']}")
    print(f"handle:  {ch['snippet'].get('customUrl', '-')}")
    print(f"videos:  {ch['statistics'].get('videoCount')}")
    return ch


def main(argv):
    if "--store" in argv:
        subprocess.run(["gh", "secret", "set", "YT_TOKEN_JSON", "--repo", REPO],
                       input=TOKEN_FILE.read_text(encoding="utf-8"), text=True, check=True)
        print("YT_TOKEN_JSON stored in", REPO)
        return 0
    if "--check" in argv:
        show()
        return 0
    authorize(Path(argv[0]))
    show()
    print("If this is Yesterday in Color: python scripts/youtube_auth.py --store")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
