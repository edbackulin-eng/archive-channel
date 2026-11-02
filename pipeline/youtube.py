"""YouTube for this channel: private uploads with a publishAt, never twice.

Token: YT_TOKEN_JSON (OAuth client of closed-tabs, as for the dinosaur channel),
from the environment only. Slots (config/channel.json "schedule"): Tuesday and
Friday 12:00 America/New_York, at least 30 minutes ahead, one video a day.

Anti-duplicate: before an upload the channel's own recent uploads are read; a
video with the same title is taken as already uploaded and its id returned. The
id is written to state/episodes.json the moment YouTube returns it.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from core.logger import get_logger

log = get_logger("youtube")
ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config" / "channel.json").read_text(encoding="utf-8"))["schedule"]
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def has_token():
    return bool(os.environ.get("YT_TOKEN_JSON", "").strip())


def service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    d = json.loads(os.environ["YT_TOKEN_JSON"])
    cred = Credentials(token=None, refresh_token=d["refresh_token"], client_id=d["client_id"],
                       client_secret=d["client_secret"], token_uri=d.get("token_uri", "https://oauth2.googleapis.com/token"),
                       scopes=d.get("scopes") or SCOPES)
    return build("youtube", "v3", credentials=cred, cache_discovery=False)


def next_slot(taken=(), now=None):
    """The nearest free Tue/Fri 12:00 New York slot, as an aware UTC datetime."""
    tz = ZoneInfo(CFG["timezone"])
    now = now or datetime.now(timezone.utc)
    hh, mm = (int(x) for x in CFG["local_time"].split(":"))
    days = {datetime.fromisoformat(t.replace("Z", "+00:00")).astimezone(tz).date() for t in taken}
    d = now.astimezone(tz).date()
    for _ in range(90):
        slot = datetime(d.year, d.month, d.day, hh, mm, tzinfo=tz)
        if slot.weekday() in CFG["weekdays"] and slot.astimezone(timezone.utc) - now >= timedelta(minutes=CFG["min_lead_minutes"]) \
                and d not in days:
            return slot.astimezone(timezone.utc)
        d += timedelta(days=1)
    raise RuntimeError("no free slot in 90 days")


def find_by_title(title, yt=None):
    yt = yt or service()
    ch = yt.channels().list(part="contentDetails", mine=True).execute()["items"][0]
    pl = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="snippet", playlistId=pl, maxResults=50).execute().get("items", [])
    for it in items:
        if it["snippet"]["title"].strip() == title.strip():
            return it["snippet"]["resourceId"]["videoId"]
    return None


def upload(video, title, description, tags, publish_at=None, yt=None):
    """Private, with publishAt when given (None: private, waiting for approval)."""
    from googleapiclient.http import MediaFileUpload
    yt = yt or service()
    same = find_by_title(title, yt)
    if same:
        log.warning("already on the channel as %s; not uploading again", same)
        return same
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if publish_at:
        status["publishAt"] = publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    body = {"snippet": {"title": title[:100], "description": description[:5000], "tags": tags[:30],
                        "categoryId": "27", "defaultLanguage": "en", "defaultAudioLanguage": "en"},
            "status": status}
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(str(video), chunksize=8 * 1024 * 1024, resumable=True))
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    log.info("uploaded %s", resp["id"])
    return resp["id"]


def set_thumbnail(video_id, path, yt=None):
    from googleapiclient.http import MediaFileUpload
    (yt or service()).thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(path))).execute()


def schedule(video_id, publish_at, yt=None):
    yt = yt or service()
    st = yt.videos().list(part="status", id=video_id).execute()["items"][0]["status"]
    keep = {k: v for k, v in st.items() if k in ("embeddable", "license", "publicStatsViewable",
                                                 "selfDeclaredMadeForKids", "containsSyntheticMedia")}
    keep.update({"privacyStatus": "private", "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")})
    yt.videos().update(part="status", body={"id": video_id, "status": keep}).execute()
