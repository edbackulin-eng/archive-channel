"""Print the next queued topic that has no episode yet and may render (empty when none).

Owner 17.11: a topic with a video, a build or a repeated subject never renders again - the rules
live in automation/topic_guard.py (the state, the queue's status, the same ground as a topic that
already has a video). A topic without its Wikipedia articles cannot be written (pipeline/sources.py)
and, owner 12.11, only a topic the archive holds film for (automation/material.py) is taken.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from automation import topic_guard  # noqa: E402

print(topic_guard.next_topic())
