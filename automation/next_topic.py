"""Print the next queued topic that has no episode yet (empty when none)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
st = ROOT / "state" / "episodes.json"
done = {r["slug"] for r in json.loads(st.read_text(encoding="utf-8"))} if st.exists() else set()
# a topic without its Wikipedia articles cannot be written (pipeline/sources.py): skipped
# owner 12.11: and only a topic the archive holds film for (automation/material.py)
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
from automation import material  # noqa: E402
print(next((t["slug"] for t in q["topics"] if t["slug"] not in done and t.get("status", "queued") == "queued"
            and t.get("wikipedia") and material.usable(t, q)), ""))
