"""Print the next queued topic that has no episode yet (empty when none)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
st = ROOT / "state" / "episodes.json"
done = {r["slug"] for r in json.loads(st.read_text(encoding="utf-8"))} if st.exists() else set()
print(next((t["slug"] for t in q["topics"] if t["slug"] not in done and t.get("status", "queued") == "queued"), ""))
