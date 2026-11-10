"""Score the router on a sentence set.

  python score_router.py dev router    # the three-level router
  python score_router.py dev llm       # baseline: language model for every sentence
  python score_router.py test router   # scored once, refuses to run twice
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from router import Router

ROOT = Path(__file__).parent
split, mode = sys.argv[1], sys.argv[2]
out_path = Path(os.environ.get("OUT_DIR", ROOT.parent / "runs")) / "router" / f"{split}_{mode}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
if split == "test" and out_path.exists():
    sys.exit(f"{out_path} exists: the test set is scored once")

data = json.loads((ROOT / "data" / f"{split}.json").read_text())
sentences = [(s["id"], s["text"], s["command_id"]) for s in data["in_scope"]]
sentences += [(s["id"], s["text"], None) for s in data["out_of_scope"]]

router = Router(use_rules=mode == "router")
for warm_up in ["put the bowl away somewhere", "please turn it on now", "do something useful"]:
    router.route(warm_up)  # first calls are slow, so they aren't timed
torch.cuda.reset_peak_memory_stats()

rows = []
for sentence_id, text, label in sentences:
    d = router.route(text)
    rows.append({"id": sentence_id, "text": text, "label": label, "pred": d.command_id, "level": d.level, "ms": d.ms})

can_do = [r for r in rows if r["label"] is not None]
cant_do = [r for r in rows if r["label"] is None]
times = np.array([r["ms"] for r in rows])
summary = {
    "split": split,
    "mode": mode,
    "right_pct": 100 * np.mean([r["pred"] == r["label"] for r in can_do]),
    "wrong_pct": 100 * np.mean([r["pred"] is not None and r["pred"] != r["label"] for r in can_do]),
    "wrongly_refused_pct": 100 * np.mean([r["pred"] is None for r in can_do]),
    "correct_refusal_pct": 100 * np.mean([r["pred"] is None for r in cant_do]),
    "no_llm_pct_of_can_do": 100 * np.mean([r["level"] != "llm" for r in can_do]),
    "levels": dict(Counter(r["level"] for r in rows)),
    "median_ms": float(np.median(times)),
    "p99_ms": float(np.percentile(times, 99)),
    "peak_gpu_mb": torch.cuda.max_memory_allocated() / 1e6,
}
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary, indent=1))
