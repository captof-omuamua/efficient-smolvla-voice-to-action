"""Split the generated sentences into dev and test. Prints counts only, never the sentences."""
import hashlib
import json
import random
from pathlib import Path

DATA = Path(__file__).parent / "data"
SEED = 20261110
DEV_PER_COMMAND = 3
DEV_OUT = {"near_miss": 10, "unrelated": 5}

raw = json.loads((DATA / "sentences_raw.json").read_text())
rng = random.Random(SEED)
dev, test = {"in_scope": [], "out_of_scope": []}, {"in_scope": [], "out_of_scope": []}

for command_id in sorted({s["command_id"] for s in raw["in_scope"]}):
    group = [s for s in raw["in_scope"] if s["command_id"] == command_id]
    rng.shuffle(group)
    dev["in_scope"] += group[:DEV_PER_COMMAND]
    test["in_scope"] += group[DEV_PER_COMMAND:]

for kind, n in DEV_OUT.items():
    group = [s for s in raw["out_of_scope"] if s["kind"] == kind]
    rng.shuffle(group)
    dev["out_of_scope"] += group[:n]
    test["out_of_scope"] += group[n:]

(DATA / "dev.json").write_text(json.dumps(dev, indent=1))
(DATA / "test.json").write_text(json.dumps(test, indent=1))
digest = hashlib.sha256((DATA / "test.json").read_bytes()).hexdigest()
print("dev:", len(dev["in_scope"]), "+", len(dev["out_of_scope"]), "| test:", len(test["in_scope"]), "+", len(test["out_of_scope"]))
print("test.json sha256:", digest)
