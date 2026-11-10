"""Append one reading of NVML's cumulative GPU energy counter to a JSON-lines file.

Usage: python nvml_energy.py --out <file.jsonl> --mark start|end|idle_start|idle_end [--idle-seconds N]
With --idle-seconds, it instead measures N seconds of idle (GPU doing nothing) and appends both ends.
"""

import argparse
import json
import time
from datetime import datetime, timezone

import pynvml


def reading(handle, mark):
    return {"mark": mark, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "unix_s": time.time(),
            "energy_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
            "compute_pids": sorted(p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--mark", required=True)
    parser.add_argument("--idle-seconds", type=float, default=0)
    args = parser.parse_args()
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    rows = [reading(handle, args.mark)]
    if args.idle_seconds:
        time.sleep(args.idle_seconds)
        rows[0]["mark"] = f"{args.mark}_start"
        rows.append(reading(handle, f"{args.mark}_end"))
    with open(args.out, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
