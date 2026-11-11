#!/usr/bin/env python3
"""Validate complete copyable-only runtime qualification, including fault causes."""
import argparse
import json
import re
from pathlib import Path

POSITIVE = {
    "ok": "VIRTUAL_APPLICATION_OK",
    "vm": "VIRTUAL_VM_OK",
    "vm-large": "VIRTUAL_VM_LARGE_OK",
    "vm-mappings": "VIRTUAL_VM_MAPPINGS_OK 9000",
    "nested": "VIRTUAL_NESTED_OK",
    "threads": "VIRTUAL_THREADS_OK",
    "heap-threads": "VIRTUAL_HEAP_THREADS_OK",
    "map-churn": "VIRTUAL_MAP_CHURN_OK rounds=300 same_address=300",
}
NEGATIVE = {
    "stale": 25, "bounds": 28, "nested-stale": 25, "retired": 25,
    "vm-ro-write": 15, "vm-none-read": 13, "vm-guard": 13, "vm-padding": 28,
}

def check(text):
    text = text.replace("\r", "")
    rows = []
    for mode in [*POSITIVE, *NEGATIVE]:
        blocks = re.findall(r"COPYABLE_BEGIN:" + re.escape(mode) + r"\n(.*?)"
                            r"COPYABLE_RESULT:" + re.escape(mode) + r":(\d+)\n",
                            text, re.S)
        row = {"mode": mode, "pass": False}
        if len(blocks) == 1:
            output, status = blocks[0]
            causes = [int(x) for x in re.findall(r"domain fault cause=(\d+)", output)]
            row.update(status=int(status), causes=causes)
            row["pass"] = (int(status) == 0 and POSITIVE[mode] in output and not causes
                           if mode in POSITIVE else
                           int(status) == 139 and causes == [NEGATIVE[mode]])
        rows.append(row)
    for marker in ["COPYABLE_ABI_PASS old-grant-refused legacy-remap-refused",
                   "COPYABLE_TIMER_WAKE_PASS", "COPYABLE_SUITE_COMPLETE",
                   "COPYABLE_GATE_EXIT:0", "VIRTUAL_STAGED_DONE"]:
        rows.append({"check": marker, "pass": marker in text})
    rows.append({"check": "no-kernel-fault", "pass": not any(
        x in text for x in ["Oops", "Kernel panic", "VM_CONTRACT_FAIL"])})
    return {"pass": all(row["pass"] for row in rows), "checks": rows}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("serial", type=Path)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = check(a.serial.read_text(errors="replace"))
    report = json.dumps(result, indent=2) + "\n"
    if a.output:
        a.output.write_text(report)
    print(report, end="")
    raise SystemExit(0 if result["pass"] else 1)
