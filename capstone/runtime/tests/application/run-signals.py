#!/usr/bin/env python3
"""Run the signal contract (signal-contract.dom) in a provisioned guest, one mode
per case of docs/plans/delegation-signals.md, and report PASS or FAIL per mode
with the reason. Exit 1 if any mode fails; --report writes the details."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# mode -> (expected kind, expected value, stdout must contain)
MODES = {
    "self": ("exit", 0, "PASS"), "self-nodefer": ("exit", 0, "PASS"), "self-defer": ("exit", 0, "PASS"),
    "before-read": ("exit", 0, "PASS"),
    "during-read-restart": ("exit", 0, "PASS"), "during-read-eintr": ("exit", 0, "PASS"),
    "handler-write": ("exit", 0, "H\n"),
    "sigsuspend": ("exit", 0, "PASS"), "ppoll": ("exit", 0, "PASS"),
    "nest": ("exit", 0, "PASS"),
    "retry-partial": ("exit", 0, "PASS"),
    "wait-restart": ("exit", 0, "PASS"), "wait-eintr": ("exit", 0, "PASS"),
    "spawn-interrupted": ("exit", 0, "PASS"),
    "resethand-die": ("signal", 10, ""), "resethand-reinstall": ("exit", 0, "PASS"),
    "rt-queue": ("exit", 0, "PASS"),
    "hint": ("exit", 0, "PASS"),
    "altstack": ("exit", 0, "PASS"),
    "ign-inherit": ("exit", 0, "PASS"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--image", default="/mnt/host/signal-contract.dom")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]
    results, failed = {}, 0
    for mode, (kind, value, needle) in MODES.items():
        if args.only and mode not in args.only:
            continue
        status = args.state / f"signal-{mode}.json"
        status.unlink(missing_ok=True)
        try:
            result = subprocess.run([*cli, "run", "--result", str(status), args.image, mode],
                                    env=env, capture_output=True, text=True, timeout=args.timeout)
            record = json.loads(status.read_text()) if status.exists() else {}
            got = (record.get("kind"), record.get("value"))
            ok = got == (kind, value) and needle in result.stdout
            reason = "" if ok else f"got {got}, stdout={result.stdout!r}, stderr={result.stderr.strip()[-200:]!r}"
        except subprocess.TimeoutExpired:
            ok, reason, result = False, f"timeout after {args.timeout}s", None
        results[mode] = {"pass": ok, "reason": reason}
        failed += not ok
        print(f"{mode}: {'PASS' if ok else 'FAIL ' + reason}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=1) + "\n")
    print(f"signal contract: {len(results) - failed}/{len(results)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
