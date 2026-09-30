#!/usr/bin/env python3
"""Run the socket contract (socket-contract.dom) in a provisioned guest, one mode
per case of docs/plans/delegation-sockets.md, and report PASS or FAIL per mode
with the reason. Exit 1 if any mode fails; --report writes the details."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# mode -> (expected kind, expected value, stdout must contain)
MODES = {
    "unix-stream": ("exit", 0, "PASS"), "unix-dgram": ("exit", 0, "PASS"),
    "inet-stream": ("exit", 0, "PASS"),
    # the image's 16 KiB region: the datagram rule must show, which Linux itself never does
    "inet-dgram": ("exit", 0, "40000-byte datagram: EMSGSIZE"),
    "scm-rights": ("exit", 0, "PASS"), "epoll": ("exit", 0, "PASS"),
    "select-poll": ("exit", 0, "PASS"), "nonblock": ("exit", 0, "PASS"),
    "inherit": ("exit", 0, "PASS"), "hosts": ("exit", 0, "PASS"),
    "tagged-buffer": ("exit", 0, "PASS"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--image", default="/mnt/host/socket-contract.dom")
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
        status = args.state / f"socket-{mode}.json"
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
        results[mode] = {"pass": ok, "reason": reason,
                         "stdout": result.stdout.strip() if result else ""}
        failed += not ok
        print(f"{mode}: {'PASS' if ok else 'FAIL ' + reason}")
        if ok and result and result.stdout.count("\n") > 1:
            for line in result.stdout.strip().splitlines()[:-1]:
                print(f"  {line}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=1) + "\n")
    print(f"socket contract: {len(results) - failed}/{len(results)} PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
