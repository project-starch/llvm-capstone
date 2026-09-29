#!/usr/bin/env python3
"""Run the Probe A context probe (context-probe.dom) in a provisioned guest,
one mode per case of docs/plans/delegation-threads.md, and report PASS or FAIL
per mode with the reason. Exit 1 if any mode fails; --report writes details."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# mode -> (expected kind, expected value, stdout must contain, stdout must not
# contain, expected fault: (cause, symbol the faulting pc must equal) or None)
MODES = {
    "nested-enter": ("exit", 0, "PASS", None, None),
    "nested-preempt": ("exit", 0, "PASS", None, None),
    "nested-reenter": ("exit", 0, "PASS", None, None),
    "remint": ("exit", 0, "PASS", None, None),
    # Negative: a revoked seal reloads untagged (capstone-qemu, ISSUES Q-11), so
    # entering it must fault at the CALL itself: cause 24, unexpected operand.
    "revoked-call": ("signal", 11, "", "REACHED", (24, "__capstone_context_call_insn")),
    # Through the monitor: ADOPT, STEP, FORGET.
    "adopt-enter": ("exit", 0, "PASS", None, None),
    "adopt-thread": ("exit", 0, "PASS", None, None),
    "adopt-preempt": ("exit", 0, "PASS", None, None),
    "adopt-reenter": ("exit", 0, "PASS", None, None),
    "adopt-dead": ("exit", 0, "PASS", None, None),
}
LINK_BASE = 0x10000   # my_first_domain/link.ld


def fault_matches(record, expected, symbols):
    """The fault record names cause, pc and the code range; the pc's offset in
    the image must be the expected symbol's address."""
    if expected is None:
        return "fault" not in record, "unexpected fault record" if "fault" in record else ""
    cause, symbol = expected
    text = record.get("fault", "")
    fields = dict(word.split("=", 1) for word in text.split() if "=" in word)
    try:
        got_cause = int(fields["cause"])
        pc = int(fields["pc"], 16)
        code = int(fields["code"].split("-")[0], 16)
    except (KeyError, ValueError):
        return False, f"no parsable fault record: {text!r}"
    offset = pc - code + LINK_BASE
    want = symbols.get(symbol)
    ok = got_cause == cause and want is not None and offset == want
    return ok, "" if ok else f"fault cause={got_cause} offset={offset:#x}, want {cause} at {symbol}={want}"


def image_symbols(nm, image):
    if not nm or not image:
        return {}
    out = subprocess.run([nm, image], capture_output=True, text=True, check=True).stdout
    return {parts[2]: int(parts[0], 16) for parts in (line.split() for line in out.splitlines())
            if len(parts) == 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--image", default="/mnt/host/context-probe.dom")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--host-image", help="The same image on the host, for fault locations")
    parser.add_argument("--nm", help="llvm-nm, for fault locations")
    args = parser.parse_args()
    symbols = image_symbols(args.nm, args.host_image)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]
    results, failed = {}, 0
    for mode, (kind, value, needle, forbidden, fault) in MODES.items():
        if args.only and mode not in args.only:
            continue
        status = args.state / f"context-{mode}.json"
        status.unlink(missing_ok=True)
        try:
            result = subprocess.run([*cli, "run", "--result", str(status), args.image, mode],
                                    env=env, capture_output=True, text=True, timeout=args.timeout)
            record = json.loads(status.read_text()) if status.exists() else {}
            got = (record.get("kind"), record.get("value"))
            fault_ok, fault_reason = fault_matches(record, fault, symbols)
            if fault and not symbols:
                fault_ok, fault_reason = False, "fault location needs --host-image and --nm"
            ok = (got == (kind, value) and needle in result.stdout
                  and not (forbidden and forbidden in result.stdout) and fault_ok)
            reason = "" if ok else (f"got {got}, {fault_reason}, stdout={result.stdout!r}, "
                                    f"stderr={result.stderr.strip()[-300:]!r}")
            out = result.stdout
        except subprocess.TimeoutExpired:
            ok, reason, out = False, f"timeout after {args.timeout}s", ""
        results[mode] = {"pass": ok, "reason": reason, "stdout": out[-400:]}
        failed += not ok
        print(f"{mode}: {'PASS' if ok else 'FAIL ' + reason}", flush=True)
    passed = len(results) - failed
    print(f"context probe: {passed}/{len(results)} PASS")
    if args.report:
        args.report.write_text(json.dumps({"modes": results, "passed": passed,
                                           "total": len(results)}, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
