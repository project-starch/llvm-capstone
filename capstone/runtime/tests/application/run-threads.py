#!/usr/bin/env python3
"""Run the Probe B domain-phase probe (thread-probe.dom) in a provisioned guest,
one mode per case of docs/plans/delegation-threads.md, and report PASS or FAIL
per mode with the reason. Exit 1 if any mode fails; --report writes details."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("run_context", HERE / "run-context.py")
run_context = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_context)

# mode -> (expected kind, expected value, stdout must contain, stdout must not
# contain, expected fault: (cause, symbol the faulting pc must equal) or None)
MODES = {
    "transport": ("exit", 0, "PASS", None, None),
    "blocking": ("exit", 0, "PASS", None, None),
    "reserve": ("exit", 0, "PASS", None, None),
    "reuse": ("exit", 0, "PASS", None, None),
    "concurrent": ("exit", 0, "PASS", None, None),
    "signals-own": ("exit", 0, "PASS", None, None),
    "no-transport": ("exit", 0, "PASS", None, None),
    "preempted": ("exit", 0, "PASS", None, None),
    "many-rounds": ("exit", 0, "PASS", None, None),
    "signal-unblocked-context": ("exit", 0, "PASS", None, None),
    "sigpipe-ignored": ("exit", 0, "PASS", None, None),
    # Linux's default action for the writing thread's SIGPIPE ends the process.
    "sigpipe-child": ("signal", 13, "", "REACHED", None),
    # The exec'd image prints its launcher's blocked mask: nothing blocked.
    "exec-child": ("exit", 0, "SigBlk:\t0000000000000000", "REACHED", None),
    # T2: parking through the launcher.
    "futex-basic": ("exit", 0, "PASS", None, None),
    "futex-wake": ("exit", 0, "PASS", None, None),
    "futex-requeue": ("exit", 0, "PASS", None, None),
    "b6": ("exit", 0, "PASS", None, None),
    "b6-control": ("exit", 0, "PASS", None, None),
    "b12": ("exit", 0, "PASS", None, None),
    "b6-count": ("exit", 0, "PASS", None, None),
    "park-signal": ("exit", 0, "PASS", None, None),
    "park-signal-eintr": ("exit", 0, "PASS", None, None),
    "park-signal-timed": ("exit", 0, "PASS", None, None),
    # T3: runtime locks and identities (Q6, Q2's first part).
    "tid-identity": ("exit", 0, "PASS", None, None),
    "stdio-lines": ("exit", 0, "PASS", None, None),
    "heap-stress": ("exit", 0, "PASS", None, None),
    "b9": ("exit", 0, "PASS", None, None),
    "b14": ("exit", 0, "PASS", None, None),
    "stdio-nested": ("exit", 0, "PASS", None, None),
    "spawn-concurrent": ("exit", 0, "PASS", None, None),
    "mmap-concurrent": ("exit", 0, "PASS", None, None),
    "nested-create": ("exit", 0, "PASS", None, None),
    # exit() in a further context ends the process with its status.
    "exit-child": ("exit", 7, "", "REACHED", None),
    # A fault in a further context ends the process: a store through a null
    # capability, cause 24, at the labelled store.
    "fault-child": ("signal", 11, "", "REACHED", (24, "probe_fault_store_insn")),
}

# T4: musl's threads (pthread-probe.dom).
PTHREAD_MODES = {
    "join-values": ("exit", 0, "PASS", None, None),
    "join-parked": ("exit", 0, "PASS", None, None),
    "detach-many": ("exit", 0, "PASS", None, None),
    "reserve-waits": ("exit", 0, "PASS", None, None),
    # the last thread's return ends the application with 0 after main left
    "main-exit": ("exit", 0, "main-exit: PASS", None, None),
    "tsd": ("exit", 0, "PASS", None, None),
    "cond": ("exit", 0, "PASS", None, None),
    "clone-refused": ("exit", 0, "PASS", None, None),
    "thread-name": ("exit", 0, "PASS", None, None),
    "affinity": ("exit", 0, "PASS", None, None),
    "pi-mutex": ("exit", 0, "PASS", None, None),
    "user-stack": ("exit", 0, "PASS", None, None),
    "main-exit-more": ("exit", 0, "main-exit-more: PASS", None, None),
    # B8: signals per thread, and cancellation.
    "kill-thread": ("exit", 0, "PASS", None, None),
    "kill-thread-early": ("exit", 0, "PASS", None, None),
    "mask-routing": ("exit", 0, "PASS", None, None),
    "raise-thread": ("exit", 0, "PASS", None, None),
    "sigwait-thread": ("exit", 0, "PASS", None, None),
    "cancel-sem": ("exit", 0, "PASS", None, None),
    "cancel-read": ("exit", 0, "PASS", None, None),
    "cancel-disabled": ("exit", 0, "PASS", None, None),
    "park-signal-thread": ("exit", 0, "PASS", None, None),
    "cancel-installed-early": ("exit", 0, "PASS", None, None),
    "sigreturn-mask": ("exit", 0, "PASS", None, None),
    "setuid-threads": ("exit", 0, "PASS", None, None),
    "sigaction-race": ("exit", 0, "PASS", None, None),
    "main-exit-signal": ("exit", 0, "main-exit-signal: PASS", None, None),
    # abort in a thread ends the application with SIGABRT.
    "abort-thread": ("signal", 6, "", "REACHED", None),
    # exit() in a thread ends the application with its status.
    "exit-thread": ("exit", 5, "", "REACHED", None),
}
SUITES = {"thread": ("/mnt/host/thread-probe.dom", MODES),
          "pthread": ("/mnt/host/pthread-probe.dom", PTHREAD_MODES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(SUITES), default="thread")
    parser.add_argument("--image", help="default: the suite's probe on the share")
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--host-image", help="The same image on the host, for fault locations")
    parser.add_argument("--nm", help="llvm-nm, for fault locations")
    args = parser.parse_args()
    symbols = run_context.image_symbols(args.nm, args.host_image)
    env = dict(os.environ, PYTHONPATH=str(HERE.parents[1] / "host"))
    cli = [sys.executable, "-m", "capstone_vm", "--state", str(args.state)]
    image, modes = SUITES[args.suite]
    image = args.image or image
    results, failed = {}, 0
    for mode, (kind, value, needle, forbidden, fault) in modes.items():
        if args.only and mode not in args.only:
            continue
        status = args.state / f"threads-{mode}.json"
        status.unlink(missing_ok=True)
        try:
            result = subprocess.run([*cli, "run", "--result", str(status), image, mode],
                                    env=env, capture_output=True, text=True, timeout=args.timeout)
            record = json.loads(status.read_text()) if status.exists() else {}
            got = (record.get("kind"), record.get("value"))
            fault_ok, fault_reason = run_context.fault_matches(record, fault, symbols)
            if fault and not symbols:
                fault_ok, fault_reason = False, "fault location needs --host-image and --nm"
            ok = (got == (kind, value) and needle in result.stdout
                  and not (forbidden and forbidden in result.stdout + result.stderr) and fault_ok)
            reason = "" if ok else (f"got {got}, {fault_reason}, stdout={result.stdout!r}, "
                                    f"stderr={result.stderr.strip()[-300:]!r}")
            out = result.stdout
        except subprocess.TimeoutExpired:
            ok, reason, out = False, f"timeout after {args.timeout}s", ""
        results[mode] = {"pass": ok, "reason": reason, "stdout": out[-400:]}
        failed += not ok
        print(f"{mode}: {'PASS' if ok else 'FAIL ' + reason}", flush=True)
    passed = len(results) - failed
    print(f"{args.suite} probe: {passed}/{len(results)} PASS")
    if args.report:
        args.report.write_text(json.dumps({"modes": results, "passed": passed,
                                           "total": len(results)}, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
