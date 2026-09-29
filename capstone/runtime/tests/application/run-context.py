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
import threading
import time

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
    "reenter-revoked": ("exit", 0, "PASS", None, None),
    "done-preempted": ("exit", 0, "PASS", None, None),
    "rollback-thread": ("exit", 0, "PASS", None, None),
    "duplicate-adopt": ("exit", 0, "PASS", None, None),
    "two-steppers": ("exit", 0, "PASS", None, None),
    "loan-preempt": ("exit", 0, "PASS", None, None),
    # Negative: the loan ended with the call; a kept copy reloads untagged.
    "loan-after-return": ("signal", 11, "", "REACHED", (24, "probe_store_insn")),
}
# A12 (Q3). ctl-wfi may hold the hart for good; it runs only when named with --only.
MODES.update({
    "ctl-csr": ("exit", 0, "PASS", None, None),
    "ctl-mret": ("exit", 0, "PASS", None, None),
    "ctl-priv": ("exit", 0, "PASS", None, None),
    "ctl-mie": ("exit", 0, "PASS", None, None),
    "ctl-priv-nested": ("signal", 11, "", "REACHED", (2, "__capstone_context_call_insn")),
})
# A2 (authority at entry) and A10 (Linux never forgets).
MODES.update({
    "entry-audit": ("exit", 0, "PASS", None, None),
    "entry-audit-control": ("exit", 0, "PASS", None, None),
    "entry-negative": ("exit", 0, "PASS", None, None),
    "ra-slot0": ("exit", 0, "PASS", None, None),
    "ra-slot16": ("exit", 0, "PASS", None, None),
    "ra-slot32": ("exit", 0, "PASS", None, None),
    "ra-gp": ("exit", 0, "PASS", None, None),
    "offer-keep": ("exit", 0, "PASS", None, None),
    "offer-replace": ("exit", 0, "PASS", None, None),
    "exhaust": ("exit", 0, "PASS", None, None),
    "exhaust-ended": ("exit", 0, "PASS", None, None),
})
EXPLICIT = {"ctl-wfi": ("exit", 0, "PASS", None, None)}
# A9 needs the test firmware whose slots start near the last generation, in a
# boot of its own: only with --only. gen-launch is launched GEN_LAUNCHES times
# and each launch's application id is read from the launcher.
EXPLICIT.update({"gen-exhaust": ("exit", 0, "PASS", None, None)})
# A7 needs capstone-qemu's node instruments: CAPSTONE_TEST_NODE_QUERY=1 at
# boot. Only with --only.
EXPLICIT.update({"aba": ("exit", 0, "PASS", None, None)})
GEN_LAUNCHES = 8
GEN_LAST = 0x7fffffff
# A10 across applications: this many `hold` processes at once, 8 slots each,
# more than the monitor's 32. Every one must pass, and all must have held at
# the same time, or the case never created the shortage it is about.
HOLDERS = 5
# Guest environment per mode.
ENV = {"rollback-thread": ["CAPSTONE_CONTEXT_TEST_THREAD_FAILS=1"]}
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


def run_holders(cli, env, args):
    """HOLDERS concurrent `hold` processes; each passes, and every one reached
    its hold before any was released (guest CLOCK_MONOTONIC, milliseconds)."""
    procs = []
    for i in range(HOLDERS):
        status = args.state / f"context-hold-{i}.json"
        status.unlink(missing_ok=True)
        procs.append((status, subprocess.Popen(
            [*cli, "run", "--result", str(status), args.image, "hold"],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
    held, released, problems, outs = [], [], [], []
    for i, (status, proc) in enumerate(procs):
        try:
            out, err = proc.communicate(timeout=args.timeout + 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            problems.append(f"holder {i}: timeout")
        record = json.loads(status.read_text()) if status.exists() else {}
        outs.append(out)
        got = (record.get("kind"), record.get("value"))
        if got != ("exit", 0) or "PASS" not in out:
            problems.append(f"holder {i}: got {got}, stdout={out!r}, stderr={err.strip()[-300:]!r}")
        for line in out.splitlines():
            if "hold: holding at " in line:
                held.append(int(line.rsplit(" ", 1)[1]))
            if "hold: released at " in line:
                released.append(int(line.rsplit(" ", 1)[1]))
    if not problems:
        if len(held) != HOLDERS or len(released) != HOLDERS:
            problems.append(f"missing hold marks: {len(held)} held, {len(released)} released")
        elif max(held) >= min(released):
            problems.append(f"holds did not overlap: last held {max(held)}, "
                            f"first released {min(released)}")
    if not problems:
        outs.append(f"all {HOLDERS} held together from {max(held)} to {min(released)}")
    return not problems, "; ".join(problems), "\n".join(outs)


def run_foreign(cli, env, args):
    """A13's foreign-owner request: a victim application publishes its context
    id (stdout) and its own id (the launcher's stats line); a foreign
    application names both in STEP and FORGET and must be refused; the victim
    then still steps and forgets its context."""
    status = args.state / "context-victim.json"
    status.unlink(missing_ok=True)
    victim = subprocess.Popen([*cli, "run", "-e", "CAPSTONE_DELEGATE_STATS=1", "--result",
                               str(status), args.image, "victim"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = {"out": [], "err": []}

    def pump(stream, into):
        for line in stream:
            into.append(line)

    pumps = [threading.Thread(target=pump, args=(victim.stdout, lines["out"]), daemon=True),
             threading.Thread(target=pump, args=(victim.stderr, lines["err"]), daemon=True)]
    for thread in pumps:
        thread.start()
    context = app = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and (context is None or app is None):
        for line in list(lines["out"]):
            if "victim: context " in line:
                context = line.rsplit(" ", 1)[1].strip()
        for line in list(lines["err"]):
            if "capstone-exec: domain id=" in line:
                app = line.rsplit("=", 1)[1].strip()
        time.sleep(0.1)
    problems, out = [], ""
    if context is None or app is None:
        problems.append(f"victim published no ids: out={lines['out']!r} err={lines['err']!r}")
    else:
        fstatus = args.state / "context-foreign.json"
        fstatus.unlink(missing_ok=True)
        result = subprocess.run([*cli, "run", "--result", str(fstatus), args.image, "foreign",
                                 context, app], env=env, capture_output=True, text=True,
                                timeout=args.timeout)
        record = json.loads(fstatus.read_text()) if fstatus.exists() else {}
        out = result.stdout
        if (record.get("kind"), record.get("value")) != ("exit", 0) or "PASS" not in result.stdout:
            problems.append(f"foreign: {record}, stdout={result.stdout!r}, "
                            f"stderr={result.stderr.strip()[-300:]!r}")
    try:
        victim.wait(timeout=args.timeout + 30)
    except subprocess.TimeoutExpired:
        victim.kill()
        victim.wait()
        problems.append("victim: timeout")
    for thread in pumps:
        thread.join(5)
    record = json.loads(status.read_text()) if status.exists() else {}
    victim_out = "".join(lines["out"])
    if (record.get("kind"), record.get("value")) != ("exit", 0) or "PASS" not in victim_out:
        problems.append(f"victim: {record}, stdout={victim_out!r}")
    return not problems, "; ".join(problems), victim_out + out


def run_gen_launches(cli, env, args):
    """GEN_LAUNCHES launches of one image in a row. Each passes; the launcher's
    application ids are new and positive, a slot's generations rise, a slot
    that had GEN_LAST is not used again, and at least one slot reached it with
    a launch after it (else the case never created its condition)."""
    ids, problems, outs = [], [], []
    for i in range(GEN_LAUNCHES):
        status = args.state / f"context-gen-launch-{i}.json"
        status.unlink(missing_ok=True)
        try:
            result = subprocess.run([*cli, "run", "-e", "CAPSTONE_DELEGATE_STATS=1", "--result",
                                     str(status), args.image, "gen-launch"],
                                    env=env, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            problems.append(f"launch {i}: timeout")
            break
        record = json.loads(status.read_text()) if status.exists() else {}
        got = (record.get("kind"), record.get("value"))
        found = [line.rsplit("=", 1)[1] for line in result.stderr.splitlines()
                 if "capstone-exec: domain id=" in line]
        if got != ("exit", 0) or "PASS" not in result.stdout or len(found) != 1:
            problems.append(f"launch {i}: got {got}, ids {found}, "
                            f"stderr={result.stderr.strip()[-300:]!r}")
            break
        ids.append(int(found[0], 16))
    outs.append("ids " + " ".join(hex(i) for i in ids))
    retired, last = set(), {}
    for n, value in enumerate(ids):
        slot, gen = value & 0xffffffff, value >> 32
        if value <= 0 or gen > GEN_LAST or value in ids[:n]:
            problems.append(f"id {value:#x} is not new, or outside the generation range")
        if slot in retired:
            problems.append(f"slot {slot} used again after its last generation")
        if slot in last and gen <= last[slot]:
            problems.append(f"slot {slot} generation {gen:#x} after {last[slot]:#x}")
        last[slot] = gen
        if gen == GEN_LAST:
            retired.add(slot)
    first = next((n for n, value in enumerate(ids) if value >> 32 == GEN_LAST), None)
    if not problems and (first is None or first == len(ids) - 1):
        problems.append("no slot reached the last generation with a launch after it")
    return not problems, "; ".join(problems), "\n".join(outs)


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
    selected = dict(MODES)
    if args.only:
        selected.update({m: e for m, e in EXPLICIT.items() if m in args.only})
    for mode, (kind, value, needle, forbidden, fault) in selected.items():
        if args.only and mode not in args.only:
            continue
        status = args.state / f"context-{mode}.json"
        status.unlink(missing_ok=True)
        try:
            extra = [word for value in ENV.get(mode, []) for word in ("-e", value)]
            result = subprocess.run([*cli, "run", *extra, "--result", str(status), args.image, mode],
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
    if args.only and "gen-launch" in args.only:
        ok, reason, out = run_gen_launches(cli, env, args)
        results["gen-launch"] = {"pass": ok, "reason": reason, "stdout": out[-800:]}
        failed += not ok
        print(f"gen-launch: {'PASS' if ok else 'FAIL ' + reason}", flush=True)
    if not args.only or "foreign" in args.only:
        ok, reason, out = run_foreign(cli, env, args)
        results["foreign"] = {"pass": ok, "reason": reason, "stdout": out[-800:]}
        failed += not ok
        print(f"foreign: {'PASS' if ok else 'FAIL ' + reason}", flush=True)
    if not args.only or "hold" in args.only:
        ok, reason, out = run_holders(cli, env, args)
        results["hold"] = {"pass": ok, "reason": reason, "stdout": out[-800:]}
        failed += not ok
        print(f"hold: {'PASS' if ok else 'FAIL ' + reason}", flush=True)
    passed = len(results) - failed
    print(f"context probe: {passed}/{len(results)} PASS")
    if args.report:
        args.report.write_text(json.dumps({"modes": results, "passed": passed,
                                           "total": len(results)}, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
