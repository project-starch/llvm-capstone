#!/usr/bin/env python3
"""Host test of compare.py: every way a launch can fail to happen must give INCOMPLETE (exit 2).

Synthetic run-arm.sh outputs, one gate at a time. The clean pair must compare IDENTICAL (exit 0),
so each INCOMPLETE below is the defect it plants and not a broken fixture.
  python3 capstone/runtime/virtual/gates/test_compare.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "compare.py"
sys.path.insert(0, str(HERE))
import compare  # noqa: E402

RUNTIME_ROWS = [r.split(":", 1)[1] for r in compare.REQUIRED_ROWS["runtime"]]
OUTPUT = {"sqlite_write": "SQLITE_SUM=5050", "sqlite_read": "SQLITE_PERSIST=100",
          "mruby": "MRUBY_SUM=5050", "perl": "PERL_VIRTUAL_OK sum=5050 file=virtual"}
CASES = ["pthread", "normal1", "normal2", "usr1", "marker", "fx1"]


def runtime(g, edit=None):
    lines = []
    for n in RUNTIME_ROWS:
        lines.append(f"VM_EXIT:{n}:0")
        if n in OUTPUT:
            lines.append(OUTPUT[n])
    if edit:
        lines = edit(lines)
    (g / "runtime.log").write_text("\n".join(lines) + "\nVIRTUAL_RUNTIME_DONE\n")
    tests = {n: True for n in RUNTIME_ROWS}
    tests.update(sqlite_results=True, mruby_results=True, perl_output=True)
    (g / "runtime.json").write_text(json.dumps({"status": "PASS", "tests": tests}))


def fp(g, serial="FP_EXIT:0\n"):
    (g / "fp.log").write_text(json.dumps({"status": "PASS", "tests": {"exit": True}}) + "\n")
    (g / "fp/guest").mkdir(parents=True)
    (g / "fp/guest/serial.log").write_text(serial)


# What the fixture-24 case directory holds in each recorded outcome (memcached rerun evidence): the header-only
# condition not created (known no-op mark 18e0018, exit 24, no fault) and created followed by the server's
# death at cause 24, address 0.
NOT_CREATED = dict(job='{"version":1,"kind":"exit","value":24}', fault="",
                   out="MCAPP-FIX 24 begin\nMCAPP-FIX 24 header-only=0\n"
                       "MCAPP-FIX 24 triggering condition not created (header-only branch not observed)\n"
                       "MCAPP-FIX 24 mark=18e0018\n")
CREATED_CRASH = dict(job='{"version":1,"kind":"signal","value":11}',
                     fault="capstone-exec: domain fault cause=24 pc=0x3fad02b134 address=0x0 ra=0x3fad02b0c4 image=./safety.dom\n",
                     out="MCAPP-FIX 24 begin\nMCAPP-FIX 24 header-only=1\nMCAPP-FIX 24 target=3f9eafd73e\nMCAPP-FIX 24 touch\n")
IA_NOT_LINKED = dict(NOT_CREATED, out="MCAPP-FIX 24 begin\nMCAPP-FIX 24 ia not linked\nMCAPP-FIX 24 mark=18e0018\n")
# the reviewer's substitution: an unrelated crash, no fault record, no fixture path in the output
OTHER_CRASH = dict(job='{"version":1,"kind":"signal","value":9}', fault="", out="")
FX24_PREDICTION_FAILS, FX24_TRIGGER_FAILS = ("FAIL", "PASS"), ("PASS", "FAIL")


def memcached(g, drop_job=None, skip=None, retry=None, fx24=None, fx24_case=None):
    """fx24: (prediction, trigger) statuses of the fixture that cannot pass as written; fx24_case: the
    files of its case directory (default: the recorded outcome that belongs to the failed check)."""
    serial, log = [], []
    cases = CASES + (["fx24"] if fx24 else [])
    for c in cases:
        if c == skip:
            continue
        if c == retry:   # a first attempt the launcher refused, then one that ran
            serial += [f"MC_BEGIN:{c}", "capstone-vexec: image: Exec format error", f"MC_END:{c}:126"]
        serial += [f"MC_BEGIN:{c}", f"MC_END:{c}:0"]
        if c != "pthread":
            (g / "memcached" / c).mkdir(parents=True)
            if c != drop_job:
                (g / "memcached" / c / "job.json").write_text('{"version":1,"kind":"exit","value":0}')
            (g / "memcached" / c / "server.err").write_text("")
    for c in cases:   # the runner checks every case it planned, run or not
        if c == "fx24":
            log += ["PASS fx24_harness", f"{fx24[0]} fx24_prediction", f"{fx24[1]} fx24_trigger"]
        else:
            log += ["PASS pthread_timed_wait"] if c == "pthread" else [f"PASS {c}_harness", f"PASS {c}_protocol"]
    if fx24:
        case = fx24_case or (CREATED_CRASH if fx24[0] == "FAIL" else NOT_CREATED)
        (g / "memcached/fx24/job.json").write_text(case["job"])
        (g / "memcached/fx24/server.out").write_text(case["out"])
        if case["fault"]:
            (g / "memcached/fx24/fault.txt").write_text(case["fault"])
    (g / "memcached/guest").mkdir(parents=True, exist_ok=True)
    (g / "memcached/guest/serial.log").write_text("\n".join(serial) + "\n")
    (g / "memcached.log").write_text("\n".join(log) + "\n")


def stamp(out, gate, mode, line, count=None, verified=True, summary=None, fast=None):
    """What run-arm.sh and the runners leave: OUT/mode, `count` memory-mode lines (default: one per module load
    of the gate) in the gate's guest console, the runner's memory_mode_verified line (True: PASS, False: FAIL,
    None: absent) and, when given, SUMMARY.txt. fast = (word, line[, count]) is OUT/fast-fault and `count` fault-path lines."""
    g = out / "gates"
    if mode:
        (out / "mode").write_text(mode + "\n")
    if summary is not None:
        (g / "SUMMARY.txt").write_text(summary)
    if line:
        n = compare.MODE_LOADS[gate] if count is None else count
        serials = sorted(g.glob(f"{gate}/**/serial.log"))
        target = g / "runtime.log" if gate == "runtime" else serials[0] if serials else g / f"{gate}.log"
        target.write_text(target.read_text() + (line + "\n") * n)
    if fast:
        (out / "fast-fault").write_text(fast[0] + "\n")
        n = compare.MODE_LOADS[gate] if len(fast) < 3 else fast[2]
        serials = sorted(g.glob(f"{gate}/**/serial.log"))
        target = g / "runtime.log" if gate == "runtime" else serials[0] if serials else g / f"{gate}.log"
        target.write_text(target.read_text() + (fast[1] + "\n") * n)
    if verified is not None:
        log = g / f"{gate}.log"
        log.write_text(log.read_text() + ("PASS" if verified else "FAIL") + " memory_mode_verified\n")


def run(gate, make_ctl, make_arm, ctl_mode=("off", "TEST_MEMORY_MODE:N"), arm_mode=("off", "TEST_MEMORY_MODE:N"),
        extra=()):
    """The exit status of compare.py on a synthetic pair; a mode is (mode, line[, count[, verified[, summary]]])."""
    with tempfile.TemporaryDirectory() as t:
        for name, make, mode in (("ctl", make_ctl, ctl_mode), ("arm", make_arm, arm_mode)):
            g = Path(t) / name / "gates"
            g.mkdir(parents=True)
            make(g)
            stamp(Path(t) / name, gate, *mode)
        return subprocess.run([sys.executable, str(COMPARE), f"{t}/ctl", f"{t}/arm", "--gates", gate, *extra],
                              capture_output=True, text=True).returncode


def check_cli(make):
    """The exit status of `compare.py --check-mode` on one synthetic output."""
    with tempfile.TemporaryDirectory() as t:
        g = Path(t) / "gates"
        g.mkdir()
        make(Path(t))
        return subprocess.run([sys.executable, str(COMPARE), "--check-mode", t], capture_output=True, text=True).returncode


def optional_gates(out, mm=True, mutants=True, mm_modes="Y" * 7, mutant_mode="Y", mutant_verified=True,
                   rc_mm=0, rc_mutants=0):
    """SUMMARY.txt and the mode witnesses of the two optional gates, ON."""
    g = out / "gates"
    (out / "mode").write_text("on\n")
    summary = ""
    if mm:
        summary += f"unpinned-mm rc={rc_mm}   5s\n"
        (g / "unpinned-mm/guest").mkdir(parents=True)
        (g / "unpinned-mm/guest/serial.log").write_text("".join(f"MM_ARM:good:multi:{m}\n" for m in mm_modes))
        (g / "unpinned-mm/result.json").write_text(json.dumps({"checks": {"completed": True, "mode": True}}))
    if mutants:
        summary += f"unpinned-mutants rc={rc_mutants}   9s\n"
        for variant in ("unbroken-adaptclip", "vma_clip"):
            (g / "unpinned-mutants" / variant / "guest").mkdir(parents=True)
            (g / "unpinned-mutants" / variant / "guest/serial.log").write_text(f"HOOK_MEMORY_MODE:{mutant_mode}\n")
            (g / "unpinned-mutants" / variant / "result.json").write_text(json.dumps(
                {"checks": {"completed": True, "memory_mode_verified": mutant_verified}}))
    (g / "SUMMARY.txt").write_text(summary)


def main():
    def required(gate, missing=None):
        def make(g):
            (g / f'{gate}.log').write_text('PASS fixture\n' + ''.join(
                f'{row}:0\n' for row in compare.REQUIRED_ROWS[gate] if row != missing))
        return make

    def replace(old, new):
        return lambda lines: [new if l == old else l for l in lines]

    def drop(row):
        return lambda lines: [l for l in lines if not l.startswith(f"VM_EXIT:{row}:")]

    def after(row, extra):
        return lambda lines: sum(([l] + ([extra] if l.startswith(f"VM_EXIT:{row}:") else []) for l in lines), [])

    both = lambda edit: (lambda g: runtime(g, edit))
    cases = [
        ('complete adaptive rows', 'step-hooks', required('step-hooks'), required('step-hooks'), 0),
        ('complete pthread rows', 'pthread', required('pthread'), required('pthread'), 0),
        ("clean runtime", "runtime", runtime, runtime, 0),
        ("sqlite_write exit 126 on both sides (the kit defect)", "runtime",
         both(replace("VM_EXIT:sqlite_write:0", "VM_EXIT:sqlite_write:126")),
         both(replace("VM_EXIT:sqlite_write:0", "VM_EXIT:sqlite_write:126")), 2),
        ("launcher refusal after an exit-0 row", "runtime", runtime,
         both(after("mruby", "capstone-vexec: image: Exec format error")), 2),
        ("contract row without its arguments", "runtime",
         both(after("normal", "VM_CONTRACT_FAIL:argv")), both(after("normal", "VM_CONTRACT_FAIL:argv")), 2),
        ("contract row missing on both sides", "runtime", both(drop("threads")), both(drop("threads")), 2),
        ("a repeated row refused the second time", "runtime", runtime,
         both(after("sqlite_write", "VM_EXIT:sqlite_write:126")), 2),
        ("clean fp", "fp", fp, fp, 0),
        ("fp exit 127", "fp", fp, lambda g: fp(g, "sh: ./fp.dom: not found\nFP_EXIT:127\n"), 2),
        ("fp launcher setup failure (exit 125) on both sides", "fp",
         lambda g: fp(g, "capstone-vexec: open: Permission denied\nFP_EXIT:125\n"),
         lambda g: fp(g, "capstone-vexec: open: Permission denied\nFP_EXIT:125\n"), 2),
        ("fp guest launched nothing", "fp", lambda g: fp(g, "boot\n"), lambda g: fp(g, "boot\n"), 2),
        ("clean memcached", "memcached", memcached, memcached, 0),
        ("memcached case without job.json", "memcached", memcached, lambda g: memcached(g, drop_job="normal2"), 2),
        ("memcached pthread refused once, then run", "memcached", memcached,
         lambda g: memcached(g, retry="pthread"), 2),
        ("memcached fixture never begun, on both sides", "memcached",
         lambda g: memcached(g, skip="fx1"), lambda g: memcached(g, skip="fx1"), 2),
    ]
    for gate, rows in [('pthread', compare.REQUIRED_ROWS['pthread'][1:]), ('step-hooks', [
            r for r in compare.REQUIRED_ROWS['step-hooks'] if r.startswith('HOOK_EXIT:adapt')])]:
        for row in rows:
            cases.append((f'{row} missing on both sides', gate, required(gate, row), required(gate, row), 2))
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "a").mkdir(); (Path(t) / "b").mkdir()
        empty = subprocess.run([sys.executable, str(COMPARE), f"{t}/a", f"{t}/b", "--gates", ""],
                               capture_output=True).returncode
    failed = int(empty != 3)
    print(f"{'PASS' if empty == 3 else 'FAIL'} no gates selected: exit {empty}, want 3")
    # The memory mode: an arm run ON is compared with an OFF control (exit 0 when its logs say Y),
    # and a log that disagrees with the requested mode, no mode line, or no mode file is refused.
    on, off = ("on", "TEST_MEMORY_MODE:Y"), ("off", "TEST_MEMORY_MODE:N")
    modes = [
        ("ON arm, logs say Y", dict(arm_mode=on, extra=("--arm-mode", "on")), 0),
        ("OFF control without the parameter (A)", dict(ctl_mode=("off", "TEST_MEMORY_MODE:A")), 0),
        ("ON requested, log says N", dict(arm_mode=("on", "TEST_MEMORY_MODE:N")), 4),
        ("OFF requested, log says Y", dict(arm_mode=("off", "TEST_MEMORY_MODE:Y")), 4),
        ("ON requested, one line of two says N", dict(arm_mode=("on", "TEST_MEMORY_MODE:Y\nTEST_MEMORY_MODE:N")), 4),
        ("no mode line", dict(arm_mode=("off", None)), 4),
        ("no mode file", dict(arm_mode=(None, "TEST_MEMORY_MODE:N")), 4),
        ("arm required ON but run OFF", dict(extra=("--arm-mode", "on")), 4),
    ]
    pred_fails, trig_fails, both_pass, both_fail = (("FAIL", "PASS"), ("PASS", "FAIL"), ("PASS", "PASS"), ("FAIL", "FAIL"))
    # (name, control (checks, case files or None), arm (checks, case files or None), want)
    fx = [
        ("fx24 fails its prediction on one side, its trigger on the other (the two recorded outcomes)",
         (pred_fails, None), (trig_fails, None), 0),
        ("fx24 fails its trigger on both sides (not created)", (trig_fails, None), (trig_fails, None), 0),
        ("fx24 fails its prediction on both sides (created, then the cause-24 fault)", (pred_fails, None), (pred_fails, None), 0),
        ("fx24 passes on one side and fails on the other", (both_pass, None), (trig_fails, None), 1),
        ("fx24 fails both checks on one side and one on the other", (both_fail, None), (trig_fails, None), 1),
        # P2-ONQ2-4: a single failure that is not one of the two recorded outcomes is never folded
        ("fx24 prediction fails on an unrelated crash (signal 9, no fault, no fixture output)",
         (trig_fails, None), (pred_fails, OTHER_CRASH), 2),
        ("fx24 unrelated crash on both sides", (pred_fails, OTHER_CRASH), (pred_fails, OTHER_CRASH), 2),
        ("fx24 trigger fails but the case shows the created-then-crash outcome", (trig_fails, None), (trig_fails, CREATED_CRASH), 2),
        ("fx24 prediction fails but the case shows the not-created outcome", (pred_fails, NOT_CREATED), (trig_fails, None), 2),
        ("fx24 created-then-crash outcome with a fault at another address", (pred_fails, None),
         (pred_fails, dict(CREATED_CRASH, fault=CREATED_CRASH["fault"].replace("address=0x0 ", "address=0x10 "))), 2),
        ("fx24 created-then-crash outcome without the header-only line", (pred_fails, None),
         (pred_fails, dict(CREATED_CRASH, out="MCAPP-FIX 24 touch\n")), 2),
        ("fx24 not-created outcome with another no-op mark", (trig_fails, None),
         (trig_fails, dict(NOT_CREATED, out=NOT_CREATED["out"].replace("18e0018", "18e0017"))), 2),
        ("fx24 not-created outcome that bails at 'ia not linked' (seen in the merged-head OFF arm)",
         (pred_fails, None), (trig_fails, IA_NOT_LINKED), 0),
        ("fx24 'ia not linked' after a header-only line", (trig_fails, None),
         (trig_fails, dict(IA_NOT_LINKED, out=IA_NOT_LINKED["out"] + "MCAPP-FIX 24 header-only=1\n")), 2),
        ("fx24 'ia not linked' with another no-op mark", (trig_fails, None),
         (trig_fails, dict(IA_NOT_LINKED, out=IA_NOT_LINKED["out"].replace("18e0018", "18e0012"))), 2),
        ("fx24 not-created outcome with a fault record", (trig_fails, None),
         (trig_fails, dict(NOT_CREATED, fault=CREATED_CRASH["fault"])), 2),
    ]
    for name, (c, c_case), (a, a_case), want in fx:
        got = run("memcached", lambda g, c=c, k=c_case: memcached(g, fx24=c, fx24_case=k),
                  lambda g, a=a, k=a_case: memcached(g, fx24=a, fx24_case=k))
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {got}, want {want}")
    for name, kwargs, want in modes:
        got = run("fp", fp, fp, **kwargs)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} memory mode: {name}: exit {got}, want {want}")
    # P2-ONQ2-1: evidence for every module load, and a failed runner verdict is fatal on its own.
    # nodes loads the module twice. The reviewer's reproduction: the second record gone from both
    # sides and the runner's own count check failing on both, which compared IDENTICAL.
    N, N_FULL = "TEST_MEMORY_MODE:N", ("off", "TEST_MEMORY_MODE:N")
    Y = "TEST_MEMORY_MODE:Y"
    complete = [
        ("nodes, two loads, two Y lines", "nodes", dict(ctl_mode=N_FULL, arm_mode=("on", Y), extra=("--arm-mode", "on")), 0),
        ("nodes, second load's record missing on both sides, runner says FAIL",
         "nodes", dict(ctl_mode=("off", N, 1, False), arm_mode=("off", N, 1, False)), 4),
        ("nodes, second load's record missing on both sides, runner says PASS (count alone)",
         "nodes", dict(ctl_mode=("off", N, 1), arm_mode=("off", N, 1)), 4),
        ("nodes, second load's record missing on one side", "nodes", dict(arm_mode=("off", N, 1)), 4),
        ("nodes, a third record", "nodes", dict(arm_mode=("off", N, 3)), 4),
        ("fp, runner memory_mode_verified FAIL on both sides, all lines present",
         "fp", dict(ctl_mode=("off", N, None, False), arm_mode=("off", N, None, False)), 4),
        ("fp, runner memory_mode_verified FAIL on the arm only", "fp", dict(arm_mode=("off", N, None, False)), 4),
        ("fp, no memory_mode_verified result on either side",
         "fp", dict(ctl_mode=("off", N, None, None), arm_mode=("off", N, None, None)), 4),
        ("step-hooks, 29 of 30 records", "step-hooks", dict(arm_mode=("off", N, 29)), 4),
        ("memcached, mode lines and verdict", "memcached", dict(ctl_mode=N_FULL), 0),
        ("runtime, mode line and verdict", "runtime", dict(ctl_mode=N_FULL), 0),
        ("runtime, runner says FAIL", "runtime", dict(arm_mode=("off", N, None, False)), 4),
        # P2-ONQ2-2: an arm's own rejection travels with its summary
        ("summary carries MODE_CHECK_FAILED (production module carries test hooks)", "fp",
         dict(arm_mode=("off", N, None, True, "MODE_CHECK_FAILED module-strings: production module carries test hooks\n")), 4),
        ("summary carries OPTIONAL_GATE_FAILED", "fp",
         dict(arm_mode=("off", N, None, True, "OPTIONAL_GATE_FAILED unpinned-mm rc=1\n")), 4),
        ("an ordinary gate ends rc=1 on both sides (compared, not refused)", "fp",
         dict(ctl_mode=("off", N, None, True, "fp         rc=1     5s\n"), arm_mode=("off", N, None, True, "fp         rc=1     5s\n")), 0),
        ("an optional gate rc=1 in the summary", "fp",
         dict(arm_mode=("on", Y, None, True, "unpinned-mm rc=1      5s\n")), 4),
    ]
    for name, gate, kwargs, want in complete:
        make = {"nodes": required("nodes"),
                "fp": fp, "memcached": memcached, "runtime": runtime,
                "step-hooks": required("step-hooks")}[gate]
        got = run(gate, make, make, **kwargs)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} memory mode evidence: {name}: exit {got}, want {want}")
    # The fault path (unpinned_fast_fault) is its own recorded and read-back fact, like the memory mode.
    FY, FN = "TEST_FAST_FAULT:Y", "TEST_FAST_FAULT:N"
    on_arm = lambda fast: ("on", Y, None, True, None, fast)
    off_arm = lambda fast: ("off", N, None, True, None, fast)
    paths = [
        ("fast on, Y lines, mode on, OFF control without the file", dict(arm_mode=on_arm(("on", FY)), extra=("--arm-mode", "on", "--arm-fast", "on", "--control-fast", "off")), 0),
        ("fast on, Y lines, control pinned fast on but run without the file", dict(arm_mode=on_arm(("on", FY)), extra=("--control-fast", "on")), 4),
        ("fast on but the log says N", dict(arm_mode=on_arm(("on", FN))), 4),
        ("fast on but no fault-path line", dict(arm_mode=on_arm(("on", FY, 0))), 4),
        ("fast on with an extra fault-path line", dict(arm_mode=on_arm(("on", FY, 2))), 4),
        ("fast on requested in mode off", dict(arm_mode=off_arm(("on", FY))), 4),
        ("fast off recorded but the log says Y", dict(arm_mode=on_arm(("off", FY))), 4),
        ("fast off, N lines (positive OFF evidence)", dict(arm_mode=on_arm(("off", FN))), 0),
        ("fast off, A lines (module without the parameter)", dict(arm_mode=on_arm(("off", "TEST_FAST_FAULT:A"))), 0),
        ("fast off recorded and no lines", dict(arm_mode=on_arm(("off", FN, 0))), 4),
        ("arm pinned fast on but recorded off", dict(arm_mode=on_arm(("off", FN)), extra=("--arm-fast", "on")), 4),
    ]
    for name, kwargs, want in paths:
        kwargs.setdefault("ctl_mode", ("off", N))
        got = run("fp", fp, fp, **kwargs)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} fault path evidence: {name}: exit {got}, want {want}")
    cli = [
        ("optional-only summary, both gates succeeded with Y witnesses", lambda o: optional_gates(o), 0),
        ("unpinned-mm rc=1 and no witnesses", lambda o: optional_gates(o, mutants=False, rc_mm=1, mm_modes=""), 1),
        ("unpinned-mm rc=1 with Y witnesses", lambda o: optional_gates(o, mutants=False, rc_mm=1), 1),
        ("unpinned-mutants rc=1", lambda o: optional_gates(o, mm=False, rc_mutants=1), 1),
        ("unpinned-mm rc=0, six of seven loads", lambda o: optional_gates(o, mutants=False, mm_modes="Y" * 6), 1),
        ("unpinned-mm rc=0, one load says N", lambda o: optional_gates(o, mutants=False, mm_modes="Y" * 6 + "N"), 1),
        ("unpinned-mutants rc=0, a run says N", lambda o: optional_gates(o, mm=False, mutant_mode="N"), 1),
        ("unpinned-mutants rc=0, a run's memory_mode_verified false", lambda o: optional_gates(o, mm=False, mutant_verified=False), 1),
        ("unpinned-mutants rc=0 without run records", lambda o: (optional_gates(o, mm=False, mutants=False),
            (o / "gates/SUMMARY.txt").write_text("unpinned-mutants rc=0   1s\n")), 1),
    ]
    for name, make, want in cli:
        got = check_cli(make)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} --check-mode: {name}: exit {got}, want {want}")
    for name, gate, c, a, want in cases:
        got = run(gate, c, a)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {got}, want {want}")
    total = len(cases) + len(modes) + len(fx) + len(complete) + len(paths) + len(cli) + 1
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
