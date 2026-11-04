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


def memcached(g, drop_job=None, skip=None, retry=None, fx24=None):
    """fx24: (prediction, trigger) statuses of the fixture that cannot pass as written."""
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
    (g / "memcached/guest").mkdir(parents=True, exist_ok=True)
    (g / "memcached/guest/serial.log").write_text("\n".join(serial) + "\n")
    (g / "memcached.log").write_text("\n".join(log) + "\n")


def stamp(out, mode, line):
    """What run-arm.sh and the runners leave: OUT/mode, and a memory-mode line in each guest log."""
    if mode:
        (out / "mode").write_text(mode + "\n")
    if line:
        for f in list((out / "gates").glob("*.log")) + list((out / "gates").glob("**/serial.log")):
            f.write_text(f.read_text() + line + "\n")


def run(gate, make_ctl, make_arm, ctl_mode=("off", "TEST_MEMORY_MODE:N"), arm_mode=("off", "TEST_MEMORY_MODE:N"),
        extra=()):
    with tempfile.TemporaryDirectory() as t:
        for name, make, mode in (("ctl", make_ctl, ctl_mode), ("arm", make_arm, arm_mode)):
            g = Path(t) / name / "gates"
            g.mkdir(parents=True)
            make(g)
            stamp(Path(t) / name, *mode)
        return subprocess.run([sys.executable, str(COMPARE), f"{t}/ctl", f"{t}/arm", "--gates", gate, *extra],
                              capture_output=True, text=True).returncode


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
    fx = [
        ("fx24 fails its prediction on one side, its trigger on the other (placement)", pred_fails, trig_fails, 0),
        ("fx24 fails its trigger on both sides", trig_fails, trig_fails, 0),
        ("fx24 passes on one side and fails on the other", both_pass, trig_fails, 1),
        ("fx24 fails both checks on one side and one on the other", both_fail, trig_fails, 1),
    ]
    for name, c, a, want in fx:
        got = run("memcached", lambda g, c=c: memcached(g, fx24=c), lambda g, a=a: memcached(g, fx24=a))
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {got}, want {want}")
    for name, kwargs, want in modes:
        got = run("fp", fp, fp, **kwargs)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} memory mode: {name}: exit {got}, want {want}")
    for name, gate, c, a, want in cases:
        got = run(gate, c, a)
        ok = got == want
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: exit {got}, want {want}")
    total = len(cases) + len(modes) + len(fx) + 1
    print(f"{total - failed}/{total} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
