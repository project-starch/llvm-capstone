#!/usr/bin/env python3
"""Run the allocator-boundary corpus on the virtual Capstone profile, and record what each case showed.

    run-virtual.py OUT --state <virtual VM> --arm virtual-malloc|virtual-nested-pools
                   --build <build-virtual.sh cpython OUT> --llvm-bin <dir> --raw <dir> [--only NN,NN]

This runner REPORTS; tools/verdicts.py decides. Each case is upstream's reproducer, trigger.py, run
by the whole interpreter as a process under capstone-vexec on a persistent VM started with
`capstone_vm up --profile virtual --exact-bounds`. The interpreter is the one heap CPython has,
musl mallocng, with pymalloc stock (virtual-malloc) or with patch 0014 (virtual-nested-pools,
CPY_SUBLET=1), under which obmalloc hands out every block as a child lifetime and revokes it on free:

  reached     `ABR BEGIN`, printed by launch.py before it runs the trigger. It is not a marker at
              the defective access, so a quiet run is weaker evidence than a probe would be, and
              every MISSED row says so. A trigger that stops on ImportError never reached its case
  completed   `ABR RETURNED`, `ABR EXIT` or `ABR RAISED <type>`: the program ended on its own
  fault       the launcher's fault line, the pc resolved from the image's own symbols
  attribution `function` when that symbol is one of the case's fault_sites, which come from a host
              ASan run (probe/asan-sites.py) recorded before any arm ran. A fault anywhere else is
              not a catch, unless the case has a negative_control.py (the same traffic with the
              offending access made valid) and that runs to its end on the same image without a
              fault: then `control`. A control that faults too leaves the fault unattributed

Before any case the image must run a JSON/GC workload; then the configuration's controls run
(controls/), which is what shows the pools arm's protection is in force. The build's
image/manifest.json says which arm it is: nested `none` or `cpython`, profile virtual.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[2]
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402

NESTED = {"virtual-malloc": "none", "virtual-nested-pools": "cpython"}
PREDICTED = {"virtual-malloc": v.MISSED, "virtual-nested-pools": v.CAUGHT}
CONTROL_SITES = {
    "uaf-block": {"check_pyobject_freed_is_freed", "test_pyobject_is_freed", "_PyObject_IsFreed"},
    "bounds-block": {"check_pyobject_forbidden_bytes_is_freed", "test_pyobject_is_freed",
                     "_PyObject_IsFreed"},
}
LAUNCH = '''import os, runpy, sys
path = sys.argv[1]
sys.argv[:] = [path]
os.write(1, b"ABR BEGIN\\n")
try:
    runpy.run_path(path, run_name="__main__")
except SystemExit as e:
    os.write(1, ("ABR EXIT %r\\n" % (e.code,)).encode())
    raise
except BaseException as e:
    os.write(1, ("ABR RAISED %s\\n" % type(e).__name__).encode())
    raise
os.write(1, b"ABR RETURNED\\n")
'''
WORKLOAD = ('import json,gc; a=[{"n":n} for n in range(1000)]; '
            'assert json.loads(json.dumps(a))==a; del a; gc.collect(); print("CPYTHON-OK 1000")')
CAPACITY = re.compile(r"MemoryError|cannot allocate|out of memory")
# The interpreter itself failing: a C function returned an error without setting one. It is not the
# program running to its end, and the physical runner already read it as a failed run.
BROKEN = re.compile(r"SystemError|returned NULL without setting an exception|"
                    r"error return without exception set")


def stage_stdlib(build, share):
    """PYTHONHOME for the guest: the release's Lib, compiled by the build's own native 3.13.7."""
    lib = build / "source/src/Python-3.13.7/Lib"
    native = build / "source/build-python/python"
    home = share / "cpy"
    stamp = home / ".staged-from"
    if stamp.is_file() and stamp.read_text() == str(lib) and \
            (home / "lib/python3.13/_sysconfigdata__linux_.py").is_file():
        return
    shutil.rmtree(home, ignore_errors=True)
    shutil.copytree(lib, home / "lib/python3.13", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    # sysconfig's build-time variables, which prepare-cpython-capstone.sh generates for the target.
    pybuild = build / "source/build" / (build / "source/build/pybuilddir.txt").read_text().strip()
    shutil.copy(pybuild / "_sysconfigdata__linux_.py", home / "lib/python3.13")
    # Some test files are deliberately invalid Python, so compileall's status is not checked.
    subprocess.run([str(native), "-m", "compileall", "-q", "-f", "-d", "/mnt/host/cpy/lib/python3.13",
                    str(home / "lib/python3.13")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stamp.write_text(str(lib))


def run_script(a, share, image, directory, script, tag):
    """Stage DIRECTORY's .py files and run SCRIPT there through launch.py."""
    stage = share / "abr" / tag
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    for f in directory.glob("*.py"):
        shutil.copy(f, stage / f.name)
    (stage / "launch.py").write_text(LAUNCH)
    env = ["-e", "PYTHONHOME=/mnt/host/cpy", "-e", "PYTHONDONTWRITEBYTECODE=1"]
    return appvm.run_app(a.state, image, ["launch.py", script], a.raw / tag,
                         run_args=("--cwd", f"/mnt/host/abr/{tag}", *env), timeout=a.timeout)


def observe(text, result, symbols, sites):
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if "ABR BEGIN" not in text:
        o.infra = "infra"
        o.notes = (f"fault in {fault.symbol} " if fault else "") + "before the trigger began"
        return o
    raised = re.search(r"ABR RAISED (\w+)", text)
    if raised and raised.group(1) in ("ImportError", "ModuleNotFoundError"):
        o.notes = f"the trigger stopped on {raised.group(1)} before its case"
        return o                                       # not reached
    o.reached = True
    o.reach_evidence = "the trigger began (ABR BEGIN; no marker at the access in an upstream reproducer)"
    if fault:
        o.fault = fault
        if fault.symbol and fault.symbol in sites:
            o.attribution = "function"
            o.attribution_evidence = f"{fault.symbol} is where host ASan reported the access"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}; ASan sites {sorted(sites) or 'none recorded'}"
        return o
    for pattern, what in ((CAPACITY, "capacity"), (BROKEN, "interpreter failure")):
        if pattern.search(text):
            o.reached, o.infra = False, "infra"
            o.notes = f"{what}: {pattern.search(text).group(0)}"
            return o
    end = re.search(r"ABR (RETURNED|EXIT [^\n]*|RAISED \w+)", text)
    o.completed = bool(end)
    if end and end.group(1) != "RETURNED":
        o.notes = f"the trigger ended with {end.group(1)}"
    return o


def attribute_by_control(a, share, image, d, o, symbols):
    """The case's negative_control.py, for a fault no fault_site names. It counts only when it
    performed its traffic, passed its own self-test and ended without a fault."""
    text, result = run_script(a, share, image, d, "negative_control.py", d.name + "-control")
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.attribution_evidence += (f"; its negative_control.py faults too (cause={fault.cause} in "
                                   f"{fault.symbol or '?'}), so the fault is not the defect's")
    elif ("NEGATIVE-CONTROL no defect performed" in text and "SELFTEST-FAILED" not in text
          and "ABR RETURNED" in text):
        o.attribution = "control"
        o.attribution_evidence = ("negative_control.py (the same traffic, the offending access made "
                                  "valid) ran to its end on this image without a fault")
    else:
        o.attribution_evidence += "; its negative_control.py did not run to an answer"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=sorted(NESTED))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--only", help="comma-separated case numbers")
    p.add_argument("--timeout", type=int, default=300)
    a = p.parse_args()

    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
    spec = v.load_arms()[config]
    manifest = json.loads((a.build / "image/manifest.json").read_text())
    image = a.build / "image/cpython.dom"
    if manifest.get("profile") != "virtual" or manifest.get("nested") != NESTED[a.arm] \
            or appvm.profile(a.state) != "virtual" or not image.is_file():
        print(f"CONTROL-FAILED {a.build} is profile {manifest.get('profile')} nested "
              f"{manifest.get('nested')}; arm {a.arm} needs virtual / {NESTED[a.arm]} on a virtual VM",
              file=sys.stderr)
        return 75
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    a.raw.mkdir(parents=True, exist_ok=True)
    stage_stdlib(a.build, share)
    symbols = v.Symbols(a.llvm_bin, image)

    # The interpreter must work before anything it does means something.
    (share / "abr").mkdir(exist_ok=True)
    text, result = appvm.run_app(a.state, image, ["-S", "-c", WORKLOAD], a.raw / "workload",
                                 run_args=("-e", "PYTHONHOME=/mnt/host/cpy"), timeout=a.timeout)
    qualified = "CPYTHON-OK 1000" in text
    print(f"  workload {'ok' if qualified else 'FAILED'}", flush=True)
    if not qualified:
        print(f"CONTROL-FAILED the interpreter did not run the workload; see {a.raw / 'workload.out'}",
              file=sys.stderr)
        return 75

    controls = []
    for name in spec["controls"]:
        script = "control_" + name.replace("-", "_") + ".py"
        text, result = run_script(a, share, image, CORPUS / "controls", script, f"control-{name}")
        fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
        if f"CONTROL {name} mark" not in text:
            seen = "none"
        elif fault:
            seen = "fault" if fault.symbol in CONTROL_SITES[name] else "none"
        else:
            seen = "complete" if f"CONTROL {name} RETURNED" in text else "none"
        controls.append(v.Control(name, seen, f"fault in {fault.symbol}" if fault else text.strip()[-160:]))
        print(f"  control {name:<13} {seen:<9} (expected {spec['controls'][name]})", flush=True)

    rows = []
    cases = sorted(CORPUS.glob("[0-9][0-9]_*"))
    if a.only:
        cases = [d for d in cases if d.name[:2] in set(a.only.split(","))]
    for d in cases:
        claims = json.loads((d / "case.json").read_text())
        text, result = run_script(a, share, image, d, claims["trigger"], d.name)
        o = observe(text, result, symbols, set(claims.get("fault_sites", [])))
        if o.fault and o.reached and not o.attribution and (d / "negative_control.py").is_file():
            attribute_by_control(a, share, image, d, o, symbols)
        o.image_sha256 = v.sha256(image)
        o.case, o.arm, o.controls = d.name, a.arm, list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{d.name[:52]:<52} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:80]}",
              flush=True)
    record = v.write_bundle(a.output, "cpython/allocator-boundary-repros", a.arm, rows, {
        "configuration": config, "predicted": PREDICTED[a.arm],
        "build": {k: manifest.get(k) for k in ("application", "profile", "nested", "image_sha256",
                                               "runtime_revision", "runtime_dirty", "stack_bytes")},
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)})
    print(f"--- {a.arm} ({config}): {record['tally']}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
