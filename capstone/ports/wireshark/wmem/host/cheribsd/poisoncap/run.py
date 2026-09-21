#!/usr/bin/env python3
"""Run the wmem defect corpus on CheriBSD: paired PoisonCap arms, or the plain
arm under the guest's own libc revocation.

Every defect arm runs under `supervise`, which reports the fault from OUTSIDE
the program: the signal and trap PC come from the kernel, and the address
`wm_defect_probe` will sit at is resolved from the child's own map plus the
target ELF. A protected arm that faults somewhere OTHER than the labelled
access carries the same exit status as one that faults at it; only the PC
comparison tells them apart.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parents[4] / "common/host/cheribsd/run.py"
SIGPROT = 34
PROBE = "wm_defect_probe"
EXAMPLES = {"allocator-example": "ALLOCATOR_EXAMPLE wireshark PASS pointer_bytes=16"}
FAULT = re.compile(r"SUPERVISE fault signal=(\d+) code=(\S+) addr=(\S+) pc=0x([0-9a-f]+)")
EXPECT = re.compile(r"SUPERVISE expect " + PROBE + r" (?:0x([0-9a-f]+)|unavailable)")
EXIT = re.compile(r"SUPERVISE exit (signalled|status)=(\d+)")


def discover(bins):
    """One program per case, named NN-slug; the directory names are the authority."""
    return [(int(p.name[:2]), p) for p in sorted(bins.glob("[0-9][0-9]-*"))
            if p.is_file() and p.suffix == ""]


def defect_case(program, mode, which, timeout):
    """One arm. Both arms run the SAME binary; only the mode argument differs."""
    case = dict(
        name=f"{program.name}-mode{mode}",
        program=str(program.parent / "supervise"),
        inputs={program.name: str(program)},
        args=["./" + program.name, str(mode), str(which)],
        timeout=timeout,
    )
    if mode == 0:
        case["exit"] = 0
        case["expect"] = f"WM_DEFECT case={which} mode=0 completed"
    else:
        case["exit"] = 128 + SIGPROT
        case["expect"] = f"SUPERVISE exit signalled={SIGPROT}"
        case["also_expect"] = [f"WM_DEFECT case={which} ready"]
    return case


def read_arm(directory):
    out = (directory / "stdout.txt").read_text(errors="replace")
    expect = EXPECT.search(out)
    exit_line = EXIT.search(out)
    faults = [dict(signal=int(s), code=c, addr=a, pc=int(pc, 16)) for s, c, a, pc in FAULT.findall(out)]
    counters = next((l for l in out.splitlines() if l.startswith("WM_POISONCAP mode=")), None)
    return dict(
        probe_address=int(expect.group(1), 16) if expect and expect.group(1) else None,
        faults=faults,
        exit_kind=exit_line.group(1) if exit_line else None,
        exit_value=int(exit_line.group(2)) if exit_line else None,
        counters=counters,
        ready=f"ready" in out,
    )


def pair_verdict(which, zero, one):
    """A pair is only a pair when the unprotected arm reached the same site."""
    completed = zero["exit_kind"] == "status" and zero["exit_value"] == 0
    faulted = one["exit_kind"] == "signalled" and one["exit_value"] == SIGPROT
    at_probe = None
    if faulted and one["probe_address"] is not None:
        at_probe = any(f["pc"] == one["probe_address"] for f in one["faults"])
    return dict(
        case=which, unprotected_completed=completed, protected_faulted=faulted,
        fault_at_probe=at_probe, paired=completed and faulted and at_probe is True,
        protected_fault_pcs=[hex(f["pc"]) for f in one["faults"]],
        probe_address=hex(one["probe_address"]) if one["probe_address"] else None,
        counters=one["counters"],
    )


def single_verdict(which, zero):
    """The plain CheriBSD arm has one mode. Completing means the layer below
    saw nothing; a SIGPROT means it caught the stale access."""
    completed = zero["exit_kind"] == "status" and zero["exit_value"] == 0
    faulted = zero["exit_kind"] == "signalled" and zero["exit_value"] == SIGPROT
    at_probe = None
    if faulted and zero["probe_address"] is not None:
        at_probe = any(f["pc"] == zero["probe_address"] for f in zero["faults"])
    return dict(case=which, completed=completed, caught=faulted, fault_at_probe=at_probe,
                fault_pcs=[hex(f["pc"]) for f in zero["faults"]], ready=zero["ready"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("build", type=Path)
    p.add_argument("output", type=Path)
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10437)
    p.add_argument("--disable-default-revocation", action="store_true")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--modes", default="0,1", help="0,1 for the PoisonCap build; 0 for the plain build")
    p.add_argument("--runtime-revocation", choices=("off", "on"), default="on")
    p.add_argument("--case", action="append", default=[])
    a = p.parse_args()
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    bins = a.build.resolve() / "bin"
    modes = [int(m) for m in a.modes.split(",")]
    if any(m not in (0, 1) for m in modes):
        p.error("modes are 0 and 1")
    found = discover(bins)
    if not found:
        p.error(f"no NN-* case programs under {bins}")
    if not (bins / "supervise").is_file():
        p.error("no supervise binary; build the corpus for cheribsd")
    cases = [dict(name=b, program=str(bins / b), expect=m, timeout=a.timeout) for b, m in EXAMPLES.items()]
    for mode in modes:
        for which, program in found:
            cases.append(defect_case(program, mode, which, a.timeout))
    if a.case:
        unknown = set(a.case) - {c["name"] for c in cases}
        if unknown:
            p.error("unknown cases: " + ", ".join(sorted(unknown)))
        cases = [c for c in cases if c["name"] in a.case]
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "selection.json").write_text(json.dumps(
        dict(modes=modes, complete_suite=not a.case, runtime_revocation=a.runtime_revocation,
             cases=[c["name"] for c in cases]), indent=2) + "\n")
    # Every arm must RUN: a wrong prediction must not cost the remaining arms of the boot.
    run = subprocess.run([
        sys.executable, str(RUNNER), str(a.output / "guest"),
        "--sdk", str(a.sdk), "--rootfs", str(a.rootfs), "--image", str(a.image),
        "--port", str(a.port), "--abi-probe", str(bins / "cheribsd-abi-probe"),
        "--runtime-revocation", a.runtime_revocation,
        "--cases", str(a.output / "cases.json"), "--continue-on-failure",
        *(["--disable-default-revocation"] if a.disable_default_revocation else []),
    ])
    report = a.output / "guest" / "summary.json"
    if not report.is_file():
        print(f"NO SUITE RECORD at {report}; the runner exited {run.returncode} before any case ran", flush=True)
        return 2
    summary = json.loads(report.read_text())
    if not summary.get("ran_all_cases"):
        print("BOOT OR TRANSPORT FAILED before every case ran; no verdict", flush=True)
        (a.output / "matrix.json").write_text(json.dumps(dict(complete=False, status=summary.get("status")), indent=2) + "\n")
        return 2
    arms = {}
    for mode in modes:
        for which, program in found:
            d = a.output / "guest" / f"{program.name}-mode{mode}"
            if d.is_dir():
                arms.setdefault(which, {})[mode] = read_arm(d)
    rows = []
    for which in sorted(arms):
        if 1 in modes and {0, 1} <= set(arms[which]):
            rows.append(pair_verdict(which, arms[which][0], arms[which][1]))
        elif modes == [0] and 0 in arms[which]:
            rows.append(single_verdict(which, arms[which][0]))
    matrix = dict(complete=True, modes=modes, runtime_revocation=a.runtime_revocation,
                  oracle_failures=summary.get("failed_cases", []), rows=rows)
    if 1 in modes:
        matrix["paired"] = sorted(r["case"] for r in rows if r["paired"])
        matrix["unpaired"] = sorted(r["case"] for r in rows if not r["paired"])
    else:
        matrix["caught"] = sorted(r["case"] for r in rows if r["caught"])
        matrix["completed"] = sorted(r["case"] for r in rows if r["completed"])
    (a.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for r in rows:
        if "paired" in r:
            site = "at probe" if r["fault_at_probe"] else "NOT at probe: " + ", ".join(r["protected_fault_pcs"] or ["no fault"])
            print(f"case {r['case']}: mode0 " + ("completed" if r["unprotected_completed"] else "DID NOT complete")
                  + ", mode1 " + ("faulted" if r["protected_faulted"] else "DID NOT fault") + f" [{site}]", flush=True)
        else:
            print(f"case {r['case']}: " + ("caught (SIGPROT" + (" at probe)" if r["fault_at_probe"] else " elsewhere)") if r["caught"]
                  else ("completed, not caught" if r["completed"] else "NEITHER completed nor SIGPROT")), flush=True)
    if 1 in modes:
        print(f"paired: {matrix['paired']}  unpaired: {matrix['unpaired']}", flush=True)
        return 0 if not matrix["unpaired"] and not matrix["oracle_failures"] else 1
    print(f"caught: {matrix['caught']}  completed: {matrix['completed']}", flush=True)
    return 0 if len(matrix["caught"]) + len(matrix["completed"]) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
