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
CORPUS = HERE.parents[5] / "bug-corpora/wireshark/wmem-repros"
# The labelled access each case makes, read from its own case.c: a read goes through wm_probe (label
# wm_defect_probe), a write through wm_write_probe (label wm_defect_write). One supervisor per label.
SUPERVISOR = {"wm_defect_probe": "supervise", "wm_defect_write": "supervise-wm_defect_write"}
EXAMPLES = {"allocator-example": "ALLOCATOR_EXAMPLE wireshark PASS pointer_bytes=16"}
FAULT = re.compile(r"SUPERVISE fault signal=(\d+) code=(\S+) addr=(\S+) pc=0x([0-9a-f]+)")
EXPECT = re.compile(r"SUPERVISE expect (?:wm_defect_probe|wm_defect_write) (?:0x([0-9a-f]+)|unavailable)")
EXIT = re.compile(r"SUPERVISE exit (signalled|status)=(\d+)")


def discover(bins):
    """One program per case, named NN-slug; the directory names are the authority."""
    return [(int(p.name[:2]), p) for p in sorted(bins.glob("[0-9][0-9]-*"))
            if p.is_file() and p.suffix == ""]


def declared_sites(which):
    """The functions the case's case.json declared, before any run, as where its defective access
    faults when it is a call INTO the allocator rather than a labelled probe (fault_sites)."""
    case = next(CORPUS.glob(f"{which:02d}_*")) / "case.json"
    return set(json.loads(case.read_text()).get("fault_sites") or ())


def probe_label(which):
    """The label of the one probe the case's own case.c calls; anything else is refused -- except a
    case that calls none and declares fault_sites (its access is the allocator call itself, e.g. a
    double free). For that case wm_defect_probe, which every program carries, is only the ANCHOR the
    supervisor resolves to learn the load base; the catch is judged against the declared sites."""
    src = next(CORPUS.glob(f"{which:02d}_*")) / "case.c"
    text = src.read_text()
    reads = "WM_READ" in text or "wm_probe(" in text
    writes = "WM_WRITE" in text or "wm_write_probe(" in text
    if not reads and not writes and declared_sites(which):
        return "wm_defect_probe"
    if reads == writes:
        raise SystemExit(f"CONTROL-FAILED case {which} calls {'both' if reads else 'no'} labelled probe")
    return "wm_defect_probe" if reads else "wm_defect_write"


def function_at(nm, program, anchor, anchor_rt, pc):
    """The defined function of PROGRAM holding runtime pc, through the anchor label's runtime address
    (the supervisor's `expect` line) minus its ELF value. None when it cannot be resolved."""
    if anchor_rt is None:
        return None
    out = subprocess.run([str(nm), "-S", "--defined-only", str(program)], capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in "tTwW":
            table[parts[3]] = (int(parts[0], 16), int(parts[1], 16))
        elif len(parts) == 3 and parts[1] in "tT":
            table.setdefault(parts[2], (int(parts[0], 16), 0))
    if anchor not in table:
        return None
    off = pc - (anchor_rt - table[anchor][0])
    for name, (value, size) in table.items():
        if size and value <= off < value + size:
            return name
    return ""


def defect_case(program, mode, which, timeout):
    """One arm. Both arms run the SAME binary; only the mode argument differs."""
    case = dict(
        name=f"{program.name}-mode{mode}",
        program=str(program.parent / SUPERVISOR[probe_label(which)]),
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


def pair_verdict(which, zero, one, site_of=None):
    """A pair is only a pair when the unprotected arm reached the same site. For a case that declares
    fault_sites, `site_of(pc)` names the function a fault lies in, and the site must be a declared one."""
    completed = zero["exit_kind"] == "status" and zero["exit_value"] == 0
    faulted = one["exit_kind"] == "signalled" and one["exit_value"] == SIGPROT
    at_probe = None
    sites = declared_sites(which) if site_of else set()
    if faulted and sites:
        at_probe = any(site_of(f["pc"], one["probe_address"]) in sites for f in one["faults"])
    elif faulted and one["probe_address"] is not None:
        at_probe = any(f["pc"] == one["probe_address"] for f in one["faults"])
    return dict(
        case=which, unprotected_completed=completed, protected_faulted=faulted,
        fault_at_probe=at_probe, paired=completed and faulted and at_probe is True,
        protected_fault_pcs=[hex(f["pc"]) for f in one["faults"]],
        probe_address=hex(one["probe_address"]) if one["probe_address"] else None,
        counters=one["counters"],
    )


def single_verdict(which, zero, site_of=None):
    """The plain CheriBSD arm has one mode. Completing means the layer below
    saw nothing; a SIGPROT counts as a catch ONLY at the case's labelled probe.
    A SIGPROT anywhere else -- or one whose probe the supervisor could not resolve
    -- is reported as faulted_elsewhere: it is not attributed to the defect.
    (Until 2026-10-10 any SIGPROT counted as caught.)"""
    completed = zero["exit_kind"] == "status" and zero["exit_value"] == 0
    faulted = zero["exit_kind"] == "signalled" and zero["exit_value"] == SIGPROT
    at_probe = None
    sites = declared_sites(which) if site_of else set()
    if faulted and sites:
        at_probe = any(site_of(f["pc"], zero["probe_address"]) in sites for f in zero["faults"])
    elif faulted and zero["probe_address"] is not None:
        at_probe = any(f["pc"] == zero["probe_address"] for f in zero["faults"])
    return dict(case=which, completed=completed, caught=faulted and at_probe is True,
                faulted_elsewhere=faulted and at_probe is not True, fault_at_probe=at_probe,
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
    for sup in sorted({SUPERVISOR[probe_label(w)] for w, _ in found}):
        if not (bins / sup).is_file():
            p.error(f"no {sup} binary; build the corpus for cheribsd")
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
    programs = dict(found)
    nm = a.sdk / "bin" / "llvm-nm"
    for which in sorted(arms):
        def site_of(pc, anchor_rt, program=programs[which]):
            return function_at(nm, program, "wm_defect_probe", anchor_rt, pc)
        if 1 in modes and {0, 1} <= set(arms[which]):
            rows.append(pair_verdict(which, arms[which][0], arms[which][1], site_of))
        elif modes == [0] and 0 in arms[which]:
            rows.append(single_verdict(which, arms[which][0], site_of))
    matrix = dict(complete=True, modes=modes, runtime_revocation=a.runtime_revocation,
                  oracle_failures=summary.get("failed_cases", []), rows=rows)
    if 1 in modes:
        matrix["paired"] = sorted(r["case"] for r in rows if r["paired"])
        matrix["unpaired"] = sorted(r["case"] for r in rows if not r["paired"])
    else:
        matrix["caught"] = sorted(r["case"] for r in rows if r["caught"])
        matrix["completed"] = sorted(r["case"] for r in rows if r["completed"])
        matrix["faulted_elsewhere"] = sorted(r["case"] for r in rows if r["faulted_elsewhere"])
    (a.output / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")
    for r in rows:
        if "paired" in r:
            site = "at probe" if r["fault_at_probe"] else "NOT at probe: " + ", ".join(r["protected_fault_pcs"] or ["no fault"])
            print(f"case {r['case']}: mode0 " + ("completed" if r["unprotected_completed"] else "DID NOT complete")
                  + ", mode1 " + ("faulted" if r["protected_faulted"] else "DID NOT fault") + f" [{site}]", flush=True)
        else:
            print(f"case {r['case']}: " + ("caught (SIGPROT at probe)" if r["caught"]
                  else "SIGPROT NOT at the labelled probe: " + ", ".join(r["fault_pcs"] or ["no pc"]) if r["faulted_elsewhere"]
                  else ("completed, not caught" if r["completed"] else "NEITHER completed nor SIGPROT")), flush=True)
    if 1 in modes:
        print(f"paired: {matrix['paired']}  unpaired: {matrix['unpaired']}", flush=True)
        return 0 if not matrix["unpaired"] and not matrix["oracle_failures"] else 1
    print(f"caught: {matrix['caught']}  completed: {matrix['completed']}", flush=True)
    return 0 if len(matrix["caught"]) + len(matrix["completed"]) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
