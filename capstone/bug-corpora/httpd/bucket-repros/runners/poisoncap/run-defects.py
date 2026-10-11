#!/usr/bin/env python3
"""Run the httpd bucket-allocator defects on CheriBSD under PoisonCap, as
paired arms: mode 0 publishes exactly bounded aliases and invalidates nothing,
mode 1 poisons and sweeps a node at its release and a bucket piece at its
individual free. Both arms run the SAME binary under `supervise`, which
reports the fault from outside -- signal, si_code and PC from the kernel, the
expected address of apr_defect_read from the child's own map and ELF. A pair
counts only when the unprotected arm completed and the protected one faulted
at that address: a fault anywhere else carries the same exit status.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1]
RUNNER = CORPUS.parents[2] / "ports/common/host/cheribsd/run.py"
MAGIC = 0x314C4F4F50525041
REPORT_BYTES = 96
SIGPROT = 34
PROT_CHERI_TAG = 2
EXIT_INFRASTRUCTURE = 75
EXPECT_LINE = re.compile(r"SUPERVISE expect (\S+) (0x[0-9a-f]+)")
FAULT_LINE = re.compile(r"SUPERVISE fault signal=(\d+) code=(\d+) addr=(0x[0-9a-f]+) pc=(0x[0-9a-f]+)")


def load_cases():
    cases = {}
    for path in sorted(CORPUS.glob("[0-9][0-9]_*/case.json")):
        claim = json.loads(path.read_text())
        cases[claim["case"]] = dict(directory=path.parent.name, slug=path.parent.name.split("_", 2)[2],
                                    upstream_fix=claim["upstream_fix"], shape=claim["shape"])
    if not cases or sorted(cases) != list(range(len(cases))):
        raise SystemExit("the corpus case numbers are not 0..N-1")
    return cases


def fixture(number, negative_control):
    count = 2 if negative_control else 1
    return struct.pack("<16Q", MAGIC, count, *([0] * 10), 0, number, 0, 0)


def build_case(number, mode, supervisor, program, input_path, timeout, slug, negative_control=False):
    case = dict(name=f"{number:02d}-{slug.replace('_', '-')}-mode{mode}", program=str(supervisor),
                args=["./target", "input.bin", "output.bin", str(mode)],
                inputs={"target": str(program), "input.bin": str(input_path)}, timeout=timeout)
    if mode == 0:
        # Only the completing arm writes a report; a faulting one dies at the
        # probe, and an undeclared output is not a failure of the suite.
        case.update(expect="SUPERVISE exit status=0", exit=0)
        if not negative_control:
            case["outputs"] = ["output.bin"]  # a refused fixture writes none, and must still be graded
    else:
        case.update(expect=f"SUPERVISE exit signalled={SIGPROT}", exit=128 + SIGPROT)
    return case


def observed(stdout):
    expect, faults, counters = None, [], None
    for line in stdout.splitlines():
        m = EXPECT_LINE.fullmatch(line)
        if m:
            expect = m[2]
        m = FAULT_LINE.fullmatch(line)
        if m:
            faults.append(dict(signal=int(m[1]), code=int(m[2]), addr=m[3], pc=m[4]))
        if line.startswith("APRP_POISONCAP mode="):
            counters = line
    return expect, faults, counters


def report_fields(blob):
    if blob is None or len(blob) != REPORT_BYTES:
        return None
    magic, count, mode, status, completed = struct.unpack("<5Q", blob[:40])
    return dict(magic_ok=magic == MAGIC, count=count, mode=mode, status=status, completed=completed)


def main():
    claims = load_cases()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("build", type=Path, help="where shared/build-cases.sh poisoncap put bin/")
    p.add_argument("output", type=Path)
    for name in ("sdk", "rootfs", "image"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--port", type=int, default=10443)
    p.add_argument("--runtime-revocation", choices=("off", "on"), default="on")
    p.add_argument("--cases", default=",".join(str(i) for i in sorted(claims)))
    p.add_argument("--modes", default="0,1")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--negative-control", action="store_true")
    a = p.parse_args()
    selected = [int(x) for x in a.cases.split(",")]
    modes = [int(x) for x in a.modes.split(",")]
    bins = a.build.resolve() / "bin"
    supervisor, probe = bins / "supervise", bins / "cheribsd-abi-probe"
    for path in [probe, supervisor] + [bins / f"defect-{n:02d}" for n in selected]:
        if not path.is_file():
            p.error(f"missing program: {path}")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    fixtures = a.output / "fixtures"
    fixtures.mkdir()
    cases, hashes = [], {}
    for number in selected:
        path = fixtures / f"case-{number}.bin"
        path.write_bytes(fixture(number, a.negative_control))
        hashes[number] = hashlib.sha256(path.read_bytes()).hexdigest()
        for mode in modes:
            cases.append(build_case(number, mode, supervisor, bins / f"defect-{number:02d}", path, a.timeout, claims[number]["slug"], a.negative_control))
    (a.output / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (a.output / "selection.json").write_text(json.dumps(dict(
        schema="httpd-bucket-defect-poisoncap-selection-v1", cases=selected, modes=modes,
        negative_control=a.negative_control, runtime_revocation=a.runtime_revocation,
        arms=[c["name"] for c in cases]), indent=2) + "\n")
    guest = a.output / "guest"
    subprocess.run([sys.executable, str(RUNNER), str(guest), "--sdk", str(a.sdk), "--rootfs", str(a.rootfs),
                    "--image", str(a.image), "--port", str(a.port), "--abi-probe", str(probe),
                    "--runtime-revocation", a.runtime_revocation, "--continue-on-failure",
                    "--cases", str(a.output / "cases.json")], check=False)
    summary_path = guest / "summary.json"
    if not summary_path.is_file():
        print(f"NO GUEST SUMMARY: {summary_path}", flush=True)
        return EXIT_INFRASTRUCTURE
    summary = json.loads(summary_path.read_text())
    rows = {row["name"]: row for row in summary["results"]}
    if not summary.get("ran_all_cases"):
        print("BOOT DID NOT REACH EVERY ARM: " + summary.get("status", "unknown"), flush=True)
        return EXIT_INFRASTRUCTURE
    for name in ("cheribsd-abi", "cheribsd-bounds"):
        if not rows.get(name, {}).get("passed"):
            print(f"PLATFORM CONTROL FAILED: {name}", flush=True)
            return EXIT_INFRASTRUCTURE
    arms, status = [], 0
    for number in selected:
        per_mode = {}
        for mode in modes:
            name = build_case(number, mode, supervisor, bins, fixtures, 0, claims[number]["slug"])["name"]
            row = rows.get(name)
            directory = guest / name
            stdout = (directory / "stdout.txt").read_text(errors="replace") if (directory / "stdout.txt").exists() else ""
            expect, faults, counters = observed(stdout)
            report = report_fields((directory / "outputs/output.bin").read_bytes()) if (directory / "outputs/output.bin").exists() else None
            if report is None and (directory / "output.bin").exists():
                report = report_fields((directory / "output.bin").read_bytes())
            v = dict(case=number, mode=mode, exit=row["exit"] if row else None, runner_oracle=bool(row and row["passed"]),
                     expect=expect, faults=faults, counters=counters, report=report)
            if mode == 0:
                v["expected"] = "completed"
                v["passed"] = bool(v["runner_oracle"] and v["exit"] == 0 and not faults and report and report["completed"] == 1 and report["status"] == 0)
            else:
                v["expected"] = "tag fault at apr_defect_read"
                first = faults[0] if faults else None
                v["fault_at_probe"] = bool(first and expect and first["pc"] == expect and first["addr"] == expect)
                v["passed"] = bool(v["runner_oracle"] and v["exit"] == 128 + SIGPROT and first and first["signal"] == SIGPROT
                                   and first["code"] == PROT_CHERI_TAG and v["fault_at_probe"] and len(faults) == 1)
            if a.negative_control:
                v["passed"] = False if v["passed"] else v["passed"]
            per_mode[mode] = v
            arms.append(v)
            extra = f" fault={first['pc'] if mode and faults else None} expect={expect}" if mode else ""
            print(f"{'OK  ' if v['passed'] else 'FAIL'} case={number} mode={mode} {claims[number]['upstream_fix']} exit={v['exit']}{extra}", flush=True)
            if not v["passed"]:
                status = 1
        if 0 in per_mode and 1 in per_mode:
            paired = per_mode[0]["passed"] and per_mode[1]["passed"]
            print(f"     case={number} {'paired' if paired else 'NOT PAIRED'}", flush=True)
    passed = sum(v["passed"] for v in arms)
    (a.output / "verdicts.json").write_text(json.dumps(dict(
        schema="httpd-bucket-defect-poisoncap-v1", negative_control=a.negative_control,
        runtime_revocation=a.runtime_revocation, platform=summary.get("platform"), binaries=summary.get("binaries"),
        platform_controls=[dict(name=n, passed=rows[n]["passed"]) for n in ("cheribsd-abi", "cheribsd-bounds") if n in rows],
        arms=arms), indent=2) + "\n")
    if a.negative_control:
        print(f"negative control: {len(arms) - passed}/{len(arms)} oracles reported FAIL", flush=True)
        return 0 if passed == 0 else 1
    print(f"{passed}/{len(arms)} arms passed", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
