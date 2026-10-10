#!/usr/bin/env python3
"""Run the eleven cases on one virtual arm, and record what each run showed.

    run-virtual.py OUT --state VM --arm {virtual-malloc,virtual-nested-pools} \
        --perl PERLD_ROOT --llvm-bin BIN [--only NN,NN] [--timeout S]

PERLD_ROOT is a virtual-profile build of ports/perl/musl/build-perl-domain.sh: stock for
virtual-malloc, PERLD_SUBLET=1 (SV heads as Sublet lifetimes) for virtual-nested-pools. Its
src/perl-5.36.3/lib is staged in the VM's share as PERL5LIB with the corpus harness, and each
trigger runs as `perl NN.pl` under capstone-vexec, one process per case.

This runner REPORTS. Per case it writes one row: the outcome (completed, exitN, fault,
timeout), the fault's cause and the function its pc lies in, from the image's own symbols,
whether that function is one of the case's fault_sites (host ASan's frames at the pin, declared
before any virtual run), and the harness's verdict line ([PASS] or [FAIL] ...) with any panic
Perl printed. Two controls run first and must pass, or no row of this arm is a reading: the
interpreter evaluates `6*7`, and the harness loads from the staged library.

Exit: 0 when both controls passed and every case produced a row, 75 otherwise.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[2]
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402

PANIC = re.compile(r"(panic: [^\n]*|Attempt to free unreferenced[^\n]*|Attempt to copy freed[^\n]*)")


def outcome(text, result, symbols):
    """(kind, cause, function, harness line)."""
    harness = next((l.strip() for l in text.splitlines() if l.startswith(("[PASS]", "[FAIL]"))), "")
    panic = PANIC.search(text)
    if panic:
        harness = (harness + " " + panic.group(1)).strip()
    if "[runner] TIMEOUT" in text:
        return "timeout", "", "", harness
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        return "fault", str(fault.cause), fault.symbol or "?", harness
    if result.get("kind") == "exit":
        return ("completed" if result.get("value") == 0 else f"exit{result.get('value')}"), "", "", harness
    return f"{result.get('kind', 'none')}{result.get('value', '')}", "", "", harness


def stage_library(perl_root, share):
    """The build's library and the corpus harness, under the share, once per library."""
    lib = perl_root / "src/perl-5.36.3/lib"
    home = share / "perl-lib"
    stamp = home / ".staged-from"
    if not (stamp.is_file() and stamp.read_text() == str(lib)):
        shutil.rmtree(home, ignore_errors=True)
        shutil.copytree(lib, home, ignore=shutil.ignore_patterns("*.o", "*.a"))
        stamp.write_text(str(lib))
    harness = share / "perl-harness"
    harness.mkdir(exist_ok=True)
    shutil.copy(CORPUS / "harness/shim.pl", harness / "shim.pl")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=("virtual-malloc", "virtual-nested-pools"))
    p.add_argument("--perl", type=Path, required=True, help="the build's PERLD_ROOT")
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--only", help="comma-separated case numbers")
    p.add_argument("--timeout", type=int, default=600, help="seconds per case")
    a = p.parse_args()
    if appvm.profile(a.state) != "virtual":
        p.error(f"{a.state} is not a virtual-profile VM")
    image = a.perl / "src/perl-5.36.3/perl"
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    stage_library(a.perl, share)
    cases_dir = share / "perl-cases"
    cases_dir.mkdir(exist_ok=True)
    env = ("-e", "PERL5LIB=/mnt/host/perl-lib:/mnt/host/perl-harness", "--cwd", "/mnt/host/perl-cases")
    raw = a.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    symbols = v.Symbols(a.llvm_bin, image)

    controls = {}
    text, result = appvm.run_app(a.state, image, ["-e", 'print 6*7, "\\n"'], raw / "control-eval",
                                 run_args=env, timeout=a.timeout)
    controls["eval"] = "42" in text.split() and outcome(text, result, symbols)[0] == "completed"
    text, result = appvm.run_app(a.state, image, ["-e", 'require "shim.pl"; ok(1); print "SHIMOK\\n"'],
                                 raw / "control-shim", run_args=env, timeout=a.timeout)
    controls["shim"] = "SHIMOK" in text and outcome(text, result, symbols)[0] == "completed"
    print(f"  controls eval {'ok' if controls['eval'] else 'FAILED'}, "
          f"shim {'ok' if controls['shim'] else 'FAILED'}", flush=True)

    rows = []
    cases = sorted(CORPUS.glob("[0-9][0-9]_*"))
    if a.only:
        cases = [d for d in cases if d.name[:2] in set(a.only.split(","))]
    for d in cases:
        claims = json.loads((d / "case.json").read_text())
        trigger = d / claims["trigger"]
        shutil.copy(trigger, cases_dir / f"{d.name[:2]}.pl")
        text, result = appvm.run_app(a.state, image, [f"/mnt/host/perl-cases/{d.name[:2]}.pl"],
                                     raw / d.name[:2], run_args=env, timeout=a.timeout)
        kind, cause, func, harness = outcome(text, result, symbols)
        sites = claims.get("fault_sites", [])
        in_sites = ("yes" if func in sites else "no") if kind == "fault" and sites else ""
        rows.append({"case": d.name, "arm": a.arm, "outcome": kind, "cause": cause, "function": func,
                     "in_fault_sites": in_sites, "harness": harness[:160],
                     "trigger_sha256": hashlib.sha256(trigger.read_bytes()).hexdigest()})
        print(f"  {d.name[:44]:44} {kind:9} {cause:>3} {func[:28]:28} {in_sites:3} {harness[:50]}", flush=True)

    cols = ("case", "arm", "outcome", "cause", "function", "in_fault_sites", "harness")
    (a.output / "matrix.tsv").write_text(
        "\t".join(cols) + "\n" + "".join("\t".join(r[c] for c in cols) + "\n" for r in rows))
    (a.output / "inputs.json").write_text(json.dumps({
        "arm": a.arm, "image_sha256": v.sha256(image), "cases": len(rows),
        "controls": controls, "timeout_s": a.timeout,
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE),
        "triggers": {r["case"]: r["trigger_sha256"] for r in rows},
    }, indent=2) + "\n")
    tally = {}
    for r in rows:
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    print(f"--- {a.arm}: controls {controls}; {tally}")
    return 0 if all(controls.values()) and len(rows) == len(cases) else 75


if __name__ == "__main__":
    sys.exit(main())
