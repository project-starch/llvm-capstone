#!/usr/bin/env python3
"""Run the 23 cases on one virtual arm, and record what each run showed.

    run-virtual.py OUT --state VM --arm {virtual-malloc,virtual-nested-pools} \
        --image MRUBY --capi-dir DIR --llvm-bin BIN [--smoke SCRIPT]

The image is the port's virtual-profile mruby (build-mruby-domain.sh with MRBD_SDK naming a
virtual SDK): stock for virtual-malloc, MRBD_SUBLET=1 for virtual-nested-pools. A case whose
trigger is a script runs as `mruby trigger.rb` under capstone-vexec, staged in the VM's share. A
case whose trigger is a C-API driver (case.json "trigger": "capi.c") runs its own image,
DIR/capi-NN.dom from probe/build-capi.sh against the same mruby build; it prints `CASE<N> ready`
just before its defective access, and its row records whether that mark came and whether the
fault lies in one of the case's declared fault_sites.

This runner REPORTS. Per case it writes one row: the outcome (completed, fault, timeout, exit),
the fault's cause and the function its pc lies in, from the image's own symbols, and the first
line the harness printed (["PASS"] or the first failure). The arm's control runs first: the
port's scripts/smoke.rb must complete with SMOKE_DONE, or no row of this arm is a reading.

What a row means is the corpus's rule (README, "The revoking arms"): a case is caught by an arm
when the arm faults on it. virtual-nested-pools differs from virtual-malloc by patch 0008 alone,
so a fault there and not on virtual-malloc is the GC-slot revocation's.

Exit: 0 when the control completed and every case produced a row, 75 otherwise.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402

TIMEOUT = 120  # seconds per case; the old in-guest watchdog was 45 s of a native-speed guest


def run(state, image, script_guest, raw):
    text, result = appvm.run_app(state, image, [script_guest], raw, timeout=TIMEOUT)
    return text, result


def outcome(text, result, symbols):
    """(kind, cause, function, first harness line)."""
    first = next((l.strip() for l in text.splitlines() if l.startswith("[")), "")
    if "[runner] TIMEOUT" in text:
        return "timeout", "", "", first
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        return "fault", str(fault.cause), fault.symbol or "?", first
    if result.get("kind") == "exit":
        return ("completed" if result.get("value") == 0 else f"exit{result.get('value')}"), "", "", first
    return f"{result.get('kind', 'none')}{result.get('value', '')}", "", "", first


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=("virtual-malloc", "virtual-nested-pools"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--capi-dir", type=Path, required=True,
                   help="probe/build-capi.sh's output for this arm's mruby build")
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--smoke", type=Path, default=REPO / "capstone/ports/mruby/app/scripts/smoke.rb")
    a = p.parse_args()
    if appvm.profile(a.state) != "virtual":
        p.error(f"{a.state} is not a virtual-profile VM")
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    stage = share / "mruby-cases"
    stage.mkdir(exist_ok=True)
    (share / "files").mkdir(exist_ok=True)          # smoke.rb writes /mnt/host/files/out.txt
    (share / "files").chmod(0o777)
    raw = a.output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    symbols = v.Symbols(a.llvm_bin, a.image)
    image_sha = v.sha256(a.image)

    shutil.copy(a.smoke, stage / "smoke.rb")
    text, result = run(a.state, a.image, "/mnt/host/mruby-cases/smoke.rb", raw / "control-smoke")
    smoke = outcome(text, result, symbols)
    smoke_ok = smoke[0] == "completed" and "SMOKE_DONE" in text
    print(f"  control smoke {smoke[0]} {'SMOKE_DONE' if 'SMOKE_DONE' in text else 'no SMOKE_DONE'}", flush=True)

    rows = []
    for d in sorted(CORPUS.glob("[0-9][0-9]_*")):
        claims = json.loads((d / "case.json").read_text())
        trigger = d / claims["trigger"]
        reached = in_sites = ""
        if trigger.suffix == ".c":
            image = a.capi_dir / f"capi-{d.name[:2]}.dom"
            if not image.is_file():
                print(f"CONTROL-FAILED no {image}", file=sys.stderr)
                return 75
            text, result = appvm.run_app(a.state, image, [], raw / d.name[:2], timeout=TIMEOUT)
            kind, cause, func, first = outcome(text, result, v.Symbols(a.llvm_bin, image))
            reached = "yes" if f"CASE{int(d.name[:2])} ready" in text else "no"
            in_sites = "yes" if func in claims.get("fault_sites", []) else ("no" if kind == "fault" else "")
            first = next((l.strip() for l in text.splitlines() if l.startswith("CASE")), first)
            driver_sha = v.sha256(image)
        else:
            script = stage / f"{d.name[:2]}.rb"
            shutil.copy(trigger, script)
            text, result = run(a.state, a.image, f"/mnt/host/mruby-cases/{script.name}", raw / d.name[:2])
            kind, cause, func, first = outcome(text, result, symbols)
            driver_sha = ""
        rows.append({"case": d.name, "arm": a.arm, "outcome": kind, "cause": cause, "function": func,
                     "reached": reached, "in_fault_sites": in_sites, "harness": first[:160],
                     "trigger_sha256": hashlib.sha256(trigger.read_bytes()).hexdigest(),
                     "driver_sha256": driver_sha})
        print(f"  {d.name[:44]:44} {kind:9} {cause:>3} {func[:28]:28} {reached:3} {first[:44]}", flush=True)

    a.output.mkdir(parents=True, exist_ok=True)
    cols = ("case", "arm", "outcome", "cause", "function", "reached", "in_fault_sites", "harness")
    (a.output / "matrix.tsv").write_text(
        "\t".join(cols) + "\n" + "".join("\t".join(r[c] for c in cols) + "\n" for r in rows))
    (a.output / "inputs.json").write_text(json.dumps({
        "arm": a.arm, "image_sha256": image_sha, "cases": len(rows),
        "control": {"smoke": smoke[0], "smoke_done": smoke_ok},
        "timeout_s": TIMEOUT,
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE),
        "triggers": {r["case"]: r["trigger_sha256"] for r in rows},
        "drivers": {r["case"]: r["driver_sha256"] for r in rows if r["driver_sha256"]},
    }, indent=2) + "\n")
    tally = {}
    for r in rows:
        tally[r["outcome"]] = tally.get(r["outcome"], 0) + 1
    print(f"--- {a.arm}: control smoke {'ok' if smoke_ok else 'FAILED'}; {tally}")
    return 0 if smoke_ok and len(rows) == 23 else 75


if __name__ == "__main__":
    sys.exit(main())
