#!/usr/bin/env python3
"""Run the pymalloc corpus on the virtual Capstone profile, and record what each case showed.

    run-virtual.py OUT --state <virtual VM> --arm virtual-malloc|virtual-nested-pools
                   --build <build-cases.sh capstone-application OUT> --controls-build <same, controls/>
                   --llvm-bin <dir> --raw <dir>

This runner REPORTS; tools/verdicts.py decides. Each case is the hosted replay (the port's
src/native/main.c around the case's PYC_CASE body), run as a process on a persistent VM started with
`capstone_vm up --profile virtual --exact-bounds`, given a one-event fixture naming the case:

  reached     `PYM completed=1`, printed only after the case body -- and the body's stale access is
              the labelled read_probe, so completing means the probe ran
  fault       the launcher's domain fault line, the pc resolved from the image's own symbols
  attribution `probe` when the pc lies in read_probe, the function that carries pyc_defect_read

WHAT THE ARM IS: virtual-malloc builds the replay with pymalloc stock (its arena one block from
virtual mallocng); virtual-nested-pools with PYMALLOC_SUBLET, the lifetime adapter owning an arena
the virtual heap lends linear, run in mode 1 (retire on free). The build's CMakeCache says which, and
the VM must be a virtual-profile one; a mismatch is refused.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[2]
REPO = HERE.parents[6]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402

SUBLET = {"virtual-malloc": "OFF", "virtual-nested-pools": "ON"}
PREDICTED = {"virtual-malloc": v.MISSED, "virtual-nested-pools": v.CAUGHT}
MAGIC = 0x31594C50524D5950


def fixture(case):
    """struct pym_header (12 x u64) then one struct pym_event whose id is the case number."""
    return struct.pack("<16Q", MAGIC, 1, *([0] * 10), 0, case, 0, 0)


def observe(text, result, symbols):
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        o.reached = fault.symbol == "read_probe"
        if o.reached:
            o.reach_evidence = "the fault is the labelled read itself"
            o.attribution, o.attribution_evidence = "probe", "the pc lies in read_probe (pyc_defect_read)"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}, not read_probe"
        return o
    failed = re.search(r"PYM failed=(\d+)", text)
    if failed or (result.get("kind") == "exit" and result.get("value") not in (0, None)):
        o.infra = "infra"
        o.notes = (failed.group(0) if failed else f"exit {result.get('value')}") + \
            " -- the program refused before or during the case; not a statement about the arm"
        return o
    o.completed = o.reached = bool(re.search(r"PYM completed=1\b", text))
    o.reach_evidence = "PYM completed=1, after the case's read_probe" if o.reached else ""
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=sorted(SUBLET))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--controls-build", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    a = p.parse_args()

    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
    spec = v.load_arms()[config]
    for build in (a.build, a.controls_build):
        cache = (build / "work/00/CMakeCache.txt").read_text()
        sublet = re.search(r"^PYMALLOC_SUBLET:BOOL=(\S+)", cache, re.M)
        sdk = re.search(r"^CAPSTONE_SDK:PATH=(\S+)", cache, re.M)
        if (sublet.group(1) if sublet else "OFF") != SUBLET[a.arm] or not sdk \
                or appvm.sdk_identity(sdk.group(1))[1] != "virtual" or appvm.profile(a.state) != "virtual":
            print(f"CONTROL-FAILED {build} is not arm {a.arm}'s build (PYMALLOC_SUBLET={SUBLET[a.arm]}, "
                  f"virtual SDK, virtual VM)", file=sys.stderr)
            return 75
    share = Path(json.loads((a.state / "config.json").read_text())["share"])
    a.raw.mkdir(parents=True, exist_ok=True)
    mode = ["1"] if a.arm == "virtual-nested-pools" else []

    def run(image, number, tag):
        name = f"pym-fixture-{number:02d}.bin"
        (share / name).write_bytes(fixture(number))
        return appvm.run_app(a.state, image, [f"/mnt/host/{name}", "/tmp/pym-report.bin", *mode],
                             a.raw / tag, timeout=300)

    controls = []
    for d in sorted((CORPUS / "controls").glob("[0-9][0-9]_*")):
        number, name = int(d.name[:2]), d.name.split("_", 2)[2].replace("_", "-")
        image = a.controls_build / "bin" / f"defect-{number:02d}"
        text, result = run(image, number, f"control-{name}")
        o = observe(text, result, v.Symbols(a.llvm_bin, image))
        seen = "fault" if o.attribution == "probe" else ("complete" if o.completed else "none")
        controls.append(v.Control(name, seen, f"fault in {o.fault.symbol}" if o.fault else o.notes))
        print(f"  control {name:<12} {seen:<9} (expected {spec['controls'].get(name)})", flush=True)

    rows = []
    for d in sorted(CORPUS.glob("[0-9][0-9]_*")):
        number = int(d.name[:2])
        image = a.build / "bin" / f"defect-{number:02d}"
        if not image.is_file():
            o = v.Observation(case=d.name, arm=a.arm, infra="build-failed", notes=f"no {image.name}")
        else:
            text, result = run(image, number, d.name)
            o = observe(text, result, v.Symbols(a.llvm_bin, image))
            o.image_sha256 = v.sha256(image)
        o.case, o.arm, o.controls = d.name, a.arm, list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{'ok  ' if verdict[0] == PREDICTED[a.arm] else 'DIFF'} {d.name[:50]:<50} "
              f"{verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:80]}", flush=True)
    record = v.write_bundle(a.output, "cpython/pymalloc-repros", a.arm, rows, {
        "configuration": config, "predicted": PREDICTED[a.arm], "mode": mode[0] if mode else "stock",
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)})
    print(f"--- {a.arm} ({config}): {record['tally']}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
