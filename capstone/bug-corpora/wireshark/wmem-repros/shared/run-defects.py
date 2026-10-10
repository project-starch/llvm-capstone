#!/usr/bin/env python3
"""Run the wmem defects on the virtual arms, and record what each run showed.

    run-defects.py OUT --state VM --arm {virtual-malloc,virtual-nested-pools} \
        --hosted-build B --controls-hosted-build C --llvm-bin BIN --raw LOGS

This runner REPORTS; it does not decide. Every run becomes one Observation (tools/verdicts.py):
whether the case reached its mark, whether it completed, where it faulted, and whether that
fault is tied to the defect. The verdict -- CAUGHT, MISSED, or NO-READING with a reason -- comes
from the shared judge, against the arm's configuration in tools/arms.json, which corpus.json names
under `arm_configurations`.

Each case is the port's hosted program for it (capstone-application preset, virtual SDK), run as
a process under capstone-vexec. virtual-malloc builds wmem as released, on virtual mallocng;
virtual-nested-pools adds patch 0001 (WM_SUBLET=ON), which makes every object of the block and
block_fast allocators a child lifetime of its block.

WHAT A FAULT HAS TO SHOW TO COUNT. Every case reads or writes its stale or out-of-bounds address
through a labelled probe; a fault counts as the defect's only when its pc resolves into wm_probe
or wm_write_probe (attribution `probe`), from the image's own symbols. A case whose access is
made inside wmem would declare the function (`fault_sites` in its case.json); none does today.

WHAT A SILENCE HAS TO SHOW. Before any case, the arm's controls run from --controls-hosted-build
(the same port, built from controls/ instead of the corpus root): a chunk use-after-free and a
chunk overread that must COMPLETE on virtual-malloc and FAULT on virtual-nested-pools, and a jumbo
use-after-free that must FAULT on both. A silence is MISSED only when they did.

Exit: 0 every row judged, 75 nothing could be judged (infrastructure).
"""

import argparse
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/ports/common/host"))
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import verdicts as v  # noqa: E402

PROBES = {"wm_probe": "the labelled read", "wm_write_probe": "the labelled write"}


def dirs(root):
    """Case directories under root, dense from zero, with their case.json (controls have none)."""
    found = sorted(d for d in root.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    for number, d in enumerate(found):
        if int(d.name[:2]) != number:
            sys.exit(f"{d}: case numbers must be dense from zero")
    return found


def stem_of(d):
    return f"{d.name[:2]}-{d.name.split('_', 2)[2].replace('_', '-')}"


VIRTUAL_PREDICTED = {"virtual-malloc": v.MISSED, "virtual-nested-pools": v.CAUGHT}
VIRTUAL_SUBLET = {"virtual-malloc": "OFF", "virtual-nested-pools": "ON"}


def observe_hosted(which, text, result, symbols, sites):
    """One hosted run (driver.c's main, mode 0), as facts. The driver prints `WM_DEFECT case=N
    ready` as its mark; a fault counts at the probe when its pc resolves into wm_probe or
    wm_write_probe, the functions that hold the labelled access, from the image's own symbols."""
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "infra"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l), "exit 75")
        return o
    o.reached = f"WM_DEFECT case={which} ready" in text
    o.reach_evidence = "the hosted driver's ready mark" if o.reached else ""
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        if fault.symbol in PROBES:
            o.attribution = "probe"
            o.attribution_evidence = f"the pc lies in {fault.symbol}, {PROBES[fault.symbol]}"
        elif fault.symbol and fault.symbol in sites:
            o.attribution, o.attribution_evidence = "function", f"{fault.symbol} is the case's declared fault site"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}, not a probe or {sorted(sites)}"
        return o
    if not o.reached and not text.strip():
        o.infra, o.notes = "infra", "no output at all"
        return o
    o.completed = f"WM_DEFECT case={which} mode=0 completed" in text
    return o


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True, help="a capstone_vm VM started --profile virtual")
    p.add_argument("--arm", required=True, choices=sorted(VIRTUAL_SUBLET))
    p.add_argument("--hosted-build", type=Path, required=True,
                   help="the port configured with the capstone-application preset, -DWM_CORPUS_DIR=<corpus>")
    p.add_argument("--controls-hosted-build", type=Path, required=True,
                   help="the same, -DWM_CORPUS_DIR=<corpus>/controls")
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True, help="console logs; outside the bundle")
    a = p.parse_args()
    import appvm
    if appvm.profile(a.state) != "virtual":
        p.error(f"{a.state} is not a virtual-profile VM")
    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
    spec = v.load_arms()[config]
    for build in (a.hosted_build, a.controls_hosted_build):
        cache = (build / "CMakeCache.txt").read_text()
        sublet = re.search(r"^WM_SUBLET:BOOL=(\S+)", cache, re.M)
        sdk = re.search(r"^CAPSTONE_SDK:PATH=(\S+)", cache, re.M)
        if (sublet.group(1) if sublet else "OFF") != VIRTUAL_SUBLET[a.arm] or not sdk or \
                appvm.sdk_identity(sdk.group(1))[1] != "virtual":
            print(f"CONTROL-FAILED {build} is not arm {a.arm}'s build (WM_SUBLET={VIRTUAL_SUBLET[a.arm]} "
                  f"on a virtual SDK)", file=sys.stderr)
            return 75
    a.raw.mkdir(parents=True, exist_ok=True)

    def program(build, d):
        return build / "bin" / stem_of(d)       # the port's hosted programs land in <build>/bin

    def run(build, d, n):
        image = program(build, d)
        text, result = appvm.run_app(a.state, image, ["0", str(n)], a.raw / f"{a.arm}-{stem_of(d)}")
        return image, text, result

    controls = []
    for n, d in enumerate(dirs(CORPUS / "controls")):
        name = d.name.split("_", 2)[2].replace("_", "-")
        image, text, result = run(a.controls_hosted_build, d, n)
        o = observe_hosted(n, text, result, v.Symbols(a.llvm_bin, image), set())
        seen = "fault" if o.fault and o.attribution == "probe" else ("complete" if o.completed else "none")
        controls.append(v.Control(name, seen, f"fault in {o.fault.symbol}" if o.fault else o.notes))
        print(f"  control {name:<16} {seen:<9} (expected {spec['controls'].get(name)})", flush=True)

    rows = []
    for n, d in enumerate(dirs(CORPUS)):
        claims = json.loads((d / "case.json").read_text())
        if not program(a.hosted_build, d).is_file():
            o = v.Observation(case=d.name, arm=a.arm, infra="build-failed", notes=f"no {stem_of(d)}")
        else:
            image, text, result = run(a.hosted_build, d, n)
            o = observe_hosted(n, text, result, v.Symbols(a.llvm_bin, image), set(claims.get("fault_sites", [])))
            o.image_sha256 = v.sha256(image)
        o.case, o.arm, o.controls = d.name, a.arm, list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{'DIFF' if verdict[0] != VIRTUAL_PREDICTED[a.arm] else 'ok  '} {d.name[:46]:<46} "
              f"{verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}", flush=True)
    record = v.write_bundle(a.output, f"wireshark/{CORPUS.name}", a.arm, rows, {
        "configuration": config, "predicted": VIRTUAL_PREDICTED[a.arm],
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)})
    print(f"--- {a.arm} ({config}): {record['tally']}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
