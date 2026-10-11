#!/usr/bin/env python3
"""Run the c-repros cases on one Capstone application-domain arm, and record what each run showed.

    run-arm.py --arm spatial|virtual-malloc --state <vm state> --bindir <build-domain.sh output>
               --llvm-bin <dir with llvm-nm, llvm-readelf> [--out <dir>] [--only 00,03]

This runner REPORTS; tools/verdicts.py decides. Each case becomes one Observation:

  reached     the case printed `PG_DEFECT case=N mark`, the last thing before its defective access
  completed   it printed `case N RETURNED`
  fault       the host's domain fault line, with the pc resolved to a function from the image's
              own symbols
  attribution `function` when that function is the one the case names in its expect_fault_in line
              or in case.json `fault_sites`; nothing else ties a fault to the defect here

WHAT THE ARM IS, read from the build and not from the label. build-domain.sh records the SDK's
heap in <bindir>/build.json; corpus.json `arm_configurations` maps the arm to a configuration in
tools/arms.json, and a configuration needs one heap (app-level0: level0, virtual-mallocng: the virtual profile).
A mismatch is refused before anything runs. The recorded `sublet` arm of 2026-10-06 is the reason:
both PostgreSQL SDKs keep the default level0 heap, so it measured level0 under a Sublet label.

Before any case, <bindir>/controls.dom runs the configuration's controls (a write one past a
malloc'd object, a read after free), and a silence is MISSED only when they behaved as declared.

Exit: 0 judged, 75 nothing could be judged (infrastructure, controls, or a wrong build).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

HEAP = {"app-level0": "level0", "virtual-mallocng": "virtual-mallocng"}


def observe(number, text, result, symbols=None, sites=()):
    """What one run showed, as facts. `sites`: functions the case declares its fault may land in."""
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "infra"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l),
                       "exit 75: the driver refused (a malloc that returned NULL, or the wrong image)")
        return o
    if f"case {number} BEGIN" not in text:
        o.infra, o.notes = "infra", "no BEGIN line: the image did not run"
        return o
    o.reached = f"PG_DEFECT case={number} mark" in text
    o.reach_evidence = "PG_DEFECT mark" if o.reached else ""
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        expect = re.search(r"expect_fault_in=([A-Za-z_][\w.]*)@", text)
        declared = set(sites) | ({expect.group(1)} if expect else set())
        if fault.symbol and fault.symbol in declared:
            o.attribution = "function"
            o.attribution_evidence = f"{fault.symbol} is the case's declared fault site"
        else:
            o.attribution_evidence = (f"the fault is in {fault.symbol or 'no known function'}, the case "
                                      f"declares {sorted(declared) or 'nothing'}")
        return o
    if result.get("kind") == "signal":
        o.notes = f"signal {result.get('value')} with no domain fault line"
        return o
    o.completed = f"case {number} RETURNED" in text
    return o


def observe_control(name, text, result):
    """'fault' when the control faulted after its mark, 'complete' when it returned, else 'none'."""
    if f"CONTROL {name} mark" not in text:
        return "none"
    if v.domain_fault(result.get("fault") or "") or v.domain_fault(text):
        return "fault"
    return "complete" if f"CONTROL {name} RETURNED" in text else "none"


def build_identity(build):
    """What a build directory's SDK was, from build.json: the virtual profile, or the heap."""
    return "virtual-mallocng" if str(build.get("virtual", "OFF")).upper() == "ON" else build.get("heap")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--state", type=Path, help="a persistent capstone-vm (physical profile)")
    where.add_argument("--virtual-kit", type=Path, help="the virtual platform kit (tools/virtualvm.py); "
                       "every control and case then runs in ONE boot")
    p.add_argument("--bindir", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--only", help="comma-separated case numbers")
    p.add_argument("--raw", type=Path, help="where console logs and the virtual stage go; never inside "
                   "the bundle, since results are summaries (SCHEMA.md rule 6). Default "
                   "/tmp/capstone/c-repros-raw/<stamp>-<arm>")
    a = p.parse_args()

    decl = json.loads((CORPUS / "corpus.json").read_text())
    config = decl["arm_configurations"].get(a.arm)
    if config not in HEAP:
        sys.exit(f"arm {a.arm!r} is not a Capstone domain arm of this corpus: {decl['arm_configurations']}")
    spec = v.load_arms()[config]
    build = json.loads((a.bindir / "build.json").read_text())
    if build_identity(build) != HEAP[config] or (spec["target"] == "capstone-virtual") != bool(a.virtual_kit or appvm.profile(a.state) == "virtual"):
        print(f"CONTROL-FAILED {a.bindir} was built as {build_identity(build)!r} and the run is on "
              f"{'the virtual kit' if a.virtual_kit else 'a physical VM'}; arm {a.arm} is {config}, "
              f"which needs {HEAP[config]!r} on {spec['target']}", file=sys.stderr)
        return 75
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or (CORPUS / "results" / f"{stamp}-qemu" / a.arm)
    raw = a.raw or Path("/tmp/capstone/c-repros-raw") / f"{stamp}-{a.arm}"
    raw.mkdir(parents=True, exist_ok=True)

    found = sorted(d for d in CORPUS.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    if a.only:
        keep = set(a.only.split(","))
        found = [d for d in found if d.name[:2] in keep]
    # The plan: (step name, image, argv). Controls first; they decide what a silence means.
    controls_image = a.bindir / "controls.dom"
    plan = [(f"control-{n}", controls_image, [n]) for n in spec["controls"] if controls_image.is_file()]
    plan += [(d.name, a.bindir / f"{d.name}.dom", [str(int(d.name[:2]))])
             for d in found if (a.bindir / f"{d.name}.dom").is_file()]

    if a.virtual_kit:
        batch = virtualvm.Batch(raw / "stage")
        for name, image, argv in plan:
            batch.put(f"images/{image.name}", image)
            batch.step(name, f"./capstone-vexec ./images/{image.name} {' '.join(argv)}")
        runs, serial, completed = batch.execute(a.virtual_kit, raw / "guest", timeout=300 + 120 * len(plan))
        print(f"  virtual boot: {'completed' if completed else 'DID NOT COMPLETE'}; serial {serial}", flush=True)
        platform = virtualvm.platform(a.virtual_kit, a.llvm_bin / "clang", HERE)
    else:
        appvm.require_up(a.state)
        share = Path(json.loads((a.state / "config.json").read_text())["share"])
        lock = appvm.share_lock(share, "c-repros")  # noqa: F841 -- held for the run
        runs = {name: appvm.run_app(a.state, image, argv, raw / name) for name, image, argv in plan}
        platform = appvm.platform(a.state, a.llvm_bin / "clang", HERE)

    controls = []
    for name in spec["controls"]:
        if f"control-{name}" not in runs:
            controls.append(v.Control(name, "none", "the control did not run (no controls.dom, or the boot ended)"))
            continue
        text, result = runs[f"control-{name}"]
        controls.append(v.Control(name, observe_control(name, text, result),
                                  (result.get("fault") or text.strip().splitlines()[-1:] or [""])[0][:160]))
        print(f"  control {name:<16} {controls[-1].observed:<9} (expected {spec['controls'][name]})", flush=True)

    rows = []
    for d in found:
        number = str(int(d.name[:2]))
        claims = json.loads((d / "case.json").read_text())
        image = a.bindir / f"{d.name}.dom"
        if not image.is_file():
            o = v.Observation(case=d.name, arm=a.arm, infra="build-failed", notes=f"no image at {image}")
        elif d.name not in runs:
            o = v.Observation(case=d.name, arm=a.arm, infra="infra", notes="the boot ended before this case ran")
        else:
            text, result = runs[d.name]
            o = observe(number, text, result, v.Symbols(a.llvm_bin, image), claims.get("fault_sites", ()))
            o.case, o.arm, o.image_sha256 = d.name, a.arm, v.sha256(image)
        o.controls = list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{d.name:<56} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}", flush=True)

    record = v.write_bundle(out, "postgres/c-repros", a.arm, rows, {
        "configuration": config, "build": build, "platform": platform})
    print(f"--- {a.arm} ({config}): {record['tally']}\nresults: {out}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
