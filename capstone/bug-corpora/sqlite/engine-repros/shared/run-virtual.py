#!/usr/bin/env python3
"""Run the SQLite engine corpus on the virtual Capstone profile, and record what each case showed.

    run-virtual.py OUT --state <virtual VM> --arm virtual-malloc|virtual-nested-pools
                   --build <build-virtual.py output for that arm> --llvm-bin <dir> --raw <dir>

This runner REPORTS; tools/verdicts.py decides. Each case is its program from
ports/sqlite/repro322/build-virtual.py, run as a process on a persistent VM started with
`capstone_vm up --profile virtual --exact-bounds`. One Observation per case:

  reached     the harness's `<tag> BEGIN` line (repro322_common.h). It is printed before the case
              body, not at the defective access, so a SILENCE here is weaker evidence than a
              marker at the access would be, and every MISSED row says so in its evidence
  completed   `<tag> RETURNED`
  fault       the launcher's domain fault line, the pc resolved from the image's own symbols
  attribution `function` when that symbol is a case.json fault_sites entry -- for this corpus the
              function host ASan reported the defect in. A fault anywhere else is not a catch

WHAT THE ARM IS, read from the build and the VM, not the label: build.json's arm (memsys5 for
virtual-malloc, memsys5-sublet for virtual-nested-pools), the SDK being a virtual one, and the VM
being a virtual-profile VM. Before any case the configuration's controls run
(controls/control_*_mem5.c, built into the same tree with --observe).
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
import appvm  # noqa: E402
import verdicts as v  # noqa: E402

BUILD_ARM = {"virtual-malloc": "memsys5", "virtual-nested-pools": "memsys5-sublet"}
PREDICTED = {"virtual-malloc": v.MISSED, "virtual-nested-pools": v.CAUGHT}


def observe(tag, text, result, symbols, sites):
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "infra"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l), "exit 75")
        return o
    if f"{tag} BEGIN" not in text:
        o.infra, o.notes = "infra", "no BEGIN line: the image did not run"
        return o
    if "CONTROL-FAILED" in text:
        o.notes = next(l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l)
        return o                                       # the case refused its own setup
    o.reached = True
    o.reach_evidence = "the case began (BEGIN line; no marker at the access on this harness)"
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        if fault.symbol and fault.symbol in sites:
            o.attribution, o.attribution_evidence = "function", f"{fault.symbol} is where host ASan reported it"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}; ASan site {sorted(sites) or 'not recorded'}"
        return o
    o.completed = f"{tag} RETURNED" in text
    return o


def control_seen(name, text, result, symbols):
    if f"CONTROL {name} mark" not in text:
        return "none"
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        return "fault" if fault.symbol in ("control_read", "control_write") else "none"
    return "complete" if "RETURNED" in text else "none"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True)
    p.add_argument("--arm", required=True, choices=sorted(BUILD_ARM))
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--only", help="comma-separated case number prefixes")
    a = p.parse_args()

    config = json.loads((CORPUS / "corpus.json").read_text())["arm_configurations"][a.arm]
    spec = v.load_arms()[config]
    build = json.loads((a.build / "build.json").read_text())
    if build.get("arm") != BUILD_ARM[a.arm] or appvm.sdk_identity(build["sdk"])[1] != "virtual" \
            or appvm.profile(a.state) != "virtual":
        print(f"CONTROL-FAILED {a.build} is arm {build.get('arm')!r} on SDK {build.get('sdk')}; arm {a.arm} "
              f"needs {BUILD_ARM[a.arm]!r} on a virtual SDK and a virtual VM", file=sys.stderr)
        return 75
    a.raw.mkdir(parents=True, exist_ok=True)

    controls = []
    for name in spec["controls"]:
        stem = "control_" + name.replace("-", "_")
        image = a.build / f"{stem}.dom"
        if not image.is_file():
            controls.append(v.Control(name, "none", f"no {image.name}"))
            continue
        text, result = appvm.run_app(a.state, image, [stem], a.raw / stem, timeout=300)
        controls.append(v.Control(name, control_seen(name, text, result, v.Symbols(a.llvm_bin, image)),
                                  (result.get("fault") or "")[:160]))
        print(f"  control {name:<12} {controls[-1].observed:<9} (expected {spec['controls'][name]})", flush=True)

    rows = []
    cases = sorted(d for d in CORPUS.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    if a.only:
        cases = [d for d in cases if d.name[:2] in set(a.only.split(","))]
    for d in cases:
        claims = json.loads((d / "case.json").read_text())
        image = a.build / f"{d.name}.dom"
        if d.name in build.get("not_applicable", []):
            o = v.Observation(case=d.name, arm=a.arm, infra="out-of-denominator",
                              notes="the builder marks it not applicable to this arm (it configures sqlite_heap itself)")
        elif not image.is_file():
            o = v.Observation(case=d.name, arm=a.arm, infra="build-failed", notes=f"no {image.name}")
        else:
            tag = build["cases"][d.name]["tag"]
            text, result = appvm.run_app(a.state, image, [tag], a.raw / d.name, timeout=300)
            o = observe(tag, text, result, v.Symbols(a.llvm_bin, image), set(claims.get("fault_sites", [])))
            o.image_sha256 = v.sha256(image)
        o.case, o.arm, o.controls = d.name, a.arm, list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{d.name[:52]:<52} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:80]}",
              flush=True)
    record = v.write_bundle(a.output, "sqlite/engine-repros", a.arm, rows, {
        "configuration": config, "predicted": PREDICTED[a.arm],
        "build": {k: build[k] for k in ("arm", "sdk", "opt", "groups")},
        "amalgamation_sha256": v.sha256(Path(build["amalgamation"]) / "sqlite3.c"),
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)})
    print(f"--- {a.arm} ({config}): {record['tally']}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
