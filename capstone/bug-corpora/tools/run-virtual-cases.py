#!/usr/bin/env python3
"""Run a case.c corpus of FFmpeg, tshark or memcached on the virtual Capstone profile, all in one boot.

    run-virtual-cases.py --corpus <corpus dir> --arm virtual-malloc --virtual-kit <kit> --sdk <virtual SDK>
                         --llvm-bin <dir with llvm-nm> [--cc-arg ...] [--out DIR] [--raw DIR] [--only 00,03]

This runner REPORTS; tools/verdicts.py decides, as for every virtual corpus. It is the
c-repros runner (postgres/c-repros/shared/run-arm.py) generalised to the corpora whose cases are
`case.c` + `shared/driver.c` programs run as `prog fixed N` / `prog buggy N`:
plain-heap, plain-temporal, subobject and carved.

Per case, the FIXED arm runs first and must print `VERDICT FIXED` and exit 0: it is the case's own
control on this platform (the same image, the same allocations, minus the defect). If it does not,
the case is NO-READING control-failed and its buggy run is not scored.

The BUGGY run becomes one Observation:
  reached     the run printed VERDICT DEFECT-REPRODUCED (the case's own claim that the defective
              access ran), or it faulted AT the defective access: the labelled probe, or a function
              the case declared before the run (case.json `fault_sites`)
  completed   VERDICT DEFECT-REPRODUCED and exit 0
  fault       the launcher's domain fault line, the pc resolved from the image's own symbols
  attribution `probe` at a labelled probe (`[prefix_]read_probe`, `[prefix_]write_probe[_u8|_u32]`),
              `function` at a declared site; nothing else ties a fault to the defect

Before any case, the configuration's controls run in the same boot (arms.json; for
virtual-mallocng the c-repros controls program: a write one past a malloc'd object, a read after
free), and the judge scores no silence from a run whose controls did otherwise.

WHAT THE ARM IS is read from the SDK and the kit, never from the label: the SDK's CMakeCache must
say CAPSTONE_APPLICATION_VIRTUAL=ON (appvm.sdk_identity), the corpus.json must map the arm to a
capstone-virtual configuration, and the run must be on a kit. A physical SDK or VM is refused --
run-capstone-domain.py checks only the heap, and a virtual SDK would pass it as `spatial`.

Exit: 0 judged, 75 nothing could be judged (infrastructure, controls, a wrong build or SDK).
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
TOOLS = HERE.parent
REPO = HERE.parents[3]
sys.path.insert(0, str(TOOLS))
import appvm  # noqa: E402
import virtualvm  # noqa: E402
import verdicts as v  # noqa: E402

PROBE = re.compile(r"(^|_)(read|write)_probe(_u8|_u32)?$")
CONTROLS_C = REPO / "capstone/bug-corpora/postgres/c-repros/shared/controls.c"
CORPORA = ("plain-heap-repros", "plain-temporal-repros", "subobject-repros", "carved-repros")


def observe(number, text, result, symbols=None, sites=()):
    """What one buggy run showed, as facts."""
    o = v.Observation(case="", arm="")
    if "[runner] TIMEOUT" in text or result.get("kind") == "none":
        o.infra, o.notes = "infra", "the step began and never ended (timeout or the guest died)"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "control-failed"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l),
                       "exit 75: the case refused its own setup (a CHECK)")
        return o
    if f"case={number} arm=buggy" not in text:
        o.infra, o.notes = "infra", "no `case=N arm=buggy` line: the image did not run the case"
        return o
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        if fault.symbol and PROBE.search(fault.symbol):
            o.reached, o.attribution = True, "probe"
            o.reach_evidence = o.attribution_evidence = f"the fault is at the labelled probe {fault.symbol}"
        elif fault.symbol and fault.symbol in set(sites):
            o.reached, o.attribution = True, "function"
            o.reach_evidence = o.attribution_evidence = (f"the fault is in {fault.symbol}, a site the case "
                                                         "declared before the run (case.json fault_sites)")
        else:
            o.attribution_evidence = (f"the fault is in {fault.symbol or 'no known function'}; the case's "
                                      f"probe or declared sites {sorted(sites) or '(none)'} are elsewhere")
        return o
    if result.get("kind") == "signal":
        o.notes = f"signal {result.get('value')} with no domain fault line"
        return o
    reproduced = re.search(r"^VERDICT DEFECT-REPRODUCED", text, re.M)
    o.reached = bool(reproduced)
    o.reach_evidence = "the case printed VERDICT DEFECT-REPRODUCED" if reproduced else ""
    o.completed = bool(reproduced) and result.get("kind") == "exit" and result.get("value") == 0
    if not reproduced:
        o.notes = next((l.strip() for l in text.splitlines() if l.startswith("VERDICT")), "no VERDICT line")
    return o


def fixed_ok(number, text, result):
    return (f"case={number} arm=fixed" in text and re.search(r"^VERDICT FIXED", text, re.M)
            and result.get("kind") == "exit" and result.get("value") == 0)


def observe_control(name, text, result):
    """'fault' when the control faulted after its mark, 'complete' when it returned, else 'none'."""
    if f"CONTROL {name} mark" not in text:
        return "none"
    if v.domain_fault(result.get("fault") or "") or v.domain_fault(text):
        return "fault"
    return "complete" if f"CONTROL {name} RETURNED" in text else "none"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--virtual-kit", type=Path, required=True)
    p.add_argument("--sdk", type=Path, required=True)
    p.add_argument("--llvm-bin", type=Path, required=True)
    p.add_argument("--cc-arg", action="append", default=[])
    p.add_argument("--out", type=Path)
    p.add_argument("--raw", type=Path)
    p.add_argument("--only")
    a = p.parse_args()
    corpus = a.corpus.resolve()
    if corpus.name not in CORPORA:
        sys.exit(f"CONTROL-FAILED {corpus}: this runner takes {CORPORA}")
    decl = json.loads((corpus / "corpus.json").read_text())
    config = (decl.get("arm_configurations") or {}).get(a.arm)
    arms = v.load_arms()
    if config not in arms or arms[config]["target"] != "capstone-virtual":
        print(f"CONTROL-FAILED arm {a.arm} maps to {config!r} in {corpus}/corpus.json; it must be a "
              "capstone-virtual configuration of tools/arms.json", file=sys.stderr)
        return 75
    spec = arms[config]
    heap, profile = appvm.sdk_identity(a.sdk)
    if profile != "virtual":
        print(f"CONTROL-FAILED {a.sdk} is a {profile} SDK (heap {heap}); this runner needs the virtual profile",
              file=sys.stderr)
        return 75
    missing = [f for f in virtualvm.FILES if not (a.virtual_kit / f).is_file()]
    if missing:
        print(f"CONTROL-FAILED kit {a.virtual_kit} lacks {missing}", file=sys.stderr)
        return 75
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or (corpus / "results" / f"{stamp}-virtual" / a.arm)
    raw = a.raw or Path("/tmp/capstone/virtual-cases-raw") / f"{stamp}-{corpus.parent.name}-{corpus.name}-{a.arm}"
    bindir = raw / "bin"
    bindir.mkdir(parents=True, exist_ok=False)
    cc = a.sdk / "capstone-cc"
    shared = corpus / "shared"
    found = sorted(d for d in corpus.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    if a.only:
        keep = set(a.only.split(","))
        found = [d for d in found if d.name[:2] in keep]

    # ---- build: the cases, and the configuration's controls ----
    built, build_errors = {}, {}
    for d in found:
        img = bindir / f"{d.name}.dom"
        b = subprocess.run([str(cc), "-O0", "-DCORPUS_VIRTUAL", f"-I{shared}", str(d / "case.c"),
                            str(shared / "driver.c"), *a.cc_arg, "-o", str(img)], capture_output=True, text=True)
        if b.returncode or not img.is_file():
            build_errors[d.name] = " | ".join((b.stderr or b.stdout).strip().splitlines()[:3])[:300]
        else:
            built[d.name] = img
    controls_img = bindir / "controls.dom"
    if spec["controls"]:
        b = subprocess.run([str(cc), "-O0", str(CONTROLS_C), "-o", str(controls_img)], capture_output=True, text=True)
        if b.returncode:
            print(f"CONTROL-FAILED controls build: {b.stderr[:300]}", file=sys.stderr)
            return 75
    plan = [(f"control-{n}", controls_img, [n]) for n in spec["controls"]]
    for d in found:
        if d.name in built:
            n = str(int(d.name[:2]))
            plan += [(f"{d.name}-fixed", built[d.name], ["fixed", n]), (f"{d.name}-buggy", built[d.name], ["buggy", n])]

    # ---- one boot ----
    batch = virtualvm.Batch(raw / "stage")
    for name, image, argv in plan:
        batch.put(f"images/{image.name}", image)
        batch.step(name, f"./capstone-vexec ./images/{image.name} {' '.join(argv)}")
    runs, serial, completed = batch.execute(a.virtual_kit, raw / "guest", timeout=300 + 60 * len(plan))
    print(f"  virtual boot: {'completed' if completed else 'DID NOT COMPLETE'}; serial {serial}", flush=True)
    platform = virtualvm.platform(a.virtual_kit, a.llvm_bin / "clang", HERE)

    controls = []
    for name in spec["controls"]:
        if f"control-{name}" not in runs:
            controls.append(v.Control(name, "none", "the control did not run"))
            continue
        text, result = runs[f"control-{name}"]
        controls.append(v.Control(name, observe_control(name, text, result),
                                  (result.get("fault") or text.strip().splitlines()[-1:] or [""])[0][:160]))
        print(f"  control {name:<14} {controls[-1].observed:<9} (expected {spec['controls'][name]})", flush=True)

    rows = []
    for d in found:
        number = str(int(d.name[:2]))
        sites = json.loads((d / "case.json").read_text()).get("fault_sites") or ()
        if d.name in build_errors:
            o = v.Observation(case=d.name, arm=a.arm, infra="build-failed", notes=build_errors[d.name])
        elif f"{d.name}-buggy" not in runs or f"{d.name}-fixed" not in runs:
            o = v.Observation(case=d.name, arm=a.arm, infra="infra", notes="the boot ended before this case ran")
        else:
            ftext, fres = runs[f"{d.name}-fixed"]
            btext, bres = runs[f"{d.name}-buggy"]
            if not fixed_ok(number, ftext, fres):
                o = v.Observation(case=d.name, arm=a.arm, infra="control-failed",
                                  notes="the FIXED run of the same image did not print VERDICT FIXED and exit 0: "
                                        + (next((l for l in ftext.splitlines() if "VERDICT" in l or "CONTROL-FAILED" in l),
                                                str(fres)))[:200])
            else:
                o = observe(number, btext, bres, v.Symbols(a.llvm_bin, built[d.name]), sites)
            o.case, o.arm, o.image_sha256 = d.name, a.arm, v.sha256(built[d.name])
        o.controls = list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{d.name:<60} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}", flush=True)

    record = v.write_bundle(out, f"{corpus.parent.name}/{corpus.name}", a.arm, rows, {
        "configuration": config, "build": {"sdk_identity": [heap, profile], "cc_args": [local_path(c) for c in a.cc_arg],
                                           "flags": ["-O0", "-DCORPUS_VIRTUAL"]},
        "platform": platform})
    print(f"--- {a.arm} ({config}): {record['tally']}\nresults: {out}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0



SCRATCH = re.compile(r"/tmp/claude-\d+/[^/]+/[^/]+/scratchpad/")


def local_path(arg):
    """A build argument as recorded: a session scratchpad prefix (which spells the host account) and
    the home directory are abbreviated; the files' identity is the image hash, not their path."""
    return SCRATCH.sub("<scratch>/", arg).replace(str(Path.home()) + "/", "~/")


if __name__ == "__main__":
    sys.exit(main())
