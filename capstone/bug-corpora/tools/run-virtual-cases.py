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

PREBUILT MODE (--prebuilt BUILD --prebuilt-controls BUILD), for the nested corpora: a case there is
not a free-standing case.c but a program linked against its port's allocator library, built by the
port's CMake with the capstone-application preset on the virtual SDK. The runner takes that build
and its controls build as given and refuses them unless their CMakeCache is the configuration the
arm names (HOSTED below: the options each arm requires, and the SDK). Per corpus it knows the argv
of each run, the reach marker the case prints before its defective access, and the labelled probe
function the case's own case.c calls; a fault anywhere else is unattributed.

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
    p.add_argument("--prebuilt", type=Path, help="the port's capstone-application build of the corpus")
    p.add_argument("--prebuilt-controls", type=Path, help="the same build of the corpus's virtual controls")
    a = p.parse_args()
    corpus = a.corpus.resolve()
    if a.prebuilt:
        return run_hosted(a, corpus)
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



# ---- prebuilt hosted programs: the nested corpora ------------------------------------------------

def wmem_probe(case_dir):
    """The probe function the case's own case.c calls; a case calling both or neither is refused. In
    the hosted build the probes are the noinline functions wm_probe / wm_write_probe (driver.c), whose
    one data access is the labelled read or write."""
    text = (case_dir / "case.c").read_text()
    reads = "WM_READ" in text or "wm_probe(" in text
    writes = "WM_WRITE" in text or "wm_write_probe(" in text
    if reads == writes:
        sys.exit(f"CONTROL-FAILED {case_dir.name} calls {'both' if reads else 'no'} labelled probe")
    return "wm_probe" if reads else "wm_write_probe"


HOSTED = {
    # wireshark/wmem-repros on the wmem port (ports/wireshark/wmem, preset capstone-application).
    #   virtual-malloc: stock wmem, g_malloc/g_free = virtual mallocng (WM_LIBC_SYSTEM), mode 0; the
    #     buggy run is the fix differential (`0 N buggy`), whose temporal cases ASSERT the stale chunk
    #     was reoccupied before reading it -- the build the CheriBSD libc arm ran.
    #   virtual-nested-pools: the Sublet region and chunk layers over a payload the virtual heap lends
    #     linear (WM_SUBLET, shared/driver-virtual.c), mode 1; the buggy run is the protected sequence
    #     (`1 N`), as the physical sublet-chunks arm ran it, with no observation added.
    # Both: the fixed run is the differential (`MODE N fixed`): VERDICT FIXED and exit 0.
    "wmem-repros": dict(
        cache={"virtual-malloc": {"WM_LIBC_SYSTEM": "ON", "WM_CHUNKS": "OFF", "WM_SUBLET": "OFF"},
               "virtual-nested-pools": {"WM_SUBLET": "ON", "WM_CHUNKS": "ON", "WM_LIBC_SYSTEM": "OFF"}},
        mode={"virtual-malloc": "0", "virtual-nested-pools": "1"},
        buggy={"virtual-malloc": ["buggy"], "virtual-nested-pools": []},
        probe=wmem_probe,
        ready="WM_DEFECT case={n} ready",
        done="WM_DEFECT case={n} mode={mode} completed",
        controls_dir="controls/virtual",
        control_names={90: "jumbo-reset-90", 91: "chunk-free-91"},
    ),
}


def stem(case_dir):
    """The program name the port's CMake gives a case directory NN_<id>_<slug>: NN-<slug with dashes>."""
    m = re.match(r"^(\d\d)_[^_]+_(.*)$", case_dir.name)
    return f"{m.group(1)}-{m.group(2).replace('_', '-')}"


def cache_says(build, want, sdk):
    """'' when BUILD's CMakeCache is the configuration `want` names on SDK, else what differs."""
    path = build / "CMakeCache.txt"
    if not path.is_file():
        return f"no {path}"
    cache = path.read_text()
    bad = []
    for key, value in want.items():
        m = re.search(rf"^{key}:BOOL=(\w+)$", cache, re.M)
        got = (m.group(1).upper() if m else "OFF")
        got = "ON" if got in ("ON", "1", "TRUE", "YES") else "OFF"
        if got != value:
            bad.append(f"{key}={got}, the arm needs {value}")
    m = re.search(r"^CAPSTONE_SDK:\w+=(.+)$", cache, re.M)
    if not m or Path(m.group(1)).resolve() != sdk.resolve():
        bad.append(f"CAPSTONE_SDK={m.group(1) if m else 'unset'}, not {sdk}")
    if not re.search(r"^PORT_PLATFORM:\w+=capstone-application$", cache, re.M):
        bad.append("PORT_PLATFORM is not capstone-application")
    return "; ".join(bad)


def observe_hosted(n, mode, text, result, symbols, probe, ready, done, differential):
    """What one buggy run of a hosted case showed, as facts."""
    o = v.Observation(case="", arm="")
    if "[runner] TIMEOUT" in text or result.get("kind") == "none":
        o.infra, o.notes = "infra", "the step began and never ended (timeout or the guest died)"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "control-failed"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l),
                       "exit 75: the case refused its own setup (a CHECK, or a payload it could not get)")
        return o
    marker = ready.format(n=n)
    o.reached = marker in text
    o.reach_evidence = f"the case printed `{marker}`, the last thing before its defective access" if o.reached else ""
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        if fault.symbol == probe:
            o.attribution = "probe"
            o.attribution_evidence = f"the fault is in {probe}, the case's labelled probe, whose one data access is the defect's"
        else:
            o.attribution_evidence = f"the fault is in {fault.symbol or 'no known function'}; the case's probe is {probe}"
        return o
    if result.get("kind") == "signal":
        o.notes = f"signal {result.get('value')} with no domain fault line"
        return o
    finished = done.format(n=n, mode=mode) in text and result.get("kind") == "exit" and result.get("value") == 0
    if differential:
        reproduced = bool(re.search(r"^VERDICT DEFECT-REPRODUCED", text, re.M))
        o.completed = finished and reproduced
        if reproduced:
            o.reach_evidence += "; VERDICT DEFECT-REPRODUCED (the stale or crossing access reached other storage)"
        else:
            o.notes = next((l.strip() for l in text.splitlines() if l.startswith("VERDICT")), "no VERDICT line")
    else:
        o.completed = finished
        if not finished:
            o.notes = f"no `{done.format(n=n, mode=mode)}` line with exit 0"
    return o


def fixed_hosted(n, mode, text, result, done):
    return (done.format(n=n, mode=mode) in text and re.search(r"^VERDICT FIXED", text, re.M)
            and result.get("kind") == "exit" and result.get("value") == 0)


def run_hosted(a, corpus):
    if corpus.name not in HOSTED:
        sys.exit(f"CONTROL-FAILED {corpus}: --prebuilt takes {sorted(HOSTED)}")
    ad = HOSTED[corpus.name]
    decl = json.loads((corpus / "corpus.json").read_text())
    config = (decl.get("arm_configurations") or {}).get(a.arm)
    arms = v.load_arms()
    if config not in arms or arms[config]["target"] != "capstone-virtual" or a.arm not in ad["mode"]:
        print(f"CONTROL-FAILED arm {a.arm} maps to {config!r}; it must be a capstone-virtual configuration "
              f"and one of {sorted(ad['mode'])}", file=sys.stderr)
        return 75
    spec = arms[config]
    heap, profile = appvm.sdk_identity(a.sdk)
    if profile != "virtual":
        print(f"CONTROL-FAILED {a.sdk} is a {profile} SDK (heap {heap})", file=sys.stderr)
        return 75
    for build in (a.prebuilt, a.prebuilt_controls):
        why = cache_says(build, ad["cache"][a.arm], a.sdk) if build else "no --prebuilt-controls"
        if why:
            print(f"CONTROL-FAILED {build} is not the {a.arm} build: {why}", file=sys.stderr)
            return 75
    missing = [f for f in virtualvm.FILES if not (a.virtual_kit / f).is_file()]
    if missing:
        print(f"CONTROL-FAILED kit {a.virtual_kit} lacks {missing}", file=sys.stderr)
        return 75
    mode = ad["mode"][a.arm]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or (corpus / "results" / f"{stamp}-virtual" / a.arm)
    raw = a.raw or Path("/tmp/capstone/virtual-cases-raw") / f"{stamp}-{corpus.parent.name}-{corpus.name}-{a.arm}"
    bindir = raw / "bin"
    bindir.mkdir(parents=True, exist_ok=False)
    found = sorted(d for d in corpus.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    if a.only:
        keep = set(a.only.split(","))
        found = [d for d in found if d.name[:2] in keep]
    programs = {d.name: a.prebuilt / "bin" / stem(d) for d in found}
    absent = [str(p) for p in programs.values() if not p.is_file()]
    if absent:
        print(f"CONTROL-FAILED the build lacks {absent[:3]}", file=sys.stderr)
        return 75
    controls_img = bindir / "controls.dom"
    b = subprocess.run([str(a.sdk / "capstone-cc"), "-O0", str(CONTROLS_C), "-o", str(controls_img)],
                       capture_output=True, text=True)
    if b.returncode:
        print(f"CONTROL-FAILED controls build: {b.stderr[:300]}", file=sys.stderr)
        return 75
    cdirs = sorted(d for d in (corpus / ad["controls_dir"]).glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    port_controls = {}
    for d in cdirs:
        name = ad["control_names"][int(d.name[:2])]
        if name not in spec["controls"]:
            continue
        prog = a.prebuilt_controls / "bin" / stem(d)
        if not prog.is_file():
            print(f"CONTROL-FAILED the controls build lacks {prog}", file=sys.stderr)
            return 75
        port_controls[name] = (d, prog)
    plan = [(f"control-{c}", controls_img, [c]) for c in spec["controls"] if c not in port_controls]
    plan += [(f"control-{name}", prog, [mode, str(int(d.name[:2]))]) for name, (d, prog) in port_controls.items()]
    for d in found:
        n = str(int(d.name[:2]))
        plan += [(f"{d.name}-fixed", programs[d.name], [mode, n, "fixed"]),
                 (f"{d.name}-buggy", programs[d.name], [mode, n, *ad["buggy"][a.arm]])]

    batch = virtualvm.Batch(raw / "stage")
    for name, image, argv in plan:
        batch.put(f"images/{image.name}", image)
        batch.step(name, f"./capstone-vexec ./images/{image.name} {' '.join(argv)}")
    runs, serial, completed = batch.execute(a.virtual_kit, raw / "guest", timeout=300 + 60 * len(plan))
    print(f"  virtual boot: {'completed' if completed else 'DID NOT COMPLETE'}; serial {serial}", flush=True)
    platform = virtualvm.platform(a.virtual_kit, a.llvm_bin / "clang", HERE)

    controls = []
    for c in spec["controls"]:
        if f"control-{c}" not in runs:
            controls.append(v.Control(c, "none", "the control did not run"))
        elif c in port_controls:
            d, prog = port_controls[c]
            text, result = runs[f"control-{c}"]
            o = observe_hosted(str(int(d.name[:2])), mode, text, result, v.Symbols(a.llvm_bin, prog),
                               ad["probe"](d), ad["ready"], ad["done"], False)
            seen = ("fault" if o.fault and o.reached and o.attribution == "probe" else
                    "complete" if o.reached and o.completed else "none")
            detail = (result.get("fault") or o.attribution_evidence or o.notes
                      or (text.strip().splitlines()[-1:] or [""])[0])
            controls.append(v.Control(c, seen, str(detail)[:160]))
        else:
            text, result = runs[f"control-{c}"]
            controls.append(v.Control(c, observe_control(c, text, result),
                                      (result.get("fault") or text.strip().splitlines()[-1:] or [""])[0][:160]))
        print(f"  control {c:<16} {controls[-1].observed:<9} (expected {spec['controls'][c]})", flush=True)

    rows = []
    for d in found:
        n = str(int(d.name[:2]))
        if f"{d.name}-buggy" not in runs or f"{d.name}-fixed" not in runs:
            o = v.Observation(case=d.name, arm=a.arm, infra="infra", notes="the boot ended before this case ran")
        else:
            ftext, fres = runs[f"{d.name}-fixed"]
            btext, bres = runs[f"{d.name}-buggy"]
            if not fixed_hosted(n, mode, ftext, fres, ad["done"]):
                o = v.Observation(case=d.name, arm=a.arm, infra="control-failed",
                                  notes="the FIXED run of the same image did not finish with VERDICT FIXED and exit 0: "
                                        + (next((l for l in ftext.splitlines() if "VERDICT" in l or "CONTROL-FAILED" in l),
                                                str(fres)))[:200])
            else:
                o = observe_hosted(n, mode, btext, bres, v.Symbols(a.llvm_bin, programs[d.name]), ad["probe"](d),
                                   ad["ready"], ad["done"], bool(ad["buggy"][a.arm]))
        o.case, o.arm, o.image_sha256 = d.name, a.arm, v.sha256(programs[d.name])
        o.controls = list(controls)
        verdict = v.judge(o, spec)
        rows.append((o, verdict))
        print(f"{d.name:<60} {verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}", flush=True)

    opts = {k: re.search(rf"^{k}:\w+=(.*)$", (a.prebuilt / "CMakeCache.txt").read_text(), re.M)
            for k in ("WM_LIBC_SYSTEM", "WM_CHUNKS", "WM_SUBLET", "CMAKE_BUILD_TYPE", "CMAKE_C_FLAGS")}
    record = v.write_bundle(out, f"{corpus.parent.name}/{corpus.name}", a.arm, rows, {
        "configuration": config,
        "build": {"sdk_identity": [heap, profile], "prebuilt": local_path(str(a.prebuilt)),
                  "cache": {k: (m.group(1) if m else None) for k, m in opts.items()},
                  "argv": {"fixed": [mode, "N", "fixed"], "buggy": [mode, "N", *ad["buggy"][a.arm]]},
                  "controls": {c: v.sha256(prog) for c, (_, prog) in port_controls.items()}},
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
