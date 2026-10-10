#!/usr/bin/env python3
"""Run the memory-context defects on the Capstone domain arms, and record what each run showed.

    run-defects.py OUT --domain-build B --controls-build C --linux-build L [--cases 0,3] [--modes spatial,sublet]

This runner REPORTS; it does not decide. Every boot becomes one Observation (tools/verdicts.py):
whether the case reached its marker, whether it completed, where it faulted, and whether that
fault is tied to the defect. The verdict -- CAUGHT, MISSED, or NO-READING with a reason -- comes
from the shared judge, against the arm's configuration in tools/arms.json, which corpus.json names
under `arm_configurations` (spatial = replay-arena, sublet = replay-sublet).

WHAT A FAULT HAS TO SHOW TO COUNT. Every case but 0 reads its stale alias through the labelled
probe, and pg_mark() publishes the probe's runtime address before the access; a fault counts as
the defect's only at that address (attribution `probe`). Case 0's stale access is a second pfree,
so the manager faults inside itself; its case.json declares the function (`fault_sites`), and the
fault counts only there (attribution `function`), resolved from the image's own symbols.

WHAT A SILENCE HAS TO SHOW. Before any case, the arm's controls run from --controls-build (the
same port, built from controls/ instead of the corpus root): a plain chunk use-after-free that
must COMPLETE on replay-arena and FAULT on replay-sublet. A silence is MISSED only when they did.

Two emulator behaviours are accepted for a fault. Without local trap delivery the fault HALTS the
domain and QEMU exits; with it the fault is DELIVERED, the launcher dies by SIGSEGV and the guest
says so with `__EXIT_CODE__139`. A delivered fault without that line is not counted as a fault.

Exit: 0 every row judged and every reading as the corpus predicts, 1 a reading differs (data),
75 nothing could be judged (infrastructure).
"""

import argparse
import json
import os
from pathlib import Path
import re
import struct
import sys

HERE = Path(__file__).resolve()
CORPUS = HERE.parents[1]
REPO = HERE.parents[5]
sys.path.insert(0, str(REPO / "capstone/ports/common/host"))
sys.path.insert(0, str(REPO / "capstone/bug-corpora/tools"))
from port_support import digest, run_guest, stage_run, write_json  # noqa: E402
import verdicts as v  # noqa: E402

MARKER_BASE = 0xCF18000000000000
MODES = ("spatial", "sublet")
# What the corpus predicts, fixed before any run: the unprotected arm lets every case through,
# Sublet's pools catch every one. A differing reading is data (exit 1), not a failure.
PREDICTED = {"spatial": v.MISSED, "sublet": v.CAUGHT}


def discover(build, mode):
    """The port builds ONE program per case, named NN-slug, and the directory names are the
    authority. Discover them instead of keeping a second list here."""
    found = {}
    for path in sorted((build / "bin").glob(f"[0-9][0-9]-*-{mode}.dom")):
        stem = path.name[: -len(f"-{mode}.dom")]
        found[int(stem[:2])] = (stem, path)
    return found


def discover_case_controls(build, mode):
    """Each case's OWN negative control, built as `NN-slug-<mode>-control.dom`.

    A different question from the programs under controls/: those ask whether this configuration
    reports at all, and this asks whether THIS fault depends on THIS defect -- the same case.c with
    PG_NEGATIVE_CONTROL, which replaces the one invalid access with a valid one and leaves the
    allocation traffic unchanged. discover() cannot pick these up: its glob ends at `-<mode>.dom`.
    """
    found = {}
    for path in sorted((build / "bin").glob(f"[0-9][0-9]-*-{mode}-control.dom")):
        stem = path.name[: -len(f"-{mode}-control.dom")]
        found[int(stem[:2])] = (stem, path)
    return found


def dirs(root):
    """Case directories under root, dense from zero, with their case.json (controls have none)."""
    found = sorted(d for d in root.glob("[0-9][0-9]_*") if (d / "case.c").is_file())
    for number, d in enumerate(found):
        if int(d.name[:2]) != number:
            sys.exit(f"{d}: case numbers must be dense from zero")
    return found


def stem_of(d):
    return f"{d.name[:2]}-{d.name.split('_', 2)[2].replace('_', '-')}"


def boot(out, stem, mode, image, which, linear_arena, env):
    """One guest boot of one program. Returns (run dir, serial text, runner exit, hashes)."""
    run, hashes = stage_run(out, f"{stem}-{mode}-", {
        "defects.dom": image,
        "loader.user": LINUX / "bin/domain-loader",
    })
    (run / "share/selection.bin").write_bytes(struct.pack("<II", which, 0))
    hashes["selection.bin"] = digest(run / "share/selection.bin")
    write_json(run / "manifest.json", {"sha256": hashes, "regions": REGIONS, "platform": PLATFORM})
    command = ("cp /mnt/host/loader.user /tmp/pg-loader && chmod 0755 /tmp/pg-loader"
               " && /tmp/pg-loader /mnt/host/defects.dom /mnt/host/selection.bin --tail")
    if linear_arena:
        command += " --linear-arena"
    result = run_guest(run, command, "__CAPSTONE_PG_HOST_DONE__", env=env, timeout_multiplier=1)
    serial_path = run / "serial.log"
    serial = serial_path.read_text(errors="replace") if serial_path.exists() else ""
    return run, serial, result.returncode, hashes


FAULT_RE = re.compile(r"domain (halted by capability fault|capability fault delivered): "
                      r"cause = (\d+), pc = (0x[0-9a-f]+)")
SITE_RE = re.compile(r"Print = Cap\(\d+, 0x[0-9a-f]+, (0x[0-9a-f]+), (0x[0-9a-f]+), (0x[0-9a-f]+)\)")


def observe(serial, which, runner_exit, image, sites_declared):
    """What one boot showed, as facts. Returns an Observation without case/arm filled in."""
    o = v.Observation(case="", arm="", image_sha256=v.sha256(image))
    if not serial.strip():
        o.infra, o.notes = "infra", f"no serial.log (runner exit {runner_exit}): the guest never started"
        return o
    marker = f"Print = Scalar(0x{MARKER_BASE | which:x})"
    if "_FAILED__" in serial or "CONTROL-FAILED" in serial:
        # pg_give_up: a CHECK in the case's own setup refused, so the condition it needs was
        # never created. Not a statement about the arm.
        o.notes = "the case's own setup CHECK refused"
        return o
    o.reached = marker in serial
    o.reach_evidence = "pg_mark() marker" if o.reached else ""
    after = serial.split(marker, 1)[1] if o.reached else ""
    sites = SITE_RE.findall(after)
    faults = FAULT_RE.findall(serial)
    if faults:
        how, cause, pc = faults[-1]
        if len(faults) > 1:
            o.notes = f"{len(faults)} fault lines; the last is scored"
        if how == "capability fault delivered" and "__EXIT_CODE__139" not in serial:
            o.notes = "a delivered fault without the launcher's SIGSEGV exit; not counted as a fault"
            return o
        pc = int(pc, 16)
        o.fault = v.Fault(cause=int(cause), pc=pc)
        if sites:
            probe, code_start = int(sites[0][0], 16), int(sites[0][1], 16)
            name, off = SYMBOLS[str(image)].lookup(pc, code_start)
            o.fault.symbol, o.fault.offset = name, off
            if pc == probe:
                o.attribution, o.attribution_evidence = "probe", f"pc equals the labelled probe {probe:#x}"
            elif name and name in sites_declared:
                o.attribution = "function"
                o.attribution_evidence = f"{name} is the case's declared fault site"
            else:
                o.attribution_evidence = (f"pc {pc:#x} is neither the probe {probe:#x} nor in a declared "
                                          f"site {sorted(sites_declared) or '(none)'}")
        return o
    o.completed = (runner_exit == 0 and "__CAPSTONE_PG_DEFECT_COMPLETED__" in serial)
    if not o.completed and not o.reached:
        o.infra, o.notes = "infra", f"no marker, no fault, no completion (runner exit {runner_exit})"
    return o


# ---- the virtual profile: the hosted replay as a Capstone process ----------------------------

VIRTUAL_PREDICTED = {"virtual-malloc": v.MISSED, "virtual-pg-pools": v.CAUGHT}
VIRTUAL_SUBLET = {"virtual-malloc": "OFF", "virtual-pg-pools": "ON"}


def observe_hosted(which, text, result, symbols, sites):
    """One hosted run (driver.c's PG_CORPUS_HOSTED main, mode 0), as facts. The hosted driver
    prints `PG_DEFECT case=N ready` as its mark; it cannot publish the probe's address the way the
    domain does, so a fault counts at the probe when its pc resolves into pg_probe, the function
    that holds the pg_defect_probe label, from the image's own symbols."""
    o = v.Observation(case="", arm="", image_sha256=result.get("image_sha256"))
    if "[runner] TIMEOUT" in text:
        o.infra, o.notes = "infra", "runner timeout"
        return o
    if result.get("kind") == "exit" and result.get("value") == 75:
        o.infra = "infra"
        o.notes = next((l.strip() for l in text.splitlines() if "CONTROL-FAILED" in l), "exit 75")
        return o
    o.reached = f"PG_DEFECT case={which} ready" in text
    o.reach_evidence = "the hosted driver's ready mark" if o.reached else ""
    fault = v.domain_fault(result.get("fault") or "", symbols) or v.domain_fault(text, symbols)
    if fault:
        o.fault = fault
        if fault.symbol == "pg_probe":
            o.attribution, o.attribution_evidence = "probe", "the pc lies in pg_probe, the labelled read"
        elif fault.symbol and fault.symbol in sites:
            o.attribution, o.attribution_evidence = "function", f"{fault.symbol} is the case's declared fault site"
        else:
            o.attribution_evidence = f"in {fault.symbol or 'no known function'}, not pg_probe or {sorted(sites)}"
        return o
    if not o.reached and not text.strip():
        o.infra, o.notes = "infra", "no output at all"
        return o
    o.completed = f"PG_DEFECT case={which} mode=0 completed" in text
    return o


def virtual_main():
    p = argparse.ArgumentParser(description="the virtual arms of run-defects.py")
    p.add_argument("output", type=Path)
    p.add_argument("--state", type=Path, required=True, help="a capstone_vm VM started --profile virtual")
    p.add_argument("--arm", required=True, choices=sorted(VIRTUAL_SUBLET))
    p.add_argument("--hosted-build", type=Path, required=True,
                   help="the port configured with the capstone-application preset, -DPG_CORPUS_DIR=<corpus>")
    p.add_argument("--controls-hosted-build", type=Path, required=True,
                   help="the same, -DPG_CORPUS_DIR=<corpus>/controls")
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
        sublet = re.search(r"^PG_SUBLET:BOOL=(\S+)", cache, re.M)
        sdk = re.search(r"^CAPSTONE_SDK:PATH=(\S+)", cache, re.M)
        if (sublet.group(1) if sublet else "OFF") != VIRTUAL_SUBLET[a.arm] or not sdk or \
                appvm.sdk_identity(sdk.group(1))[1] != "virtual":
            print(f"CONTROL-FAILED {build} is not arm {a.arm}'s build (PG_SUBLET={VIRTUAL_SUBLET[a.arm]} "
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
    record = v.write_bundle(a.output, f"postgres/{CORPUS.name}", a.arm, rows, {
        "configuration": config, "predicted": VIRTUAL_PREDICTED[a.arm],
        "platform": appvm.platform(a.state, a.llvm_bin / "clang", HERE)})
    print(f"--- {a.arm} ({config}): {record['tally']}")
    return 75 if rows and all(r[1][0] == v.NO_READING for r in rows) else 0


def main():
    if "--state" in sys.argv[1:]:
        return virtual_main()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output", type=Path)
    p.add_argument("--domain-build", type=Path, required=True)
    p.add_argument("--controls-build", type=Path, required=True)
    p.add_argument("--linux-build", type=Path, required=True)
    p.add_argument("--cases", help="comma-separated case numbers; default: every case directory")
    p.add_argument("--modes", default=",".join(MODES))
    a = p.parse_args()

    global LINUX, REGIONS, PLATFORM, SYMBOLS
    LINUX = a.linux_build
    REGIONS = json.loads((a.domain_build / "regions.json").read_text())
    for other in (a.linux_build, a.controls_build):
        if REGIONS != json.loads((other / "regions.json").read_text()):
            p.error(f"region configurations differ between {a.domain_build} and {other}")
    corpus_json = json.loads((CORPUS / "corpus.json").read_text())
    configurations = corpus_json["arm_configurations"]
    arms = v.load_arms()
    llvm = Path(os.environ["CAPSTONE_LLVM_BUILD_DIR"])
    br = Path(os.environ["CAPSTONE_BUILDROOT_DIR"]) / "build/images"
    PLATFORM = {
        "compiler": digest(llvm / "bin/clang"),
        "qemu": digest(os.environ["CAPSTONE_QEMU_BINARY"]),
        "firmware": digest(br / "fw_jump.elf"),
        "kernel": digest(br / "Image"),
        "rootfs": digest(br / "rootfs.ext2"),
        "runner": v.sha256(HERE),
        "judge": v.sha256(Path(v.__file__)),
    }
    cases = dirs(CORPUS)
    controls = dirs(CORPUS / "controls")
    wanted = [int(n) for n in a.cases.split(",")] if a.cases else list(range(len(cases)))
    modes = a.modes.split(",")
    if not set(modes) <= set(MODES) or any(not 0 <= n < len(cases) for n in wanted):
        p.error(f"modes must be among {MODES}; cases among 0..{len(cases) - 1}")

    env = dict(os.environ)
    env.setdefault("CAPSTONE_QEMU_LOGIN_TIMEOUT", "60")
    env.setdefault("CAPSTONE_GUEST_COMMAND_TIMEOUT", "60")
    a.output.mkdir(parents=True, exist_ok=True)
    SYMBOLS = {}
    status = 0
    for mode in modes:
        config = configurations[mode]
        spec = arms[config]
        built = {**{("case", n): x for n, x in discover(a.domain_build, mode).items()},
                 **{("control", n): x for n, x in discover(a.controls_build, mode).items()},
                 **{("case-control", n): x
                    for n, x in discover_case_controls(a.domain_build, mode).items()}}
        for (kind, n), (stem, image) in built.items():
            SYMBOLS[str(image)] = v.Symbols(llvm / "bin", image)

        # The arm's controls first: they decide whether any silence below can be a reading.
        observed_controls = []
        for n, d in enumerate(controls):
            name = d.name.split("_", 2)[2].replace("_", "-")
            if ("control", n) not in built or built[("control", n)][0] != stem_of(d):
                observed_controls.append(v.Control(name, "none", "no control program built"))
                continue
            stem, image = built[("control", n)]
            run, serial, rc, _ = boot(a.output, f"control-{stem}", mode, image, n, mode == "sublet", env)
            o = observe(serial, n, rc, image, set())
            seen = "fault" if (o.fault and o.attribution == "probe") else (
                "complete" if o.completed and o.reached else "none")
            observed_controls.append(v.Control(name, seen, f"{run.name}: fault={o.fault} {o.notes}"))
            print(f"  control {name:<28} {mode:<8} {seen:<9} (expected {spec['controls'].get(name)})",
                  flush=True)

        rows = []
        for n in wanted:
            d = cases[n]
            claims = json.loads((d / "case.json").read_text())
            if ("case", n) not in built or built[("case", n)][0] != stem_of(d):
                o = v.Observation(case=d.name, arm=mode, infra="build-failed",
                                  notes=f"no {stem_of(d)}-{mode}.dom in {a.domain_build}/bin",
                                  controls=list(observed_controls))
            else:
                stem, image = built[("case", n)]
                run, serial, rc, _ = boot(a.output, stem, mode, image, n, mode == "sublet", env)
                o = observe(serial, n, rc, image, set(claims.get("fault_sites", [])))
                o.case, o.arm, o.controls = d.name, mode, list(observed_controls)
                o.notes = (o.notes + f" run={run.name}").strip()
                # The case's own control, on the same build, when the fault is not attributed yet.
                if o.fault and o.reached and not o.attribution:
                    if ("case-control", n) not in built:
                        o.attribution_evidence += ("; this case has no negative control program in "
                                                   "this build")
                    else:
                        cstem, cimage = built[("case-control", n)]
                        crun, cserial, crc, _ = boot(a.output, cstem + "-control", mode, cimage, n,
                                                     mode == "sublet", env)
                        co = observe(cserial, n, crc, cimage, set())
                        if co.fault:
                            o.attribution_evidence += (
                                f"; its negative control FAULTS too ({crun.name}), so this fault "
                                "does not depend on the defect")
                        elif co.completed and co.reached:
                            o.attribution = "control"
                            o.attribution_evidence = (
                                "its own negative control -- the same program with the one invalid "
                                f"access made valid -- completed on this build ({crun.name})")
                        else:
                            o.attribution_evidence += (
                                f"; its negative control neither completed nor faulted ({crun.name})")
            verdict = v.judge(o, spec)
            rows.append((o, verdict))
            differs = verdict[0] != PREDICTED[mode]
            status = 1 if differs else status
            print(f"{'DIFF' if differs else 'ok  '} case={n} {d.name[:44]:<44} {mode:<8} "
                  f"{verdict[0]}{':' + verdict[1] if verdict[1] else ''}  {verdict[2][:90]}", flush=True)
        record = v.write_bundle(a.output / mode, f"postgres/{CORPUS.name}", mode, rows, {
            "configuration": config, "platform": PLATFORM, "regions": REGIONS,
            "predicted": PREDICTED[mode]})
        print(f"--- {mode} ({config}): {record['tally']}", flush=True)
        if all(r[1][0] == v.NO_READING for r in rows):
            status = 75
    return status


if __name__ == "__main__":
    sys.exit(main())
