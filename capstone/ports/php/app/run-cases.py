#!/usr/bin/env python3
"""Run the Phase A spatial pair on the virtual profile and report Observations to the judge.

WHAT THIS DOES AND DOES NOT DECIDE. It observes -- reached, completed, faulted, where -- and
hands each Observation to capstone/bug-corpora/tools/verdicts.py, which owns the verdict. The
physical suite (engine/run-triggers.sh) decides its own verdicts in bash; that is the thing the
retarget replaces, because a judge inside the runner cannot be re-run over stored results when
the rules change.

THE DENOMINATOR IS DERIVED FROM THE PHYSICAL SUITE, NOT RETYPED. The case list, each case's
axis, and the file a fault must land in are all read out of engine/run-triggers.sh. Two lists
that must agree and live in two files is how a port quietly stops being the same experiment as
its comparison column.

ONLY THE SPATIAL AXIS IS IN SCOPE HERE, and that is a property of Phase A rather than of the
profile. The physical suite measures the temporal axis in `seam` mode -- PHP's allocator
dropped, the port's arena serving emalloc directly, with revocation per object. That arena is
physical-profile code; replacing it with Sublet regions over a block the virtual heap lends is
the allocator-seam work the plan treats as its own design question. Of the 24 reachable cases,
6 are on the spatial axis (axis_of says `heap`); for the other 18 this runner reports nothing
at all, which is not the same as reporting a miss.

  run-cases.py --qemu Q --images I --adapter A --control php-control.dom --fault php-fault.dom
               --out <bundle dir> [--case CRASH-110 ...]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
TRIGGERS = HERE / "triggers"
TOOLS = REPO / "capstone" / "bug-corpora" / "tools"
SUITE = HERE.parent / "engine" / "run-triggers.sh"
sys.path.insert(0, str(TOOLS))
import verdicts                                                  # noqa: E402

# The arm names as arms.json knows them, and the image each one is built from. The build
# derives its image name from the same flag, so a mislabelled pair cannot reach here.
ARMS = {"php-virtual-zend-rounded": "control", "php-virtual-zend-exact": "fault"}
CONTROL_RUN = "emalloc-overread"      # must match arms.json's `controls` key


def suite_text():
    if not SUITE.is_file():
        sys.exit(f"the physical suite is not at {SUITE}; the case list is read from it")
    return SUITE.read_text()


def read_expect_file(text):
    """case -> the source file a fault must land in (EXPECT_FILE in the physical suite)."""
    block = re.search(r"declare -A EXPECT_FILE=\((.*?)\)", text, re.S)
    if not block:
        sys.exit("run-triggers.sh no longer declares EXPECT_FILE")
    pairs = dict(re.findall(r"\[(CRASH-\d+)\]=(\S+)", block.group(1)))
    if not pairs:
        sys.exit("EXPECT_FILE parsed empty")
    return pairs


def read_axis(text):
    """case -> axis, from axis_of's case statement."""
    body = re.search(r"axis_of\(\)\s*\{(.*?)\n\}", text, re.S)
    if not body:
        sys.exit("run-triggers.sh no longer defines axis_of")
    axis = {}
    for arms_, name in re.findall(r"^\s*((?:CRASH-\d+\|?)+)\)\s*echo (\w+)", body.group(1), re.M):
        for case in arms_.split("|"):
            if case:
                axis[case] = name
    if "heap" not in axis.values() or "temporal" not in axis.values():
        sys.exit("axis_of parsed without both axes -- its format changed")
    return axis


def read_reachable(text):
    """The 24 cases the physical suite actually runs: BLIND_REACHABLE plus the pureext list."""
    blind = re.search(r"BLIND_REACHABLE=\((.*?)\)", text, re.S)
    if not blind:
        sys.exit("run-triggers.sh no longer declares BLIND_REACHABLE")
    cases = re.findall(r"CRASH-\d+", blind.group(1))
    pureext = re.search(r"pureext\)\s*\n(.*?);;", text, re.S)
    if not pureext:
        sys.exit("run-triggers.sh no longer has a pureext suite")
    cases += re.findall(r"CRASH-\d+", pureext.group(1))
    out = sorted(set(cases))
    if len(out) < 20:
        sys.exit(f"only {len(out)} reachable cases parsed from the suite -- refusing to run a "
                 f"silently truncated denominator")
    return out


def read_copycases(text):
    """Cases whose over-read is performed BY a shared copy primitive (COPYCASE in the suite).

    A fault inside memcpy is attributable to no case by its site -- and it is also the shape of
    a bound the port itself got wrong, which is how _erealloc's narrow-pointer bug once scored
    as a catch. These cases are therefore attributed at the ALLOCATOR BOUNDARY instead, by the
    audit arm, never by where the pc landed.
    """
    block = re.search(r"declare -A COPYCASE=\((.*?)\)", text, re.S)
    if not block:
        sys.exit("run-triggers.sh no longer declares COPYCASE")
    return dict(re.findall(r"\[(CRASH-\d+)\]=(\w+)", block.group(1)))


# php_fault_report's word, as zc_audit_src packs it (php_capstone_zend_hier.c:117):
#   0xE3 | who | requested<<24 | available<<8 | overrun
AUDIT_MARKER = 0xE3
AUDIT_LINE = re.compile(r"PHP_VIRTUAL FAULT_REPORT 0x([0-9a-f]{16})")


def decode_audit(text):
    """(who, requested, available, overrun) from an audit run's output, or None."""
    m = AUDIT_LINE.search(text or "")
    if not m:
        return None
    v = int(m.group(1), 16)
    if (v >> 56) & 0xFF != AUDIT_MARKER:
        return None
    return ((v >> 48) & 0xFF, (v >> 24) & 0xFFFF, (v >> 8) & 0xFFFF, v & 0xFF)


def run_audit(args, cases, manifest):
    """One boot of the audit image. Returns {case: (who, requested, available, overrun)}.

    The audit arm is diagnostic and never scored: it halts at the first overrun, so a case that
    reaches its defect at all cannot also complete here.
    """
    runs = [(c, f"{args.audit.name} {manifest[c]['mode']} {c} < /mnt/vm/{c}.php") for c in cases]
    record = args.out / "audit-run.json"
    rc, rec = run_guest(args, args.audit, runs, record, args.out / "audit-serial.log")
    out = {}
    for case in cases:
        got = decode_audit("\n".join(rec["output"].get(case, [])))
        if got:
            out[case] = got
            print(f"    audit {case}: requested {got[1]} from a source authorising {got[2]}"
                  f" -- overrun {got[3]} byte(s)")
        else:
            print(f"    audit {case}: no overrun reported (exit={rec['exits'].get(case)})")
    return out


class Sites:
    """pc -> file:line, for the attribution the physical suite makes on EXPECT_FILE."""

    def __init__(self, llvm_bin, image):
        self.bin, self.image = Path(llvm_bin), Path(image)
        self.symbols = verdicts.Symbols(llvm_bin, image)

    def locate(self, pc, code_start):
        static = pc - code_start + self.symbols.base
        out = subprocess.run([str(self.bin / "llvm-symbolizer"), "--obj", str(self.image),
                              "--functions=short", f"{static:#x}"],
                             capture_output=True, text=True).stdout.splitlines()
        func = out[0].strip() if out else ""
        where = out[1].strip() if len(out) > 1 else ""
        return func, where


def run_guest(args, image, runs, record, serial):
    cmd = [sys.executable, str(HERE / "run-virtual.py"),
           "--qemu", str(args.qemu), "--images", str(args.images), "--adapter", str(args.adapter),
           "--image", str(image), "--record", str(record), "--serial-log", str(serial),
           "--timeout", str(args.timeout)]
    for name, argv in runs:
        cmd += ["--run", f"{name}={argv}"]
    for case in args.cases:
        cmd += ["--extra", str(TRIGGERS / f"{case}.php")]
    print("  " + " ".join(cmd[:6]) + " ...", flush=True)
    rc = subprocess.call(cmd)
    if not record.is_file():
        sys.exit(f"run-virtual.py wrote no record (rc={rc}); nothing to judge")
    return rc, json.loads(record.read_text())


# The line the control prints immediately before the access under test. A control that faulted
# WITHOUT printing it did not reach its own probe, and must not be scored as the bound firing.
CONTROL_AT_PROBE = "now reading index 9"


def read_control(rec):
    """What the arm's control actually did -- and WHERE, which is the half that matters.

    A control is only evidence about the bound if it reached the access under test. On the
    first run of the exact arm it faulted during zend_startup, at zend_alloc.c:404, long
    before its probe -- and a bare `fault if there was a fault` test scored that as the arm
    behaving exactly as declared, which is how a broken arm comes to look like a controlled
    one. `none` sends it to the judge as control-failed instead, with the site in evidence.
    """
    exits, chunks = rec["exits"], rec["output"]
    text = "\n".join(chunks.get(CONTROL_RUN, []))
    fault = verdicts.domain_fault(text, None)
    at_probe = CONTROL_AT_PROBE in text
    if fault and at_probe:
        observed, evidence = "fault", f"cause={fault.cause} pc={fault.pc:#x} at its own probe"
    elif fault:
        observed = "none"
        evidence = (f"cause={fault.cause} pc={fault.pc:#x} BEFORE the probe -- the control died "
                    f"in setup, so it says nothing about the bound")
    elif exits.get(CONTROL_RUN) == 0 and "CONTROL completed" in text:
        observed, evidence = "complete", "ran past the over-read and completed"
    else:
        observed = "none"
        evidence = f"exit={exits.get(CONTROL_RUN)}, no fault and no completion marker"
    return verdicts.Control(name=CONTROL_RUN, observed=observed, evidence=evidence)


def observe(case, arm, image_sha, exits, chunks, sites, expect, control, audit=None):
    """One Observation. Facts only -- what the judge needs and nothing it should infer."""
    text = "\n".join(chunks.get(case, []))
    rc = exits.get(case)
    obs = verdicts.Observation(case=case, arm=arm, image_sha256=image_sha, controls=[control])

    if rc is None:
        obs.infra = "infra"
        obs.notes = "the run produced no exit marker"
        return obs

    # REACHED means the engine got as far as evaluating this case's trigger. Everything before
    # that line is setup -- zend_startup, the compiler, the executor, registering functions --
    # and a fault there is a port artefact, not a reading, which is exactly what the judge's
    # `setup-fault` is for.
    obs.reached = "evaluating the trigger" in text
    obs.reach_evidence = "stage line 'evaluating the trigger' printed"
    obs.completed = "PHP_VIRTUAL_DONE" in text and rc == 0

    fault = verdicts.domain_fault(text, sites.symbols)
    if fault is not None:
        obs.fault = fault
        m = re.search(r"code=(0x[0-9a-f]+)-", text)
        func, where = sites.locate(fault.pc, int(m.group(1), 16)) if m else ("", "")
        obs.notes = f"{func} at {where}" if where else obs.notes
        # The physical suite attributes on the FILE the fault lands in (EXPECT_FILE); same
        # criterion here, so the two columns are comparable.
        if expect and expect in where:
            obs.attribution = "function"
            obs.attribution_evidence = f"fault in {expect} ({func} at {where}), as declared"
        elif audit:
            # The over-read was performed BY a copy primitive, so the pc attributes nothing.
            # The audit arm asked the question the pc cannot: did the CALLER request more than
            # its source capability authorises?
            who, requested, available, overrun = audit
            obs.attribution = "probe"
            obs.attribution_evidence = (
                f"fault at {where or hex(fault.pc)} is inside the copy primitive, which "
                f"attributes nothing by site; the audit arm reports the caller requesting "
                f"{requested} bytes from a source authorising {available} -- an overrun of "
                f"{overrun} (zc_audit_src who={who})")
        else:
            obs.attribution_evidence = (f"fault at {where or hex(fault.pc)} in "
                                        f"{func or fault.symbol or '?'}, not in {expect}")
    elif rc not in (0, None):
        # A nonzero exit with no fault line: the launcher or the image failed, and that is not
        # a reading either way.
        obs.infra = "infra"
        obs.notes = f"exit {rc} with no domain fault reported"
    return obs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qemu", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True, help="php-control.dom (rounded bounds)")
    p.add_argument("--fault", type=Path, required=True, help="php-fault.dom (exact bounds)")
    p.add_argument("--audit", type=Path,
                   help="php-audit.dom (-DZEND_CAP_ESTRNDUP_AUDIT): the diagnostic arm that "
                        "attributes an over-read performed by a shared copy primitive")
    p.add_argument("--out", type=Path, required=True, help="directory for the result bundles")
    p.add_argument("--case", action="append", dest="only", metavar="CRASH-nnn",
                   help="restrict to these cases (default: every reachable spatial case)")
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--from-record", action="store_true",
                   help="re-judge the runs already stored in --out instead of booting")
    args = p.parse_args()

    text = suite_text()
    expect_file = read_expect_file(text)
    axis = read_axis(text)
    reachable = read_reachable(text)
    manifest = json.loads((TRIGGERS / "manifest.json").read_text())["cases"]

    spatial = [c for c in reachable if axis.get(c, "heap") == "heap"]
    off_axis = len(reachable) - len(spatial)      # counted BEFORE --case narrows the run
    missing = [c for c in spatial if c not in manifest]
    if missing:
        sys.exit(f"no trigger file for {missing}; run extract-triggers.py")
    if args.only:
        unknown = [c for c in args.only if c not in spatial]
        if unknown:
            sys.exit(f"{unknown} are not reachable spatial cases; spatial = {spatial}")
        spatial = [c for c in spatial if c in args.only]
    args.cases = spatial

    # THE PAIR MUST NOT BE ONE IMAGE TWICE. The physical suite refuses byte-identical arms for
    # the same reason: a pair built the same way reports a clean sweep having varied nothing.
    sha = {arm: verdicts.sha256(getattr(args, kind)) for arm, kind in ARMS.items()}
    if len(set(sha.values())) != len(sha):
        sys.exit("the two arms are byte-identical -- -DZEND_CAP_BOUNDS_REAL_SIZE did nothing")

    print(f"reachable={len(reachable)} spatial={len(spatial)}: {' '.join(spatial)}")
    print(f"not in scope on this axis: {off_axis} temporal cases "
          f"(they need the allocator seam, not this pair)"
          + (f"; --case narrowed this run to {len(spatial)}" if args.only else ""))

    copycases = read_copycases(text)
    audit_cases = [c for c in spatial if c in copycases]

    llvm_bin = os.environ.get("CAPSTONE_LLVM_BIN")
    if not llvm_bin:
        sys.exit("CAPSTONE_LLVM_BIN is unset; source capstone/tests/capstone-test-env.sh")
    args.out.mkdir(parents=True, exist_ok=True)

    platform = {"compiler": verdicts.sha256(Path(llvm_bin) / "clang"),
                "qemu": verdicts.sha256(args.qemu),
                "firmware": verdicts.sha256(args.images / "fw_jump.elf"),
                "kernel": verdicts.sha256(args.images / "Image"),
                "runner": verdicts.sha256(HERE / "run-virtual.py")}

    # THE AUDIT ARM FIRST, so its evidence exists when the scored arms are judged. It is its
    # own image and its own boot; a case it cannot attribute stays unattributed rather than
    # being credited to the nearest plausible site.
    audit = {}
    if audit_cases:
        if args.audit and not args.from_record:
            print(f"\n==> audit arm ({args.audit.name}) for {' '.join(audit_cases)}", flush=True)
            audit = run_audit(args, audit_cases, manifest)
        else:
            why = "no --audit image given" if not args.audit else "--from-record"
            print(f"\n==> audit arm NOT run ({why}); {' '.join(audit_cases)} can only be "
                  f"attributed by site, which for a copy-performed over-read means not at all")

    worst = 0
    for arm, kind in ARMS.items():
        image = getattr(args, kind)
        print(f"\n==> {arm}  ({image.name})", flush=True)
        # The control runs FIRST, so its reading exists even if a later run takes the guest
        # down. Each case is its own capstone-vexec process, so a fault ends the child only.
        runs = [(CONTROL_RUN, f"{image.name} control {CONTROL_RUN}")]
        for case in spatial:
            mode = manifest[case]["mode"]
            runs.append((case, f"{image.name} {mode} {case} < /mnt/vm/{case}.php"))
        record = args.out / f"{arm}-run.json"
        if args.from_record:
            # Re-judge a stored run. judge() is a pure function of (Observation, arm), so a
            # change to the rules or to how a transcript is read can be applied to runs already
            # on disk -- which is most of the reason for using the shared judge at all.
            if not record.is_file():
                sys.exit(f"--from-record: no stored run at {record}")
            rc, rec = 0, json.loads(record.read_text())
            # THE STORED RUN MUST BE THIS BINARY'S. capstone-exec prints the image's sha256 on
            # every fault line, so a record produced by a different build can be refused
            # instead of silently re-judged against whatever is on disk now -- the stale
            # artifact being read as the current one.
            stored = {m for chunk in rec["output"].values()
                      for line in chunk
                      for m in re.findall(r"sha256=([0-9a-f]{64}) image=", line)}
            if stored and sha[arm] not in stored:
                sys.exit(f"--from-record: {record.name} was produced by image "
                         f"{sorted(stored)[0][:16]}..., but {image.name} is now "
                         f"{sha[arm][:16]}...; rebuild or re-run rather than re-judging")
            print(f"    re-judging the stored run in {record.name} (no boot)"
                  + (f", image {sorted(stored)[0][:16]}... confirmed" if stored else ""))
        else:
            rc, rec = run_guest(args, image, runs, record, args.out / f"{arm}-serial.log")

        exits, chunks = rec["exits"], rec["output"]
        control = read_control(rec)
        print(f"    control {CONTROL_RUN}: observed={control.observed} ({control.evidence})")

        sites = Sites(llvm_bin, image)
        rows = []
        for case in spatial:
            obs = observe(case, arm, sha[arm], exits, chunks, sites,
                          expect_file.get(case, "url.c"), control, audit.get(case))
            if not rec["completed"]:
                # The runner did not get through its own script: every row from this boot is
                # infrastructure, whatever the individual markers say.
                obs.infra = "infra"
                obs.notes = (obs.notes + "; " if obs.notes else "") + \
                            "the runner did not complete its guest script"
            verdict = verdicts.judge(obs, verdicts.load_arms()[arm])
            rows.append((obs, verdict))
            site = obs.notes.split(";")[0]
            print(f"    {case:<11} {verdict[0]:<10} {verdict[1] or ''} {site}")
        bundle = verdicts.write_bundle(args.out / arm, "php-5.0.0-engine-repros", arm, rows,
                                       {"platform": platform,
                                        "judge": verdicts.sha256(TOOLS / "verdicts.py"),
                                        "image": sha[arm], "axis": "spatial",
                                        "phase": "A (hier mode; the temporal axis needs the seam)"})
        print(f"    bundle: {args.out / arm}  tally={bundle['tally']}")
        worst = max(worst, rc)
    return worst


if __name__ == "__main__":
    sys.exit(main())
