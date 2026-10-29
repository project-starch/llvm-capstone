#!/usr/bin/env python3
"""Run a PHP bug corpus on one or more arms and write a verdicts-v1 bundle per arm.

WHY THIS EXISTS BESIDE ports/php/app/run-realphp.py. That runner is the PORT's instrument: its
population and its per-case oracle are recomputed from the EXTERNAL php-5.0.0 bug corpus's own
result files, which is right for the port's qualification and wrong for a corpus in this tree --
here the authority has to be the case directories, because that is what a reader can audit. So
this runner reads `case.json` for the trigger, the fault sites and the oracle, and reuses
everything else rather than copying it: run-virtual.py drives the guest, run-realphp.py's own
sanity script is the control, and tools/verdicts.py is the judge.

ONE GUEST BOOT PER ARM, and the arms run in the order given. `virtual-malloc` is also the reach
oracle for the arms after it (see `reached` below), so pass it first when you pass more than one.

    run-php-corpus.py --corpus capstone/bug-corpora/php/zend-alloc-repros \\
        --arm virtual-malloc --arm virtual-nested-pools \\
        --qemu <qemu-system-riscv64> --images <dir> --adapter <dir> \\
        --image <php-real-seam.dom> --probe <probe-seam.dom> \\
        --llvm-bin <llvm/bin> --out <corpus>/results/<stamp>

    run-php-corpus.py --self-test     # the judge wiring, against synthetic transcripts

HOW `reached` IS ESTABLISHED, because a silence that is not reached is not a MISSED. Two routes,
the same two run-realphp.py uses and for the same reasons; each Observation records which one
licensed it:

  A. the arm's `revocation-sanity` control COMPLETED (so the image starts PHP and runs a script
     to its end on this arm), AND the baseline arm ran this very case to its normal end, AND the
     corpus's own host record says the case completes without a sanitizer. Then a fault here is
     after startup and the trigger is live in these bytes.
  B. the fault landed in one of the case's declared `fault_sites`. That line cannot execute
     unless the trigger ran, so the fault is its own reach evidence. This route exists because a
     case the platform catches even on the baseline arm has no completing control run by
     construction, and route A would make it un-reached on every arm.

A fault inside the seam's own sources is never a catch: it is the port faulting, not the program.
"""
import argparse, importlib.util, json, os, re, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PORT = REPO / "capstone" / "ports" / "php" / "app"
sys.path.insert(0, str(REPO / "capstone" / "bug-corpora" / "tools"))
import verdicts                                                          # noqa: E402

SANITY = "revocation-sanity"
BOUND_PROBE = "seam-bound-probe"
SANITY_SCRIPT = "phpb-sanity.php"
TIMEOUT_MARK = "VM_EXIT_TIMEOUT:"
# A fault in the seam's own translation units is a port artefact, not a reading about the program.
SEAM_OWN = ("zend_cap_", "zend-cap-", "__capstone_sublet", "php_virtual_lower")
BASELINE_ARM = "virtual-malloc"


def port_module(name):
    """Import a hyphenated script from the port rather than copying what it holds."""
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), PORT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- the corpus ----------------------------------------------------------------------------

def load_corpus(path):
    """(declaration, [(directory name, case.json)]), in case order."""
    decl = json.loads((path / "corpus.json").read_text())
    if decl.get("case_schema") != "script-trigger":
        sys.exit(f"{path}: this runner reads script-trigger corpora, not {decl.get('case_schema')!r}")
    cases = []
    for d in sorted(path.glob("[0-9][0-9]_*")):
        if d.is_dir():
            cases.append((d.name, json.loads((d / "case.json").read_text())))
    if len(cases) != decl["cases"]:
        sys.exit(f"{path}: corpus.json says {decl['cases']} cases, the tree has {len(cases)}")
    return decl, cases


def host_record(path):
    """{case directory: the host reading}, from the corpus's own host-asan bundle.

    `pristine` is what route A needs: whether the case completes with no sanitizer and no
    protection. It is a column of the committed matrix.tsv, so the runner cannot invent it.
    """
    bundles = sorted((path / "results").glob("*-host-asan"))
    if not bundles:
        sys.exit(f"{path}: no host-asan bundle under results/, so `reached` has no oracle")
    rows, head = {}, None
    for line in (bundles[-1] / "matrix.tsv").read_text().splitlines():
        cells = line.split("\t")
        if head is None:
            head = cells
            continue
        rows[cells[head.index("source_id")]] = dict(zip(head, cells))
    return bundles[-1], rows


def designed_exit(host_row):
    """The exit code the host's STOCK-allocator build ends this case with, when it ends by
    exiting rather than by crashing.

    "Ran to its normal end" cannot mean rc == 0 for every trigger. One of these cases ends in an
    uncaught exception BY DESIGN -- its own error handler throws -- and PHP exits 255 there on
    every platform, the host's stock build included (matrix.tsv: EXIT:255). Demanding 0 made the
    judge call that case's designed ending "neither a fault nor a clean exit" and refuse a reading
    that was perfectly good. The reference is the stock column, not the detector one, because the
    guest also runs Zend's allocator as shipped.
    """
    m = re.match(r"EXIT:(\d+)$", (host_row or {}).get("stockalloc", ""))
    return int(m.group(1)) if m else None


def ran_to_its_end(rc, host_row):
    return rc == 0 or (designed_exit(host_row) is not None and rc == designed_exit(host_row))


def host_ends_normally(host_row):
    """Does the host's unsanitised, unprotected build end this case WITHOUT crashing?

    Route A's oracle. It must speak the same language as ran_to_its_end: a case whose designed
    ending is an uncaught exception ends 255 on the host too, so demanding pristine == OK refused
    reach for an ending both sides agree about. A pristine SIGNAL:n is a crash, and says nothing
    about whether a fault here came after startup -- those keep no route A.
    """
    p = (host_row or {}).get("pristine", "")
    return p == "OK" or re.match(r"EXIT:\d+$", p) is not None


def case_host_row(case, rows):
    """The host row for a case, found by the source id its note records."""
    m = re.search(r"source corpus id (CRASH-\d+)", case.get("note", ""))
    return rows.get(m.group(1)) if m else None


# ---- one arm -------------------------------------------------------------------------------

def stage(scratch, corpus, cases, sanity_text):
    scratch.mkdir(parents=True, exist_ok=True)
    for name, case in cases:
        src = corpus / name / case["trigger"]
        if not src.is_file():
            sys.exit(f"{name}: no trigger at {src}")
        (scratch / f"{name}.php").write_bytes(src.read_bytes())
    (scratch / SANITY_SCRIPT).write_text(sanity_text)
    (scratch / END_FILE).write_text(
        "<?php\n"
        "// auto_append_file: PHP runs this after the script body and before request shutdown.\n"
        f"fwrite(STDERR, \"{END_MARK}\\n\");\n")
    (scratch / START_FILE).write_text(
        "<?php\n"
        "// auto_prepend_file: PHP prints this before the trigger's body, so every case carries\n"
        "// the same reach evidence whether or not the trigger itself prints.\n"
        f"fwrite(STDERR, \"{START_MARK}\\n\");\n")


def guest(args, mode, names, wants_probe, record, serial):
    """One boot: the sanity control, the bound probe if the configuration declares it, the cases."""
    img = args.image.name
    runs = [(SANITY, f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{SANITY_SCRIPT}")]
    if wants_probe:
        runs.append((BOUND_PROBE, f"{args.probe.name} 7"))
    for name in names:
        runs.append((name, f"PHP_CAP_MODE={mode} {img} -d memory_limit=512M /mnt/vm/{name}.php"))
    cmd = [sys.executable, str(PORT / "run-virtual.py"),
           "--qemu", str(args.qemu), "--images", str(args.images), "--adapter", str(args.adapter),
           "--image", str(args.image), "--record", str(record), "--serial-log", str(serial),
           "--timeout", str(args.timeout), "--run-timeout", str(args.run_timeout)]
    if wants_probe:
        cmd += ["--image", str(args.probe)]
    cmd += ["--extra", str(args.scratch / SANITY_SCRIPT)]
    for name in names:
        cmd += ["--extra", str(args.scratch / f"{name}.php")]
    for name, argv in runs:
        cmd += ["--run", f"{name}={argv}"]
    rc = subprocess.call(cmd)
    if not record.is_file():
        sys.exit(f"run-virtual.py wrote no record (rc={rc}); nothing to judge")
    return json.loads(record.read_text())


def control_state(rec, name):
    """"fault", "complete" or "none" for one control run, from the transcript it produced."""
    if name not in rec["exits"]:
        return "none"
    text = "\n".join(rec["output"].get(name, []))
    if TIMEOUT_MARK in text:
        return "none"
    if verdicts.domain_fault(text) is not None:
        return "fault"
    return "complete" if rec["exits"][name] == 0 else "none"


def observe(name, case, rec, arm, symbols, host_row, baseline_completed, controls, image_sha):
    """One run of one case on one arm, as facts. No verdict is formed here."""
    obs = verdicts.Observation(case=name, arm=arm, controls=controls, image_sha256=image_sha)
    if name not in rec["exits"]:
        obs.infra = "infra"
        obs.notes = "the run produced no VM_EXIT marker"
        return obs
    text = "\n".join(rec["output"].get(name, []))
    if TIMEOUT_MARK in text:
        obs.infra = "infra"
        obs.notes = f"killed by the per-run watchdog after {rec.get('run_timeout', '?')}s"
        return obs
    fault = verdicts.domain_fault(text, symbols)
    sites = case.get("fault_sites", [])
    if fault is not None:
        obs.fault = fault
        symbol = fault.symbol or ""
        if any(symbol.startswith(p) for p in SEAM_OWN):
            obs.infra = "infra"
            obs.notes = (f"the fault is in the seam's own code ({symbol}), so it is the port "
                         f"faulting rather than a reading about the program")
            return obs
        if symbol in sites:
            obs.reached = True
            obs.reach_evidence = (f"route B: the fault is at {symbol}, one of this case's declared "
                                  f"fault_sites, and that line cannot run unless the trigger did")
            obs.attribution = "function"
            obs.attribution_evidence = (f"{symbol} is declared in case.json fault_sites, from the "
                                        f"host-ASan access stack recorded before any Capstone run")
        else:
            obs.notes = f"fault in {symbol or '?'}, which this case does not declare as a site"
    else:
        rc = rec["exits"][name]
        obs.completed = ran_to_its_end(rc, host_row)
        if obs.completed and rc != 0:
            obs.notes = (f"exit {rc}, which is how the host's stock build ends this case too "
                         f"(its designed ending), so it ran to its end")
        elif not obs.completed:
            obs.notes = f"no domain fault and exit {rc}"
    if not obs.reached:
        pristine = (host_row or {}).get("pristine", "")
        sanity_ok = any(c.name == SANITY and c.observed == "complete" for c in controls)
        if sanity_ok and baseline_completed and host_ends_normally(host_row):
            obs.reached = True
            obs.reach_evidence = (f"route A: {SANITY} completed on this arm, the baseline arm ran "
                                  f"this case to its normal end, and the host record ends it the "
                                  f"same way with no sanitizer (pristine={pristine})")
        elif obs.completed and not obs.reached:
            obs.notes = ((obs.notes + "; ") if obs.notes else "") + (
                f"silent, but reach is not established: sanity={'ok' if sanity_ok else 'no'}, "
                f"baseline completed={baseline_completed}, host pristine={pristine or 'unrecorded'}")
    return obs


# ---- the CheriBSD arm ----------------------------------------------------------------------

CHERI_ARM = "cheribsd-revocation"
CHERI_CONTROL = "bounds-malloc"
# PHP prints this itself, before the trigger's body, from an auto_prepend_file. It is the reach
# evidence that does not depend on whether a trigger happens to print: a fault after it is after
# startup. Without it a case that faults silently reads as setup-fault, which blames the harness.
START_FILE = "phpb-start.php"
START_MARK = "PHPB_TRIGGER_START"
# The other end of the same idea: PHP runs an auto_append_file after the script BODY, before the
# request shuts down and the allocator's cache is drained back to libc. A fault with this marker
# present came after the body; one without it came inside it. Without the pair, a fault that the
# eager configuration produces cannot be placed, and "revocation caught the defect" and "the
# drain tripped over it at shutdown" look identical.
END_FILE = "phpb-end.php"
END_MARK = "PHPB_TRIGGER_END"
SIGPROT = 34                      # FreeBSD's capability-protection signal; a shell reports 128+34
SIGABRT = 6                       # the allocator's own check raises it; see controls/abort-probe.c
WATCHDOG_RC = 124                 # timeout(1)


def scratch_for(args, arm):
    """Where a run's RAW transcripts go -- never the corpus tree.

    SCHEMA.md: results are summaries, never captures. A serial log in results/ is also a
    release-blocking hit for precommit-scan.sh, which reads the guest's own shell prompt as a
    user@host build string. So the bundle lands in --out and everything raw lands here.
    """
    root = Path(os.environ.get("CAPSTONE_TMP_ROOT", "/tmp/capstone")) / "php-corpus-runs"
    # The CORPUS is in the path, not just the stamp: two corpora run on the same day with the
    # same --out stamp, and without this the second run overwrote the first's transcripts --
    # which is how the first ktrace of an aborting case had to be re-run.
    d = root / Path(args.corpus).name / Path(args.out).name / arm
    d.mkdir(parents=True, exist_ok=True)
    return d


def needs_extensions(case):
    """Does this case need bundled extensions the --disable-all build does not carry?

    The fidelity says so in the case's own words, written when the corpus was built from the
    source corpus's needs_build column. Reading it here keeps one statement of the fact instead
    of a list in the runner that would drift from the cases.
    """
    return "needs bundled extensions" in case.get("fidelity", "")


# The three CheriBSD configurations. `default` is the arm (the platform's own setting) and the
# only one a verdict may come from. `off` is the pair that attributes a fault, since a purecap
# SIGPROT carries no symbol. `eager` sweeps on EVERY free and is CONTEXT ONLY: it is an upper
# bound that faults on things nothing is meant to catch, so it may never decide a verdict -- what
# it is for is telling a quarantined-but-never-reissued chunk (SCHEMA.md's NOT-REISSUED) apart
# from a genuine miss, which the default configuration alone cannot do because neither reports.
CHERI_CONFIGS = {
    "default": (1, 0),
    "off":     (0, 0),
    "eager":   (1, 1),
}


def cheribsd_guest(args, cases, out, config):
    """One boot of the CheriBSD guest at one revocation configuration."""
    import subprocess
    revocation, every_free = CHERI_CONFIGS[config]
    runs = [(CHERI_CONTROL, "/tmp/bounds-malloc")]
    for name, case in cases:
        php = "/tmp/php-fullext" if (needs_extensions(case) and args.php_fullext) else "/tmp/php"
        runs.append((name, f"{php} -n -d memory_limit=512M "
                           f"-d auto_prepend_file=/mnt/php/{START_FILE} "
                           f"-d auto_append_file=/mnt/php/{END_FILE} /mnt/php/{name}.php"))
    raw = scratch_for(args, "cheribsd-revocation")
    record = raw / f"cheribsd-{config}.json"
    cmd = [sys.executable, str(REPO / "capstone/ports/php/cheribsd/run-cheribsd-cases.py"),
           "--stage", str(args.scratch), "--record", str(record),
           "--log", str(record.with_suffix(".serial.log")),
           "--copy", "bounds-malloc",
           "--run-timeout", str(args.eager_timeout if config == "eager" else args.run_timeout),
           "--sysctl", f"security.cheri.runtime_revocation_default={revocation}",
           "--sysctl", f"security.cheri.runtime_revocation_every_free_default={every_free}"]
    if args.php_fullext:
        cmd += ["--copy", "php-fullext"]
    for name, argv in runs:
        cmd += ["--run", f"{name}={argv}"]
    if args.cheri_img:
        cmd += ["--img", str(args.cheri_img)]
    subprocess.call(cmd)
    if not record.is_file():
        sys.exit(f"run-cheribsd-cases.py wrote no record for the {config} configuration")
    return json.loads(record.read_text())


def cheri_outcome(rec, name, host_row=None):
    """(faulted, completed, infra, own_output) for one run of the CheriBSD guest."""
    if name not in rec["exits"]:
        return False, False, "the run produced no marker", []
    rc = rec["exits"][name]
    lines = [l for l in rec["output"].get(name, []) if l.strip()]
    own = [l for l in lines if "security exception" not in l.lower()
           and not l.startswith("CONTROL ") and "Protection fault" not in l
           and START_MARK not in l and END_MARK not in l]
    if rc == WATCHDOG_RC:
        return False, False, f"killed by the guest watchdog after {rec.get('run_timeout')}s", own
    faulted = rc == 128 + SIGPROT or any("security exception" in l.lower() for l in lines)
    return faulted, ran_to_its_end(rc, host_row), None, own


def run_cheribsd_arm(args, decl, cases, arm, rows, out):
    """The arm, plus its revocation-off twin -- the pair is what ties a fault to the defect."""
    spec = verdicts.load_arms()[decl["arm_configurations"][arm]]
    on = cheribsd_guest(args, cases, out, "default")
    off = cheribsd_guest(args, cases, out, "off")
    eager = cheribsd_guest(args, cases, out, "eager") if args.eager else None

    c_fault, _, c_infra, _ = cheri_outcome(on, CHERI_CONTROL)
    controls = [verdicts.Control(CHERI_CONTROL, "fault" if c_fault else ("none" if c_infra else "complete"),
                                 "a read 4096 bytes past a 32-byte malloc'd block")]
    rows_out = []
    for name, case in cases:
        obs = verdicts.Observation(case=name, arm=arm, controls=controls,
                                   image_sha256=verdicts.sha256(args.scratch / "php"))
        host = case_host_row(case, rows)
        faulted, completed, infra, own = cheri_outcome(on, name, host)
        off_fault, off_completed, _, _ = cheri_outcome(off, name, host)
        # REACH FIRST, AND INDEPENDENTLY OF THE OUTCOME. A run that printed the trigger's own
        # output ran the trigger, whether it then completed, faulted or aborted; deciding reach
        # inside the fault branch made an abort read as `not-reached`, which blames the harness
        # for a result the harness observed perfectly well.
        started = any(START_MARK in l for l in on["output"].get(name, []))
        def where(rec):
            """Where an eager fault lands -- but only when THIS RUN can say.

            The absence of the END marker means "faulted before request shutdown" only if the
            marker was installed at all. A record from a boot that predates it has no END line
            anywhere, so "inside the script body" would be printed for all 26 cases of a corpus
            whose placement was never measured -- a reading manufactured by a check that cannot
            produce the other answer. The positive control is the run itself: at least one case
            must have printed the marker.
            """
            if rec is None:
                return "unplaced: no eager run"
            if not any(END_MARK in l for ls in rec["output"].values() for l in ls):
                return ("unplaced: this boot printed no " + END_MARK + " for any case, so it "
                        "predates the marker and cannot place a fault")
            lines = rec["output"].get(name, [])
            if not any(END_MARK in l for l in lines):
                return "inside the script body"
            return "after the script body, in request shutdown"
        e_fault, _, _, _ = cheri_outcome(eager, name) if eager is not None else (False, 0, 0, 0)
        if own:
            obs.reached = True
            obs.reach_evidence = ("route B: the trigger printed its own output, so it ran: "
                                  + own[0][:80])
        elif started:
            obs.reached = True
            obs.reach_evidence = (f"route D: PHP printed {START_MARK} from auto_prepend_file, so "
                                  f"the interpreter started and entered the trigger")
        elif e_fault:
            obs.reached = True
            obs.reach_evidence = ("route C: the eager configuration faults on this case, and that "
                                  "access cannot execute unless the trigger ran")
        elif off_completed and host_ends_normally(host):
            obs.reached = True
            obs.reach_evidence = ("route A: the same binary with revocation off ran this case to "
                                  "its normal end, and the host record ends it the same way with "
                                  f"no sanitizer (pristine={(host or {}).get('pristine','?')})")
        if infra:
            obs.infra, obs.notes = "infra", infra
        elif faulted:
            obs.fault = verdicts.Fault(cause=SIGPROT, pc=0, symbol=None)
            if obs.reached:
                obs.attribution = "control"
                obs.attribution_evidence = (
                    "the paired run of the same binary with "
                    "security.cheri.runtime_revocation_default=0 " +
                    ("ALSO faulted, so bounds or another check fired rather than revocation"
                     if off_fault else "did not fault, so revocation is what fired"))
                if off_fault:
                    obs.notes = ("faults with revocation off as well; the fault is the defect's, "
                                 "but it is not evidence about revocation")
        else:
            obs.completed = completed
            if not completed:
                rc = on["exits"].get(name)
                # WHICH DEFENCE FIRED IS NOT A GUESS. ports/php/cheribsd/controls/abort-probe.c
                # reproduces this exact signature from a plain double free: SIGABRT raised by the
                # process on itself (ktrace: thr_kill, code=SI_LWP) with nothing written to
                # stderr. It is CheriBSD's ALLOCATOR consistency check, which any allocator could
                # carry -- not a capability fault -- so it is named here and never scored as a
                # catch on an arm that measures CHERI.
                why = ("ALLOCATOR-ABORT: CheriBSD's allocator aborted, the signature its own "
                       "double-free check produces (ports/php/cheribsd/results/"
                       "20261029-allocator-probe). Not a capability fault"
                       if rc == 128 + SIGABRT else f"exit {rc}")
                obs.notes = (f"neither a capability fault nor a clean exit -- {why}; with "
                             f"revocation off this case "
                             f"{'faulted' if off_fault else 'exited ' + str(off['exits'].get(name))}")
            elif not obs.reached:
                obs.notes = (f"silent, but reach is not established: host pristine="
                             f"{(host or {}).get('pristine', 'unrecorded')}")
        # QUARANTINED IS NOT THE SAME AS MISSED, AND NEITHER REPORTS. A chunk CheriBSD freed and
        # held in quarantine is never reissued, so the stale pointer reads the old bytes and the
        # run completes silently -- SCHEMA.md calls that reading NOT-REISSUED and counts it as
        # caught, shown apart. The default configuration cannot tell it from a genuine miss; the
        # eager configuration can, because sweeping on every free turns the same access into a
        # fault. It is recorded as a NOTE and never as the verdict: eager also faults on things
        # nothing is meant to catch, which is why it is context, not evidence.
        if eager is not None and not obs.fault:
            if e_fault:
                obs.notes = ((obs.notes + "; ") if obs.notes else "") + (
                    "NOT-REISSUED: no capability fault at the platform default, but the eager "
                    "configuration (runtime_revocation_every_free_default=1) faults on this case, "
                    f"so the stale pointer IS followed. The eager fault lands {where(eager)}, "
                    f"which is what tells a revoked object apart from the allocator's cache being "
                    f"drained back to libc at shutdown. Context, not a verdict: every-free also "
                    f"faults on accesses nothing is meant to catch")
        rows_out.append((obs, verdicts.judge(obs, spec)))
    inputs = {
        "configuration": decl["arm_configurations"][arm],
        "population": f"the {len(cases)} cases of {decl['program']}/{Path(args.corpus).name}",
        "platform": {"compiler": "the CheriBSD SDK clang that built ports/php/cheribsd",
                     "qemu": "cheri-qemu, ports/php/cheribsd/run-cheribsd-cases.py",
                     "firmware": "bbl purecap (the SDK's own)",
                     "kernel": "the CheriBSD purecap rootfs kernel",
                     "runner": str(Path(__file__).resolve().relative_to(REPO))},
        "revocation": {"default": on.get("sysctl"), "off": off.get("sysctl"),
                       "eager": eager.get("sysctl") if eager else "not run"},
        "boot": {k: on.get(k) for k in ("booted", "mounted", "completed", "panic")},
        "binaries": {"php": verdicts.sha256(args.scratch / "php"),
                     "php-fullext": (verdicts.sha256(args.scratch / "php-fullext")
                                     if args.php_fullext else "not staged"),
                     "which_cases_use_fullext": [n for n, c in cases if needs_extensions(c)]},
        "note": ("boots: the arm at the platform default, the same binary with revocation off, "
                 "and (when --eager) a third at every-free. The default is the only one a verdict "
                 "comes from; off is what attributes a fault, since a purecap SIGPROT carries no "
                 "symbol here; eager only separates a quarantined-but-never-reissued chunk from a "
                 "genuine miss, and is recorded in the notes rather than scored"),
    }
    verdicts.write_bundle(out / arm, Path(args.corpus).name, arm, rows_out, inputs)
    return {n: o.completed for (n, _), (o, _) in zip(cases, rows_out)}, rows_out


def run_arm(args, decl, cases, arm, rows, baseline):
    config = decl["arm_configurations"][arm]
    spec = verdicts.load_arms()[config]
    mode = spec.get("mode")
    if mode is None:
        sys.exit(f"{config}: arms.json records no PHP_CAP_MODE for this configuration")
    wants_probe = BOUND_PROBE in spec["controls"]
    if wants_probe and not args.probe:
        sys.exit(f"{config} declares the {BOUND_PROBE} control; pass --probe")
    names = [n for n, _ in cases]
    out = Path(args.out) / arm
    out.mkdir(parents=True, exist_ok=True)
    raw = scratch_for(args, arm)
    rec = guest(args, mode, names, wants_probe, raw / "run-virtual.json", raw / "serial.log")

    controls = [verdicts.Control(SANITY, control_state(rec, SANITY),
                                 "the benign workload every arm must run to its end")]
    if wants_probe:
        controls.append(verdicts.Control(BOUND_PROBE, control_state(rec, BOUND_PROBE),
                                         "probe-seam stage 7: the seam's own bound must fault"))
    symbols = verdicts.Symbols(args.llvm_bin, args.image) if args.llvm_bin else None
    image_sha = verdicts.sha256(args.image)

    rows_out, completed = [], {}
    for name, case in cases:
        obs = observe(name, case, rec, arm, symbols, case_host_row(case, rows),
                      baseline.get(name, False), controls, image_sha)
        completed[name] = obs.completed
        rows_out.append((obs, verdicts.judge(obs, spec)))
    inputs = {
        "configuration": config,
        "population": f"the {len(cases)} cases of {decl['program']}/{Path(args.corpus).name}",
        "platform": {
            "compiler": str(args.llvm_bin or "unrecorded"),
            "qemu": str(args.qemu),
            "firmware": "unrecorded",
            "kernel": str(args.images),
            "runner": str(Path(__file__).resolve().relative_to(REPO)),
        },
        "image_sha256": image_sha,
        "probe_sha256": verdicts.sha256(args.probe) if wants_probe else None,
        "php_cap_mode": mode,
        "boot": {k: rec[k] for k in ("booted", "reached_script", "completed", "panic")},
        "note": ("one boot per arm; every run in it is one capstone-vexec process at this arm's "
                 "PHP_CAP_MODE. The trigger files are the corpus's own, staged verbatim"),
    }
    verdicts.write_bundle(out, Path(args.corpus).name, arm, rows_out, inputs)
    return completed, rows_out


# ---- the instrument's own control ----------------------------------------------------------

SYNTHETIC = {
    "exits": {SANITY: 0, BOUND_PROBE: 134, "01_deadbeef0000_caught": 134,
              "02_deadbeef0001_silent": 0, "03_deadbeef0002_elsewhere": 134},
    "output": {
        SANITY: ["sanity done"],
        BOUND_PROBE: ["domain fault cause=24 pc=0x80000100 address=0x0 foo code=0x80000000-0x80010000"],
        "01_deadbeef0000_caught":
            ["domain fault cause=28 pc=0x80000200 address=0x0 x code=0x80000000-0x80010000"],
        "02_deadbeef0001_silent": ["done"],
        "03_deadbeef0002_elsewhere":
            ["domain fault cause=28 pc=0x80000300 address=0x0 x code=0x80000000-0x80010000"],
    },
    "booted": True, "reached_script": True, "completed": True, "panic": False,
}


class FakeSymbols:
    MAP = {0x80000200: "zend_assign_to_variable", 0x80000300: "zend_startup"}

    def lookup(self, pc, code_start):
        return self.MAP.get(pc), 0


def self_test():
    """A judge that cannot say MISSED proves nothing by saying CAUGHT, so both are exercised."""
    spec = verdicts.load_arms()["php-real-exact"]
    controls = [verdicts.Control(SANITY, "complete"), verdicts.Control(BOUND_PROBE, "fault")]
    cases = {
        "01_deadbeef0000_caught": {"fault_sites": ["zend_assign_to_variable"], "note": ""},
        "02_deadbeef0001_silent": {"fault_sites": ["zend_hash_destroy"],
                                   "note": "source corpus id CRASH-004"},
        # reached by route A, and then faults somewhere it did not declare
        "03_deadbeef0002_elsewhere": {"fault_sites": ["zend_assign_to_variable"],
                                      "note": "source corpus id CRASH-005"},
        # the same fault with no reach evidence at all: a fault in startup, not a reading
        "04_deadbeef0003_setup": {"fault_sites": ["zend_assign_to_variable"], "note": ""},
        # ends by exiting 255, which is how the host's stock build ends it too: a reading, not
        # "neither a fault nor a clean exit"
        "05_deadbeef0004_designed_end": {"fault_sites": ["zend_hash_destroy"],
                                         "note": "source corpus id CRASH-069"},
    }
    SYNTHETIC["exits"]["04_deadbeef0003_setup"] = 134
    SYNTHETIC["exits"]["05_deadbeef0004_designed_end"] = 255
    SYNTHETIC["output"]["05_deadbeef0004_designed_end"] = ["Fatal error: Uncaught exception"]
    SYNTHETIC["output"]["04_deadbeef0003_setup"] = SYNTHETIC["output"]["03_deadbeef0002_elsewhere"]
    rows = {"CRASH-004": {"pristine": "OK"}, "CRASH-005": {"pristine": "OK"},
            "CRASH-069": {"pristine": "EXIT:255", "stockalloc": "EXIT:255"}}
    got = {}
    for name, case in cases.items():
        obs = observe(name, case, SYNTHETIC, "virtual-nested-pools", FakeSymbols(),
                      case_host_row(case, rows), True, controls, "0" * 64)
        got[name] = verdicts.judge(obs, spec)[:2]
    expect = {"01_deadbeef0000_caught": (verdicts.CAUGHT, None),
              "02_deadbeef0001_silent": (verdicts.MISSED, None),
              "03_deadbeef0002_elsewhere": (verdicts.NO_READING, "unattributed"),
              "04_deadbeef0003_setup": (verdicts.NO_READING, "setup-fault"),
              "05_deadbeef0004_designed_end": (verdicts.MISSED, None)}
    bad = {k: (got[k], expect[k]) for k in expect if got[k] != expect[k]}
    # and a control that misbehaves must void the whole arm, not just fail to fire
    broken = observe("01_deadbeef0000_caught", cases["01_deadbeef0000_caught"], SYNTHETIC,
                     "virtual-nested-pools", FakeSymbols(), None, True,
                     [verdicts.Control(SANITY, "none"), verdicts.Control(BOUND_PROBE, "fault")],
                     "0" * 64)
    if verdicts.judge(broken, spec)[:2] != (verdicts.NO_READING, "control-failed"):
        bad["control-failed"] = (verdicts.judge(broken, spec)[:2], (verdicts.NO_READING, "control-failed"))
    for k, (g, e) in bad.items():
        print(f"FAIL {k}: got {g}, expected {e}")
    print("self-test:", "FAILED" if bad else "every arm of the judge fired as declared")
    return 1 if bad else 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--corpus", type=Path)
    p.add_argument("--arm", action="append", default=[])
    p.add_argument("--out", type=Path)
    p.add_argument("--qemu", type=Path)
    p.add_argument("--images", type=Path)
    p.add_argument("--adapter", type=Path)
    p.add_argument("--image", type=Path, help="php-real-seam.dom")
    p.add_argument("--probe", type=Path, help="probe-seam.dom, for the bound control")
    p.add_argument("--llvm-bin", type=verdicts.llvm_bin_dir,
                   help="llvm-nm and llvm-readelf, for pc attribution")
    p.add_argument("--case", action="append", default=[],
                   help="restrict to these case directories (a prefix is enough); for debugging "
                        "the harness, never for a measurement -- a bundle of a subset says so in "
                        "its population line")
    p.add_argument("--scratch", type=Path, help="where triggers are staged (default: --out/stage)")
    p.add_argument("--timeout", type=int, default=4200)
    p.add_argument("--run-timeout", type=int, default=40)
    p.add_argument("--cheri-img", type=Path,
                   help="the CheriBSD image for the cheribsd-revocation arm (default: the SDK's)")
    p.add_argument("--php-fullext", action="store_true",
                   help="a php-fullext binary is staged beside php; the cases whose fidelity says "
                        "they need bundled extensions run on it")
    p.add_argument("--eager-timeout", type=int, default=0,
                   help="guest seconds per run in the eager configuration (default: 5x "
                        "--run-timeout). Every free sweeps there, so PHP's own shutdown is orders "
                        "of magnitude slower and a default-sized budget reads as a hang")
    p.add_argument("--eager", action="store_true",
                   help="a third CheriBSD boot with every-free revocation, to tell a quarantined "
                        "chunk from a miss. Context only: it never decides a verdict")
    args = p.parse_args()
    if args.self_test:
        return self_test()
    virtual = [a for a in args.arm if a != CHERI_ARM]
    needs = ("corpus", "out") + (("qemu", "images", "adapter", "image") if virtual else ())
    for need in needs:
        if getattr(args, need) is None:
            p.error(f"--{need.replace('_', '-')} is required for arm(s) {virtual or args.arm}")
    if not args.arm:
        p.error("--arm is required")

    args.eager_timeout = args.eager_timeout or args.run_timeout * 5
    decl, cases = load_corpus(args.corpus)
    if args.case:
        cases = [(n, c) for n, c in cases if any(n.startswith(w) for w in args.case)]
        if not cases:
            sys.exit(f"--case matched nothing in {args.corpus}")
    missing = [a for a in args.arm if a not in decl.get("arm_configurations", {})]
    if missing:
        sys.exit(f"{args.corpus}: corpus.json maps no configuration for {missing}; the judge "
                 f"measures a configuration, never an arm name")
    bundle, rows = host_record(args.corpus)
    args.scratch = args.scratch or (args.out / "stage")
    stage(args.scratch, args.corpus, cases, port_module("run-realphp").SANITY_TEXT)
    if CHERI_ARM in args.arm:
        for f in (("php", "bounds-malloc") + (("php-fullext",) if args.php_fullext else ())):
            if not (args.scratch / f).is_file():
                sys.exit(f"the {CHERI_ARM} arm needs {args.scratch / f}: copy the purecap binary "
                         f"built by ports/php/cheribsd/build.sh and its control there first")
    print(f"{args.corpus.name}: {len(cases)} cases, host record from {bundle.name}")

    baseline, tally = {}, {}
    for arm in args.arm:
        if arm == CHERI_ARM:
            completed, judged = run_cheribsd_arm(args, decl, cases, arm, rows, Path(args.out))
        else:
            completed, judged = run_arm(args, decl, cases, arm, rows, baseline)
        if arm == BASELINE_ARM:
            baseline = completed
        counts = {}
        for _, (verdict, reason, _) in judged:
            key = verdict if verdict != verdicts.NO_READING else f"{verdict}:{reason}"
            counts[key] = counts.get(key, 0) + 1
        tally[arm] = counts
        print(f"  {arm}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(json.dumps(tally, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
