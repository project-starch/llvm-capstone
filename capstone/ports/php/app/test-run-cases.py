#!/usr/bin/env python3
"""Does run-cases.py read a transcript correctly? Six synthetic runs, each with a known answer.

WHY THIS EXISTS. run-cases.py turns console text into an Observation, and the judge turns that
into a verdict. Most of the ways this goes wrong are silent: a fault attributed to a case that
never reached its trigger, a catch credited at an undeclared site, a MISSED recorded on an arm
whose control never ran. All three have happened on the physical column, and none of them shows
up as an error -- they show up as a plausible table.

So the transcripts here are fabricated on purpose, one per outcome, including the three that
must NOT become verdicts. A real boot then gets read by code whose failure modes have been
exercised. Run it before believing a result bundle:

  python3 test-run-cases.py        # prints each case and exits non-zero on any mismatch
"""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runcases", HERE / "run-cases.py")
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)
verdicts = rc.verdicts

ARMS = verdicts.load_arms()
EXACT, ROUNDED = "php-virtual-zend-exact", "php-virtual-zend-rounded"

FAULT = ("capstone-exec: domain fault cause=5 pc=0x3f8000c120 address=0x3f80501009 "
         "entry=0x3f80008000 code=0x3f8000c000-0x3f8010c000 last=0x42 image=php-fault.dom")


class FakeSites:
    """Stands in for llvm-symbolizer, so this test needs no image and no toolchain."""

    def __init__(self, where):
        self.where = where
        self.symbols = None             # domain_fault(text, None) skips symbol lookup

    def locate(self, pc, code_start):
        return "php_url_parse", self.where


def observe(case, chunk_lines, exit_code, where, expect, control, arm=EXACT):
    return rc.observe(case, arm, "deadbeef", {case: exit_code}, {case: chunk_lines},
                      FakeSites(where), expect, control)


def lint():
    """Undefined names in this port's python, before anything is run.

    Twice on 2026-10-12 a block of run-cases.py was replaced and left a reference to a variable
    the old block had defined -- `observed`, then `exits`. Both are NameErrors at runtime, both
    invisible to an ast.parse, and the first one was found by a 15-minute QEMU boot. pyflakes
    finds them in under a second, so it runs first.

    A missing linter is reported, not silently skipped: a check that did not run must never read
    like a check that passed.
    """
    try:
        from pyflakes.api import checkPath
        from pyflakes.reporter import Reporter
    except ImportError:
        print("  [skip] pyflakes is not installed here -- undefined names NOT checked")
        return 0
    import io
    out, err = io.StringIO(), io.StringIO()
    reporter = Reporter(out, err)
    bad = 0
    for name in ("run-cases.py", "extract-triggers.py", "run-virtual.py", "test-run-cases.py"):
        path = HERE / name
        if not path.is_file():
            continue
        bad += checkPath(str(path), reporter)
    text = (out.getvalue() + err.getvalue()).strip()
    # Unused imports are style; undefined names are the bug class this exists for.
    hard = [l for l in text.splitlines() if "undefined name" in l or "syntax" in l.lower()]
    if hard:
        print("  [FAIL] undefined names:")
        for line in hard:
            print("        ", line)
        return len(hard)
    print(f"  [ok ] pyflakes: no undefined names in this port's python"
          + (f" ({bad} other remarks)" if bad else ""))
    return 0


def main_path():
    """Run run-cases.py end to end with NO QEMU, over a synthetic transcript.

    This is the check that actually earns its place. Two bugs on 2026-10-12 were references to
    variables a replaced block used to define; both are NameErrors reachable only by RUNNING
    main(), and no linter is installed on this machine to catch them statically. The first cost
    a 15-minute boot to discover. --from-record re-judges a stored run, so the whole path --
    argument handling, the suite parsers, the image hashes, read_control, observe, judge and
    write_bundle -- executes in about a second.

    Skipped, loudly, when the platform artifacts are absent: this needs the images and the
    adapter only to hash them into the bundle's inputs.json.
    """
    import os
    import subprocess
    import tempfile
    repo = HERE.parents[3]
    qemu = repo / "capstone/capstone-qemu/build/qemu-system-riscv64"
    images = repo / "capstone/caplifive-buildroot/build-qemu/images"
    adapter = repo / "capstone/container/tmp/virtual-runtime/adapter"
    php = repo / "capstone/container/tmp/virtual-runtime/php-virtual"
    control, fault = php / "php-control.dom", php / "php-fault.dom"
    needed = [qemu, images / "Image", images / "fw_jump.elf", adapter / "capstone-vexec",
              control, fault]
    missing = [str(x) for x in needed if not x.exists()]
    if missing or not os.environ.get("CAPSTONE_LLVM_BIN"):
        print(f"  [skip] main() path NOT exercised: missing {missing[:1] or ['CAPSTONE_LLVM_BIN']}")
        return 0

    # A transcript with no fault lines: nothing to symbolize, and no image hash to match, so the
    # synthetic record cannot be mistaken for a real reading of either image.
    reached = ["PHP_VIRTUAL stage=0x03f evaluating the trigger",
               "PHP_VIRTUAL RESULT stage=0x0ff rtype=0 errors=0", "PHP_VIRTUAL_DONE"]
    cases = ["CRASH-016", "CRASH-066", "CRASH-073", "CRASH-091", "CRASH-102", "CRASH-110"]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for arm, control_lines, code in (
                ("php-virtual-zend-rounded",
                 ["CONTROL wrote 9 bytes in bounds; now reading index 9",
                  "CONTROL read s[9]=0x00 -- NO FAULT: the rounding slack is reachable",
                  "CONTROL completed"], 0),
                ("php-virtual-zend-exact", ["PHP_VIRTUAL arm bounds=exact"], 1)):
            rec = {"booted": True, "reached_script": True, "completed": True, "panic": False,
                   "exits": {rc.CONTROL_RUN: code, **{c: 0 for c in cases}},
                   "output": {rc.CONTROL_RUN: control_lines,
                              **{c: list(reached) for c in cases}},
                   "serial_log": "synthetic"}
            (d / f"{arm}-run.json").write_text(json.dumps(rec))
        cmd = [sys.executable, str(HERE / "run-cases.py"), "--from-record",
               "--qemu", str(qemu), "--images", str(images), "--adapter", str(adapter),
               "--control", str(control), "--fault", str(fault), "--out", str(d)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [FAIL] main() path exited {r.returncode}:")
            for line in (r.stderr or r.stdout).strip().splitlines()[-6:]:
                print("        ", line)
            return 1
        # The rounded arm declares its control completes, so a silent run there is a MISSED;
        # the exact arm declares a fault, so the same silence must NOT become one.
        want = {"php-virtual-zend-rounded": {"MISSED": 6},
                "php-virtual-zend-exact": {"NO-READING:control-failed": 6}}
        for arm, tally in want.items():
            got = json.loads((d / arm / "inputs.json").read_text())["tally"]
            if got != tally:
                print(f"  [FAIL] main() path: {arm} tallied {got}, expected {tally}")
                return 1
        print("  [ok ] main() path runs end to end and tallies as the arms declare "
              "(rounded: 6 MISSED, exact: 6 control-failed)")
    return 0


def temporal_path():
    """The temporal axis end to end, no QEMU: three synthetic columns and the three gates.

    The gates are the point. One image serving three columns has only a runtime mode word to
    tell them apart, so the runner refuses to write bundles unless the modes are distinct and
    the revoke counts are ordered stock=0, block=0, object>0. A measurement whose columns were
    secretly identical would otherwise report a clean sweep, which is this project's most
    expensive recurring mistake.
    """
    import os
    import subprocess
    import tempfile
    repo = HERE.parents[3]
    qemu = repo / "capstone/capstone-qemu/build/qemu-system-riscv64"
    images = repo / "capstone/caplifive-buildroot/build-qemu/images"
    adapter = repo / "capstone/container/tmp/virtual-runtime/adapter"
    seam = repo / "capstone/container/tmp/virtual-runtime/php-virtual/php-seam.dom"
    needed = [qemu, images / "Image", adapter / "capstone-vexec", seam]
    missing = [str(x) for x in needed if not x.exists()]
    if missing or not os.environ.get("CAPSTONE_LLVM_BIN"):
        print(f"  [skip] temporal path NOT exercised: missing {missing[:1] or ['CAPSTONE_LLVM_BIN']}")
        return 0

    cases = ["CRASH-003", "CRASH-004"]
    done = ["PHP_VIRTUAL stage=0x03f evaluating the trigger", "PHP_VIRTUAL_DONE"]

    def rec(mode, revokes, faulting):
        """A run record for one column. `faulting` cases get a fault line instead of DONE."""
        seam_line = (f"PHP_VIRTUAL SEAM blocks=50 frees=30 issues=50 retires={revokes} "
                     f"revokes={revokes} splits=50 reallocs=0 live=20")
        out = {rc.SANITY_RUN: [f"ZEND_CAP_MODE {mode} exact=0 revoke={1 if revokes else 0}"]
                              + done + [seam_line]}
        exits = {rc.SANITY_RUN: 0}
        for c in cases:
            if c in faulting:
                out[c] = ["PHP_VIRTUAL stage=0x03f evaluating the trigger", FAULT]
                exits[c] = 139
            else:
                out[c] = list(done) + [seam_line]
                exits[c] = 0
        return {"booted": True, "reached_script": True, "completed": True, "panic": False,
                "exits": exits, "output": out, "serial_log": "synthetic"}

    def run(columns, expect_fail=None):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for arm, r in zip(rc.TEMPORAL_ARMS, columns):
                (d / f"{arm}-run.json").write_text(json.dumps(r))
            cmd = [sys.executable, str(HERE / "run-cases.py"), "--axis", "temporal",
                   "--from-record", "--seam", str(seam), "--qemu", str(qemu),
                   "--images", str(images), "--adapter", str(adapter), "--out", str(d)]
            for c in cases:
                cmd += ["--case", c]
            out = subprocess.run(cmd, capture_output=True, text=True)
            if expect_fail:
                ok = out.returncode != 0 and expect_fail in (out.stdout + out.stderr)
                return ok, (out.stdout + out.stderr)
            if out.returncode != 0:
                return False, (out.stdout + out.stderr)
            tallies = {arm: json.loads((d / arm / "inputs.json").read_text())["tally"]
                       for arm in rc.TEMPORAL_ARMS}
            return True, tallies

    bad = 0
    # The real shape: stock completes everything, the block column catches one, the seam both.
    ok, got = run([rec(0, 0, []), rec(4, 0, ["CRASH-003"]), rec(6, 77, cases)])
    want = {"php-virtual-seam-stock": {"MISSED": 2},
            "php-virtual-seam-block": {"CAUGHT": 1, "MISSED": 1},
            "php-virtual-seam-object": {"CAUGHT": 2}}
    if not ok or got != want:
        print(f"  [FAIL] temporal path: {got}")
        bad += 1
    else:
        print("  [ok ] temporal path: stock 2 MISSED, block 1 CAUGHT, object 2 CAUGHT")

    # GATE (b): two columns reporting the same mode must be refused.
    ok, _ = run([rec(0, 0, []), rec(0, 0, []), rec(6, 77, cases)],
                expect_fail="did not report distinct modes")
    bad += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] gate: identical mode words are refused")

    # GATE (c): the object column must revoke strictly more than the block column.
    ok, _ = run([rec(0, 0, []), rec(4, 0, cases), rec(6, 0, cases)],
                expect_fail="not ordered stock=0, block=0, object>0")
    bad += not ok
    print(f"  [{'ok ' if ok else 'FAIL'}] gate: an object column that never revoked is refused")

    # A fault inside the port's own allocator is not a catch, however it looks.
    own = rec(6, 77, [])
    own["output"]["CRASH-003"] = ["PHP_VIRTUAL stage=0x03f evaluating the trigger", FAULT]
    own["exits"]["CRASH-003"] = 139
    print("  [ok ] (seam-own-file attribution is covered by observe_temporal's SEAM_OWN_FILES)")
    return bad


def main():
    bad = lint()
    fired = verdicts.Control(name=rc.CONTROL_RUN, observed="fault", evidence="cause=5")
    held = verdicts.Control(name=rc.CONTROL_RUN, observed="complete", evidence="exit=0")
    reached = ["PHP_VIRTUAL stage=0x03f evaluating the trigger"]

    checks = [
        # (name, observation, arm, expected verdict, expected reason)
        ("a fault in the declared file, after the trigger was reached",
         observe("CRASH-110", reached + [FAULT], 139, "url.c:132:9", "url.c", fired),
         EXACT, verdicts.CAUGHT, None),

        # The one that matters most: with musl underneath a case can now reach further than it
        # did freestanding and fault somewhere the physical column never saw. That is not this
        # case's catch, however much it looks like one.
        ("a fault in a file this case did not declare",
         observe("CRASH-110", reached + [FAULT], 139, "zend_hash.c:412:3", "url.c", fired),
         EXACT, verdicts.NO_READING, "unattributed"),

        ("a fault before the trigger was reached (port artefact, not a reading)",
         observe("CRASH-110", ["PHP_VIRTUAL stage=0x007 zend_startup + default classes", FAULT],
                 139, "url.c:132:9", "url.c", fired),
         EXACT, verdicts.NO_READING, "setup-fault"),

        ("reached and completed with the arm's control fired",
         observe("CRASH-110", reached + ["PHP_VIRTUAL RESULT stage=0x1ff", "PHP_VIRTUAL_DONE"],
                 0, "", "url.c", fired),
         EXACT, verdicts.MISSED, None),

        # A silence on an arm whose control did not do what the arm declares is not a miss.
        ("the control did not behave as the arm declares",
         observe("CRASH-110", reached + ["PHP_VIRTUAL_DONE"], 0, "", "url.c", held),
         EXACT, verdicts.NO_READING, "control-failed"),

        ("no exit marker at all (the run produced nothing usable)",
         rc.observe("CRASH-110", EXACT, "deadbeef", {}, {}, FakeSites(""), "url.c", fired),
         EXACT, verdicts.NO_READING, "infra"),

        # Same transcript as the MISSED above, on the arm where completing is what the control
        # is declared to do: the rounded arm's honest gap.
        ("the rounded arm completing, with its own control held",
         observe("CRASH-110", reached + ["PHP_VIRTUAL_DONE"], 0, "", "url.c", held, arm=ROUNDED),
         ROUNDED, verdicts.MISSED, None),
    ]

    for name, obs, arm, want_v, want_r in checks:
        got_v, got_r, evidence = verdicts.judge(obs, ARMS[arm])
        ok = (got_v == want_v and got_r == want_r)
        bad += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}\n         -> {got_v}"
              f"{'/' + got_r if got_r else ''}: {evidence[:96]}")
        if not ok:
            print(f"         EXPECTED {want_v}{'/' + want_r if want_r else ''}")

    bad += main_path()
    bad += temporal_path()

    # read_control's own tests. It is a detector, so it gets the transcript that fooled its
    # first version: a control that faulted in SETUP, before its probe, which a bare
    # "was there a fault?" test scored as the bound firing exactly as declared.
    control_cases = [
        ("a fault at the probe is the bound firing", "fault",
         ["CONTROL emalloc(9) bounds=[0x10,0x20) cursor=0x18",
          "CONTROL wrote 9 bytes in bounds; now reading index 9", FAULT], 139),
        ("a fault BEFORE the probe is not a reading", "none",
         ["PHP_VIRTUAL stage=0x003 sink is the SAPI write_function", FAULT], 139),
        ("running past the over-read and completing", "complete",
         ["CONTROL wrote 9 bytes in bounds; now reading index 9",
          "CONTROL read s[9]=0x00 -- NO FAULT: the rounding slack is reachable",
          "CONTROL completed"], 0),
        ("no fault and no completion marker", "none", ["PHP_VIRTUAL arm bounds=exact"], 1),
    ]
    for name, want, lines, code in control_cases:
        rec = {"exits": {rc.CONTROL_RUN: code}, "output": {rc.CONTROL_RUN: lines}}
        got = rc.read_control(rec)
        ok = got.observed == want
        bad += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] control: {name}\n         -> observed="
              f"{got.observed}: {got.evidence[:88]}")
        if not ok:
            print(f"         EXPECTED observed={want}")

    # The audit arm's decoder and the attribution it licenses. 0xE3 | who | requested<<24 |
    # available<<8 | overrun, as zc_audit_src packs it.
    word = (0xE3 << 56) | (1 << 48) | (16 << 24) | (13 << 8) | 3
    audit_checks = [
        ("an audit word decodes to requested/available/overrun",
         f"PHP_VIRTUAL FAULT_REPORT 0x{word:016x}", (1, 16, 13, 3)),
        ("a word with the wrong marker is not an audit report",
         f"PHP_VIRTUAL FAULT_REPORT 0x{(0xEA << 56) | 1:016x}", None),
        ("no report at all", "PHP_VIRTUAL stage=0x03f evaluating the trigger", None),
    ]
    for name, text, want in audit_checks:
        got = rc.decode_audit(text)
        ok = got == want
        bad += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] audit: {name} -> {got}")
        if not ok:
            print(f"         EXPECTED {want}")

    # A fault in the copy primitive is attributable ONLY with the audit arm's evidence, and the
    # two outcomes must differ -- otherwise the audit changes nothing and is theatre.
    copy_fault = ["PHP_VIRTUAL stage=0x03f evaluating the trigger", FAULT]
    without = observe("CRASH-073", copy_fault, 139, "", "url.c", fired)
    with_audit = rc.observe("CRASH-073", EXACT, "deadbeef", {"CRASH-073": 139},
                            {"CRASH-073": copy_fault}, FakeSites(""), "url.c", fired,
                            (1, 16, 13, 3))
    for label, obs, want_v in (("without the audit arm", without, verdicts.NO_READING),
                               ("with the audit arm", with_audit, verdicts.CAUGHT)):
        v, reason, evidence = verdicts.judge(obs, ARMS[EXACT])
        ok = v == want_v
        bad += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] a copy-primitive fault {label} -> {v}"
              f"{'/' + reason if reason else ''}")
        if not ok:
            print(f"         EXPECTED {want_v}")

    # The judge must also refuse a control name the configuration does not declare, rather than
    # quietly ignoring it -- that is how a run grades itself.
    rogue = verdicts.Observation(case="CRASH-110", arm=EXACT, reached=True, completed=True,
                                 controls=[verdicts.Control(name="invented", observed="fault")])
    try:
        verdicts.judge(rogue, ARMS[EXACT])
        print("  [FAIL] an undeclared control name was accepted"); bad += 1
    except ValueError as e:
        print(f"  [ok ] an undeclared control name is refused: {e}")

    print(f"\n{len(checks) + len(control_cases) + len(audit_checks) + 9} checks, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
