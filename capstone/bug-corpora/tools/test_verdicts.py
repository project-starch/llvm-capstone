"""Tests for tools/verdicts.py: every rule of the judge, the registry, the parser and the bundle.

    python3 -m unittest discover -s capstone/bug-corpora/tools -p 'test_*.py'

Each rule is tested from the side where it must FIRE, because a judge whose NO-READING branches
never trigger would still pass a suite of happy-path rows.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import verdicts as v  # noqa: E402

ARMS = v.load_arms()
SPEC = ARMS["app-level0"]          # bounds-malloc must fault, uaf-malloc must complete
OK_CONTROLS = [v.Control("bounds-malloc", "fault"), v.Control("uaf-malloc", "complete")]
FAULT = v.Fault(cause=7, pc=0xc02020f0, symbol="pqGetnchar", offset=0x10)


def obs(**kw):
    base = dict(case="00_x", arm="spatial", controls=list(OK_CONTROLS))
    base.update(kw)
    return v.Observation(**base)


class Judge(unittest.TestCase):
    def test_infra_is_never_a_verdict(self):
        verdict, reason, _ = v.judge(obs(infra="infra", fault=FAULT, reached=True,
                                         attribution="probe"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "infra"))

    def test_unknown_reason_is_an_error(self):
        with self.assertRaises(ValueError):
            v.judge(obs(infra="flaky"), SPEC)

    def test_failed_control_voids_even_an_attributed_fault(self):
        bad = [v.Control("bounds-malloc", "complete"), v.Control("uaf-malloc", "complete")]
        verdict, reason, evidence = v.judge(obs(controls=bad, fault=FAULT, reached=True,
                                                attribution="probe"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "control-failed"))
        self.assertIn("bounds-malloc: expected fault, observed complete", evidence)

    def test_control_not_declared_for_the_configuration_is_an_error(self):
        with self.assertRaises(ValueError):
            v.judge(obs(controls=[v.Control("uaf-chunk", "complete")]), SPEC)

    def test_fault_before_the_marker_is_setup(self):
        verdict, reason, _ = v.judge(obs(fault=FAULT, reached=False, attribution="function"), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "setup-fault"))

    def test_fault_without_attribution_is_not_a_catch(self):
        verdict, reason, _ = v.judge(obs(fault=FAULT, reached=True), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "unattributed"))

    def test_attributed_fault_is_caught(self):
        for how in v.ATTRIBUTIONS:
            verdict, reason, evidence = v.judge(obs(fault=FAULT, reached=True, attribution=how,
                                                    attribution_evidence="x"), SPEC)
            self.assertEqual((verdict, reason), (v.CAUGHT, None), how)
            self.assertIn(f"by {how}", evidence)

    def test_silence_without_the_marker_is_not_missed(self):
        verdict, reason, _ = v.judge(obs(reached=False, completed=True), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "not-reached"))

    def test_reached_but_never_ended(self):
        verdict, reason, _ = v.judge(obs(reached=True, completed=False), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "inconclusive"))

    def test_silence_needs_the_arms_controls(self):
        verdict, reason, evidence = v.judge(obs(reached=True, completed=True,
                                                controls=[OK_CONTROLS[0]]), SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "control-failed"))
        self.assertIn("uaf-malloc", evidence)

    def test_missed(self):
        verdict, reason, _ = v.judge(obs(reached=True, completed=True), SPEC)
        self.assertEqual((verdict, reason), (v.MISSED, None))


class CatchCauses(unittest.TestCase):
    """`catch_causes` is optional per configuration: which fault causes that arm's MECHANISM can
    report. It exists because a fault anywhere in a declared function cannot tell a rejected
    access from a crash two frames later on a value the program was allowed to read."""

    SPEC = {**SPEC, "catch_causes": [25, 28]}

    def fault(self, cause):
        return v.Fault(cause=cause, pc=0xc02020f0, symbol="pqGetnchar", offset=0x10)

    def test_a_mechanism_cause_is_still_a_catch(self):
        for cause in (25, 28):
            verdict, reason, evidence = v.judge(
                obs(fault=self.fault(cause), reached=True, attribution="probe"), self.SPEC)
            self.assertEqual((verdict, reason), (v.CAUGHT, None), cause)
            self.assertIn(f"cause={cause}", evidence)

    def test_a_consequence_cause_is_a_miss_not_a_catch(self):
        verdict, reason, evidence = v.judge(
            obs(fault=self.fault(24), reached=True, attribution="probe"), self.SPEC)
        self.assertEqual((verdict, reason), (v.MISSED, None))
        self.assertIn("cause=24", evidence)
        self.assertIn("[25, 28]", evidence)
        self.assertIn("consequence", evidence)

    def test_a_consequence_needs_the_controls_to_score_missed(self):
        # A MISSED needs the configuration's controls_for_missed behaving in the same
        # invocation, exactly as an ordinary MISSED does.
        verdict, reason, evidence = v.judge(
            obs(fault=self.fault(24), reached=True, attribution="probe",
                controls=[v.Control("bounds-malloc", "fault")]), self.SPEC)
        self.assertEqual((verdict, reason), (v.NO_READING, "control-failed"))
        self.assertIn("uaf-malloc", evidence)

    def test_an_empty_set_means_no_fault_on_this_arm_can_be_a_catch(self):
        # cheribsd-sqlite-memsys5 declares []: revocation acts on a libc free and under
        # SQLITE_CONFIG_HEAP there is none, so every cause it can report is a consequence.
        spec = {**SPEC, "catch_causes": []}
        for cause in (1, 2, 25, 28):
            verdict, reason, evidence = v.judge(
                obs(fault=self.fault(cause), reached=True, attribution="probe"), spec)
            self.assertEqual((verdict, reason), (v.MISSED, None), cause)
            self.assertIn("no cause at all", evidence)
            self.assertNotIn("causes []", evidence)

    def test_a_configuration_without_the_key_behaves_as_before(self):
        verdict, reason, _ = v.judge(
            obs(fault=self.fault(24), reached=True, attribution="probe"), SPEC)
        self.assertEqual((verdict, reason), (v.CAUGHT, None))


class Registry(unittest.TestCase):
    def test_every_configuration_is_well_formed(self):
        self.assertTrue(ARMS)
        for name, spec in ARMS.items():
            self.assertTrue(set(spec["controls_for_missed"]) <= set(spec["controls"]), name)
            self.assertTrue(set(spec["controls"].values()) <= {"fault", "complete"}, name)
            self.assertTrue(spec["controls_for_missed"], f"{name}: a silence would need no proof")

    def test_corpus_arm_configurations_exist(self):
        for path in sorted(HERE.parent.glob("*/*/corpus.json")):
            mapping = json.loads(path.read_text()).get("arm_configurations", {})
            for arm, config in mapping.items():
                self.assertIn(config, ARMS, f"{path}: {arm}")


class Parser(unittest.TestCase):
    # A committed raw result from the c-repros sublet run of 2026-10-06.
    C00 = (HERE.parent / "postgres/c-repros/results/sublet-20261006-161014"
           / "00_CVE-2026-6477_pqfn_unbounded_result_copy.json")

    def test_domain_fault_line(self):
        fault = v.domain_fault(json.loads(self.C00.read_text())["fault"])
        self.assertEqual((fault.cause, fault.pc), (7, 0xc02020f0))

    def test_no_fault_line(self):
        self.assertIsNone(v.domain_fault("capstone-vm: application exited 0"))

    # The launcher's line in its three shapes: before 2026-10-27 (no register words), with the
    # words, and with the words at zero (a platform that wrote no register slot).
    BASE = "capstone-exec: domain fault cause=25 pc=0x3f90411888 address=0x3f90a0ba20"
    TAIL = " entry=0x3f904009a0 code=0x3f90400000-0x3f90800000 last=0x39"
    LINES = {"old": BASE + TAIL,
             "regs": BASE + " ra=0x3f90508d7c fp=0x3f80a01f00" + TAIL,
             "zero": BASE + " ra=0x0 fp=0x0" + TAIL,
             "chain": BASE + " ra=0x3f90508d7c fp=0x3f80a01f00" + TAIL
                      + " sha256=ab image=x.dom chain=0x3f90508d7c,0x3f90412000,0x3f90400aaa"}

    def test_frame_chain(self):
        # Printed last on the line since 2026-10-28; read as a tuple of addresses, empty when
        # absent, and never from the next line.
        with_chain = v.domain_fault(self.LINES["chain"] + "\ncapstone-vm: application exited 139")
        self.assertEqual(with_chain.chain, (0x3f90508d7c, 0x3f90412000, 0x3f90400aaa))
        self.assertEqual((with_chain.ra, with_chain.fp), (0x3f90508d7c, 0x3f80a01f00))
        self.assertEqual(v.domain_fault(self.LINES["regs"]).chain, ())

    def test_register_words(self):
        # Read when present and non-zero, None otherwise, so nothing downstream can mistake a
        # missing word for an address.
        with_ra = v.domain_fault(self.LINES["regs"])
        self.assertEqual((with_ra.cause, with_ra.pc, with_ra.ra, with_ra.fp),
                         (25, 0x3f90411888, 0x3f90508d7c, 0x3f80a01f00))
        old = v.domain_fault(self.LINES["old"])
        self.assertEqual((old.cause, old.ra, old.fp), (25, None, None))
        zero = v.domain_fault(self.LINES["zero"])
        self.assertEqual((zero.pc, zero.ra, zero.fp), (0x3f90411888, None, None))

    def test_fault_line_groups_are_what_the_runners_read(self):
        # run-capstone-domain.py and run-virtual-cases.py take FAULT_LINE's group(4) as the
        # code base (`int(m.group(4), 16)`); the register words must not shift it. Every shape
        # of the line, and the committed pre-2026-10-27 record, give the same four groups.
        for shape, line in self.LINES.items():
            m = v.FAULT_LINE.search(line)
            self.assertEqual((m.group(1), int(m.group(2), 16), m.group(3), int(m.group(4), 16)),
                             ("25", 0x3f90411888, "0x3f90a0ba20", 0x3f90400000), shape)
            self.assertEqual(m.lastindex, 4, shape)
        m = v.FAULT_LINE.search(json.loads(self.C00.read_text())["fault"])
        self.assertEqual((m.group(1), int(m.group(4), 16) < int(m.group(2), 16)), ("7", True))


class Bundle(unittest.TestCase):
    PLATFORM = {k: "unrecorded" for k in v.PLATFORM_KEYS}

    def test_round_trip_and_tally(self):
        rows = [(obs(fault=FAULT, reached=True, attribution="probe"), None),
                (obs(reached=False), None)]
        rows = [(o, v.judge(o, SPEC)) for o, _ in rows]
        with tempfile.TemporaryDirectory() as d:
            record = v.write_bundle(d, "postgres/c-repros", "spatial", rows,
                                    {"platform": self.PLATFORM, "configuration": "app-level0"})
            self.assertEqual(record["tally"], {"CAUGHT": 1, "NO-READING:not-reached": 1})
            back = v.read_bundle(d)
        self.assertEqual(back[0][0].fault.pc, FAULT.pc)
        self.assertEqual([r[1] for r in back], [v.CAUGHT, v.NO_READING])

    def test_platform_must_be_said(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            v.write_bundle(d, "c", "spatial", [], {"platform": {"compiler": "x"}})


class LlvmBin(unittest.TestCase):
    """--llvm-bin is refused at argument parsing unless the judge's two tools are there and executable."""

    def dir_with(self, d, tools, mode=0o755):
        for t in tools:
            (Path(d) / t).write_text("#!/bin/sh\n")
            (Path(d) / t).chmod(mode)
        return d

    def test_both_present(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(v.llvm_bin_dir(self.dir_with(d, ("llvm-nm", "llvm-readelf"))), Path(d))

    def test_readelf_missing(self):   # the copied bin+lib set of 2026-10-17: llvm-nm and llvm-readobj, no llvm-readelf
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(Exception, "llvm-readelf"):
                v.llvm_bin_dir(self.dir_with(d, ("llvm-nm", "llvm-readobj")))

    def test_not_executable(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(Exception, "llvm-nm or llvm-readelf"):
                v.llvm_bin_dir(self.dir_with(d, ("llvm-nm", "llvm-readelf"), mode=0o644))

    def test_dangling_link(self):
        with tempfile.TemporaryDirectory() as d:
            self.dir_with(d, ("llvm-readelf",))
            (Path(d) / "llvm-nm").symlink_to(Path(d) / "nowhere")
            with self.assertRaisesRegex(Exception, "llvm-nm"):
                v.llvm_bin_dir(d)


if __name__ == "__main__":
    unittest.main()
