#!/usr/bin/env python3
"""Show that the oracle can say FAIL.

The program under test no longer reports anything about itself: a supervisor
observes it from outside and prints what the kernel said. That removes the
program's ability to mislead us. It does not remove OURS to misread the answer,
and every check below is a mistake that was actually made while building this:
an unresolved address taken as a match, a plausible but wrong expectation, a
value read from the wrong line.
"""

import importlib.util
from pathlib import Path
import struct
import unittest

path = Path(__file__).resolve().parents[2] / "runners/cheribsd/run-defects.py"
spec = importlib.util.spec_from_file_location("run_defects", path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

EXPECT = "SUPERVISE expect pyc_defect_read 0x1084aa"
FAULT = "SUPERVISE fault signal=34 code=2 addr=0x1084aa pc=0x1084aa"
ROW = dict(exit=162, passed=True, stdout_sha256="0" * 64)


def report(mode=0, completed=1, count=1, status=0, magic=oracle.MAGIC):
    return struct.pack("<12Q", magic, count, mode, status, completed, *([0] * 7))


class ProtectedArm(unittest.TestCase):
    def verdict(self, *lines, row=None):
        return oracle.evaluate(5, 1, row or ROW, "\n".join(lines) + "\n", None)

    def test_a_tag_fault_at_the_resolved_address_passes(self):
        self.assertTrue(self.verdict(EXPECT, FAULT)["passed"])

    def test_an_unresolved_expectation_is_not_a_match(self):
        # The supervisor says so when it cannot resolve the address. Treating
        # that as agreement would compare nothing; it happened twice.
        self.assertFalse(
            self.verdict("SUPERVISE expect pyc_defect_read unavailable", FAULT)[
                "passed"
            ]
        )

    def test_a_missing_expectation_is_not_a_match(self):
        self.assertFalse(self.verdict(FAULT)["passed"])

    def test_a_plausible_but_wrong_expectation_fails(self):
        # Resolving the base from the executable mapping instead of the lowest
        # one overshoots by the text segment's offset and yields exactly this.
        wrong = "SUPERVISE expect pyc_defect_read 0x10b4aa"
        self.assertFalse(self.verdict(wrong, FAULT)["passed"])

    def test_a_bounds_fault_is_not_temporal_protection(self):
        self.assertFalse(
            self.verdict(EXPECT, FAULT.replace("code=2", "code=1"))["passed"]
        )

    def test_a_fault_elsewhere_fails(self):
        self.assertFalse(
            self.verdict(
                EXPECT, "SUPERVISE fault signal=34 code=2 addr=0x40001 pc=0x40001"
            )["passed"]
        )

    def test_addr_and_pc_must_agree_with_each_other(self):
        self.assertFalse(
            self.verdict(
                EXPECT, "SUPERVISE fault signal=34 code=2 addr=0x1084aa pc=0x40001"
            )["passed"]
        )

    def test_no_fault_at_all_fails(self):
        self.assertFalse(self.verdict(EXPECT)["passed"])

    def test_the_first_fault_decides_not_a_later_one(self):
        # A process that dies can produce more than one stop. A later, matching
        # line must not rescue an arm whose own fault was wrong.
        self.assertFalse(
            self.verdict(
                EXPECT,
                "SUPERVISE fault signal=34 code=1 addr=0x40001 pc=0x40001",
                FAULT,
            )["passed"]
        )

    def test_a_line_must_be_whole_not_a_substring(self):
        self.assertFalse(self.verdict(EXPECT, "echo " + FAULT)["passed"])

    def test_the_platform_outcome_still_has_to_hold(self):
        self.assertFalse(
            self.verdict(EXPECT, FAULT, row=dict(ROW, exit=0, passed=False))["passed"]
        )


class SpatialArm(unittest.TestCase):
    row = dict(exit=0, passed=True, stdout_sha256="0" * 64)

    def verdict(self, blob=None, row=None):
        return oracle.evaluate(
            5,
            0,
            row or self.row,
            "SUPERVISE exit status=0\n",
            report() if blob is None else blob,
        )

    def test_a_completed_case_with_a_matching_report_passes(self):
        self.assertTrue(self.verdict()["passed"])

    def test_a_missing_or_truncated_report_fails(self):
        self.assertFalse(self.verdict(blob=b"")["passed"])
        self.assertFalse(self.verdict(blob=report()[:88])["passed"])

    def test_a_report_that_did_not_complete_or_names_another_mode_fails(self):
        self.assertFalse(self.verdict(blob=report(completed=0))["passed"])
        self.assertFalse(self.verdict(blob=report(mode=1))["passed"])

    def test_a_foreign_or_multi_case_report_fails(self):
        self.assertFalse(self.verdict(blob=report(magic=1))["passed"])
        self.assertFalse(self.verdict(blob=report(count=2))["passed"])
        self.assertFalse(self.verdict(blob=report(status=9))["passed"])


class NegativeControl(unittest.TestCase):
    """The control's own verdict. It decides whether a whole control run means
    anything, and two refactors broke it where only a guest run could notice."""

    def test_an_arm_that_never_ran_its_case_has_fired(self):
        refused = oracle.evaluate(
            5, 1, dict(exit=3, passed=False, stdout_sha256="0" * 64), "", None
        )
        self.assertTrue(oracle.control_fired(refused))

    def test_an_arm_that_faulted_at_the_probe_is_vacuous_not_fired(self):
        # If the oracle still reports a fault under the control, the fixture
        # did not stop the case and the control proves nothing.
        ran = oracle.evaluate(5, 1, ROW, EXPECT + "\n" + FAULT + "\n", None)
        self.assertFalse(oracle.control_fired(ran))

    def test_a_completed_report_is_vacuous_too(self):
        completed = oracle.evaluate(
            5,
            0,
            dict(exit=0, passed=True, stdout_sha256="0" * 64),
            "SUPERVISE exit status=0\n",
            report(),
        )
        self.assertFalse(oracle.control_fired(completed))


class ModeSelection(unittest.TestCase):
    def select(self, text):
        return oracle.numbers(text, {0, 1}, "mode", oracle.MODE_NAMES)

    def test_numbers_and_arm_names_mean_the_same(self):
        self.assertEqual(self.select("0,1"), self.select("spatial,protected"))

    def test_the_other_targets_arm_name_is_refused(self):
        with self.assertRaises(SystemExit):
            self.select("sublet")


class Fixture(unittest.TestCase):
    def test_one_event_and_the_case_number_in_it(self):
        blob = oracle.fixture(13, False)
        self.assertEqual(len(blob), oracle.FIXTURE_BYTES)
        self.assertEqual(struct.unpack("<12Q", blob[:96])[1], 1)
        self.assertEqual(struct.unpack("<4Q", blob[96:])[1], 13)

    def test_the_negative_control_declares_an_event_it_does_not_carry(self):
        self.assertEqual(struct.unpack("<12Q", oracle.fixture(13, True)[:96])[1], 2)


if __name__ == "__main__":
    unittest.main()
