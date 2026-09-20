#!/usr/bin/env python3
"""Show that the PoisonCap defect oracles can say FAIL.

Every check here is a negative control for one way a run could be mistaken for
protection: exit 162 on its own, a fault before the case marker, a fault of the
wrong kind, a fault at another instruction, a handler verdict that disagrees
with its own printed addresses, and a spatial arm that exits cleanly without
having completed the case.
"""

import importlib.util
from pathlib import Path
import struct
import unittest

path = Path(__file__).resolve().parents[2] / "runners/cheribsd/run-defects.py"
spec = importlib.util.spec_from_file_location("run_poisoncap", path)
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)

READY = "PYC_DEFECT case=7 ready"
FAULT = "PYC_DEFECT_FAULT case=7 signal=34 code=2 pc=0x10a5c expected=0x10a5c exact=1"
ROW = dict(exit=162, passed=True, stdout_sha256="0" * 64)


def report(mode=0, completed=1, count=1, status=0, magic=oracle.MAGIC):
    return struct.pack("<12Q", magic, count, mode, status, completed, *([0] * 7))


class FixtureTests(unittest.TestCase):
    def test_shape_is_one_event_in_a_hundred_and_twenty_eight_bytes(self):
        blob = oracle.fixture(13, False)
        self.assertEqual(len(blob), oracle.FIXTURE_BYTES)
        header = struct.unpack("<12Q", blob[: oracle.REPORT_BYTES])
        self.assertEqual(header[0], oracle.MAGIC)
        self.assertEqual(header[1], 1)
        self.assertEqual(struct.unpack("<4Q", blob[oracle.REPORT_BYTES :])[1], 13)

    def test_the_negative_control_declares_an_event_it_does_not_carry(self):
        blob = oracle.fixture(13, True)
        self.assertEqual(len(blob), oracle.FIXTURE_BYTES)
        self.assertEqual(struct.unpack("<12Q", blob[:96])[1], 2)


class ProtectedArmTests(unittest.TestCase):
    def evaluate(self, stdout, row=None):
        return oracle.evaluate(7, 1, row or ROW, stdout, None)

    def test_the_exact_fault_passes(self):
        self.assertTrue(self.evaluate(READY + "\n" + FAULT + "\n")["passed"])

    def test_sigprot_alone_is_not_protection(self):
        self.assertFalse(self.evaluate(READY + "\n", dict(ROW, passed=False))["passed"])

    def test_a_fault_before_the_case_marker_is_not_protection(self):
        self.assertFalse(self.evaluate(FAULT + "\n")["passed"])

    def test_a_bounds_fault_at_the_probe_is_not_protection(self):
        bounds = FAULT.replace("code=2", "code=1")
        self.assertFalse(self.evaluate(READY + "\n" + bounds + "\n")["passed"])

    def test_a_tag_fault_at_another_instruction_is_not_protection(self):
        elsewhere = FAULT.replace("pc=0x10a5c", "pc=0x10b40").replace(
            "exact=1", "exact=0"
        )
        self.assertFalse(self.evaluate(READY + "\n" + elsewhere + "\n")["passed"])

    def test_the_handler_verdict_must_agree_with_its_own_addresses(self):
        disagreeing = FAULT.replace("exact=1", "exact=0")
        self.assertFalse(self.evaluate(READY + "\n" + disagreeing + "\n")["passed"])

    def test_another_cases_fault_line_does_not_count(self):
        other = FAULT.replace("case=7", "case=6")
        self.assertFalse(self.evaluate(READY + "\n" + other + "\n")["passed"])

    def test_the_line_must_be_produced_not_echoed(self):
        self.assertFalse(self.evaluate(READY + "\necho " + FAULT + "\n")["passed"])

    def test_a_clean_exit_is_never_a_protected_result(self):
        row = dict(ROW, exit=0, passed=False)
        self.assertFalse(self.evaluate(READY + "\n" + FAULT + "\n", row)["passed"])

    def test_the_common_runners_own_oracle_still_has_to_hold(self):
        row = dict(ROW, passed=False)
        self.assertFalse(self.evaluate(READY + "\n" + FAULT + "\n", row)["passed"])

    def test_the_published_regex_requires_both_addresses_to_match(self):
        import re

        pattern = re.compile(oracle.fault_oracle(7))
        self.assertTrue(pattern.fullmatch(FAULT))
        self.assertFalse(
            pattern.fullmatch(FAULT.replace("expected=0x10a5c", "expected=0x10b40"))
        )


class SpatialArmTests(unittest.TestCase):
    stdout = "PYC_DEFECT case=7 ready\nPYC_DEFECT case=7 completed\n"
    row = dict(exit=0, passed=True, stdout_sha256="0" * 64)

    def evaluate(self, stdout=None, row=None, blob=None):
        return oracle.evaluate(
            7,
            0,
            row or self.row,
            self.stdout if stdout is None else stdout,
            report() if blob is None else blob,
        )

    def test_a_completed_case_with_a_matching_report_passes(self):
        self.assertTrue(self.evaluate()["passed"])

    def test_a_clean_exit_without_the_completion_marker_fails(self):
        self.assertFalse(self.evaluate(stdout="PYC_DEFECT case=7 ready\n")["passed"])

    def test_a_missing_report_fails(self):
        self.assertFalse(self.evaluate(blob=b"")["passed"])

    def test_a_truncated_report_fails(self):
        self.assertFalse(self.evaluate(blob=report()[:88])["passed"])

    def test_a_report_that_did_not_complete_fails(self):
        self.assertFalse(self.evaluate(blob=report(completed=0))["passed"])

    def test_a_report_from_the_other_mode_fails(self):
        self.assertFalse(self.evaluate(blob=report(mode=1))["passed"])

    def test_a_nonzero_status_fails(self):
        self.assertFalse(self.evaluate(blob=report(status=9))["passed"])

    def test_a_report_of_more_than_one_case_fails(self):
        self.assertFalse(self.evaluate(blob=report(count=2))["passed"])

    def test_a_foreign_report_fails(self):
        self.assertFalse(self.evaluate(blob=report(magic=1))["passed"])


class ModeSelectionTests(unittest.TestCase):
    """--modes takes the arm names each case.json declares, or the numbers the
    binary takes. A name from the OTHER target is not silently accepted."""

    def select(self, text):
        return oracle.numbers(text, {0, 1}, "mode", oracle.MODE_NAMES)

    def test_numbers_and_names_mean_the_same_thing(self):
        self.assertEqual(self.select("0,1"), self.select("spatial,protected"))

    def test_a_single_name_selects_one_arm(self):
        self.assertEqual(self.select("protected"), [1])
        self.assertEqual(self.select("spatial"), [0])

    def test_case_does_not_matter_and_duplicates_collapse(self):
        self.assertEqual(self.select("PROTECTED"), [1])
        self.assertEqual(self.select("1,protected"), [1])

    def test_order_is_preserved(self):
        self.assertEqual(self.select("protected,spatial"), [1, 0])

    def test_the_other_targets_arm_name_is_refused(self):
        # sublet is a Capstone arm; the CheriBSD protected arm is not Sublet.
        with self.assertRaises(SystemExit):
            self.select("sublet")
        with self.assertRaises(SystemExit):
            self.select("spatial,sublet")

    def test_an_out_of_range_number_and_an_empty_selection_are_refused(self):
        for text in ("2", "", "-1", "0,", "eins"):
            with self.subTest(text=text), self.assertRaises(SystemExit):
                self.select(text)


class CaseConstructionTests(unittest.TestCase):
    def test_both_arms_name_the_same_binary_and_differ_only_in_mode(self):
        spatial = oracle.build_case(3, 0, "/b/bin/defects", "/f/case-3.bin", 60)
        protected = oracle.build_case(3, 1, "/b/bin/defects", "/f/case-3.bin", 60)
        self.assertEqual(spatial["program"], protected["program"])
        self.assertEqual(spatial["inputs"], protected["inputs"])
        self.assertEqual(spatial["args"][:2], protected["args"][:2])
        self.assertEqual(spatial["args"][2], "0")
        self.assertEqual(protected["args"][2], "1")
        self.assertEqual(spatial["exit"], 0)
        self.assertEqual(protected["exit"], oracle.SIGPROT_EXIT)
        for case in (spatial, protected):
            self.assertEqual(case["also_expect"], ["PYC_DEFECT case=3 ready"])


if __name__ == "__main__":
    unittest.main()
