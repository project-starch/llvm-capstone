#!/usr/bin/env python3
"""Positive and negative controls for the exact-access classifier."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "security-tests"))
from context_checks import classify


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.serial = (
            "Print = Scalar(0xcf17000000000001)\n"
            "Print = Cap(0, 0x100, 0x123,\n"
            "Print = Cap(0, 0x100, 0x456,\n"
            "domain halted by capability fault: cause = 24, pc = 0x123\n"
        )

    def test_fault_and_rejections(self):
        self.assertTrue(classify(self.serial, 0, 1, "sublet", 1)["passed"])
        for serial in (
            "",
            self.serial.replace("0xcf17", "0xcf18"),
            self.serial.replace("pc = 0x123", "pc = 0x456"),
            self.serial.replace("cause = 24", "cause = 5"),
            self.serial + "__CAPSTONE_PG_CONTEXTS_FAILED__",
            self.serial + self.serial,
        ):
            with self.subTest(serial=serial):
                self.assertFalse(classify(serial, 0, 1, "sublet", 1)["passed"])
        self.assertFalse(classify(self.serial, 0, 1, "spatial", 1)["passed"])

    def test_write_and_bounds_sites(self):
        write = self.serial.replace("000001)", "000005)").replace(
            "pc = 0x123", "pc = 0x456"
        )

        self.assertTrue(classify(write, 0, 5, "sublet", 1)["passed"])
        self.assertFalse(
            classify(write.replace("pc = 0x456", "pc = 0x123"), 0, 5, "sublet", 1)[
                "passed"
            ]
        )
        bounds = self.serial.replace("000001)", "000009)").replace(
            "cause = 24", "cause = 5"
        )
        self.assertTrue(classify(bounds, 0, 9, "sublet", 1)["passed"])
        self.assertFalse(
            classify(bounds.replace("cause = 5", "cause = 24"), 0, 9, "sublet", 1)[
                "passed"
            ]
        )

    def test_delivered_fault_requires_process_exit(self):
        serial = self.serial.replace(
            "halted by capability fault", "capability fault delivered"
        )
        self.assertFalse(classify(serial, 0, 1, "sublet", 1)["passed"])
        serial += "__CAPSTONE_PG_DOMAIN_FAULT__\n__EXIT_CODE__139\n"
        self.assertTrue(classify(serial, 0, 1, "sublet", 1)["passed"])
        self.assertFalse(
            classify(serial.replace("139", "0"), 0, 1, "sublet", 1)["passed"]
        )

    def test_control(self):
        serial = (
            "Print = Scalar(0xcf17000000000000)\n" "__CAPSTONE_PG_CONTEXTS_GOOD__\n"
        )
        self.assertTrue(classify(serial, 0, 0, "sublet", 0)["passed"])
        self.assertFalse(classify(serial, 0, 0, "sublet", 1)["passed"])
        self.assertFalse(classify(serial, 1, 0, "sublet", 0)["passed"])
        self.assertFalse(classify("", 0, 0, "sublet", 0)["passed"])

    def test_policy(self):
        serial = "PG_CONTEXT_RESULT 0 POLICY 123\n__CAPSTONE_PG_CONTEXTS_GOOD__"
        self.assertEqual(classify(serial, 0, 10, "sublet", 0)["policy"], 123)
        self.assertTrue(classify(serial, 0, 10, "sublet", 0)["passed"])
        self.assertFalse(
            classify(serial.replace("RESULT 0", "RESULT 1"), 0, 10, "sublet", 0)[
                "passed"
            ]
        )
        self.assertFalse(classify(serial + serial, 0, 10, "sublet", 0)["passed"])


unittest.main()
