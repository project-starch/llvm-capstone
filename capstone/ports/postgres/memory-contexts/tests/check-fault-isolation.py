#!/usr/bin/env python3
"""Negative controls for the process-isolation and quarantine verdicts."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "security-tests/run-fault-isolation.py"
spec = importlib.util.spec_from_file_location("fault_isolation", path)
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def delivery(pc="0x100", cause=24):
    return f"domain capability fault delivered: cause = {cause}, pc = {pc}\n"


def site(pc="0x100"):
    return f"Print = Cap(1, 0x0, {pc}, 0x200)\n"


class Verdicts(unittest.TestCase):
    def setUp(self):
        self.reentry = (
            site()
            + delivery()
            + "".join(
                f"FAULT_REENTRY {i} RESULT {checks.SENTINEL} ENTERED 1 AFTER 0\n"
                for i in range(3)
            )
            + "__CAPSTONE_FAULT_REENTRY_DONE__\n"
        )
        healthy = "FAULT_ISOLATION HEALTHY\nPG_CLIENT generation RESULT 0\n"
        self.isolation = healthy
        for kind in range(3):
            self.isolation += (
                f"Print = Scalar(0x{0xcf17000000000000 | (kind << 8) | 2:x})\n"
                + site()
                + site("0x180")
                + delivery()
                + "__CAPSTONE_PG_DOMAIN_FAULT__\nFAULT_ISOLATION CHILD SIGSEGV\n"
            )
        self.isolation += healthy + "__CAPSTONE_FAULT_ISOLATION_DONE__\n"

    def test_valid(self):
        self.assertTrue(checks.reentry_passed(self.reentry, 1, 0))
        self.assertTrue(checks.isolation_passed(self.isolation, 0))

    def test_no_data_and_runner_failure(self):
        for text in ("", self.reentry):
            self.assertFalse(checks.reentry_passed(text, 1, 1))
        self.assertFalse(checks.reentry_passed("", 1, 0))
        self.assertFalse(checks.isolation_passed("", 0))
        self.assertFalse(checks.isolation_passed(self.isolation, 1))

    def test_reentry_mutations(self):
        for text in (
            self.reentry.replace("ENTERED 1", "ENTERED 2"),
            self.reentry.replace("AFTER 0", "AFTER 1"),
            self.reentry.replace("pc = 0x100", "pc = 0x104"),
            self.reentry.replace("cause = 24", "cause = 2"),
            self.reentry.replace("__CAPSTONE_FAULT_REENTRY_DONE__", ""),
            self.reentry + delivery(),
            self.reentry + "domain halted by capability fault",
        ):
            with self.subTest(text=text):
                self.assertFalse(checks.reentry_passed(text, 1, 0))

    def test_fallback(self):
        halted = "domain halted by capability fault: cause = 24, pc = 0x100\n"
        direct = site() + halted
        nested = site() + delivery() + halted.replace("0x100", "0x104")
        self.assertTrue(checks.fallback_passed(direct, 4, 1))
        self.assertTrue(checks.fallback_passed(nested, 5, 1))
        for text in ("", direct, nested + "Assertion", nested + halted):
            self.assertFalse(checks.fallback_passed(text, 5, 1))
        self.assertFalse(checks.fallback_passed(nested, 5, 0))
        self.assertFalse(checks.fallback_passed(nested, 4, 1))

    def test_isolation_mutations(self):
        markers = [
            f"Print = Scalar(0x{0xcf17000000000000 | (kind << 8) | 2:x})\n"
            for kind in range(3)
        ]
        misplaced = self.isolation
        for marker in markers:
            misplaced = misplaced.replace(marker, "")
        misplaced = "".join(markers) + misplaced
        for text in (
            misplaced,
            self.isolation.replace("CHILD SIGSEGV", "CHILD EXIT 139"),
            self.isolation.replace("__CAPSTONE_PG_DOMAIN_FAULT__", ""),
            self.isolation.replace("FAULT_ISOLATION HEALTHY", "", 1),
            self.isolation.replace("PG_CLIENT generation RESULT 0", ""),
            self.isolation.replace("pc = 0x100", "pc = 0x104"),
            self.isolation.replace("cause = 24", "cause = 2"),
            self.isolation.replace("__CAPSTONE_FAULT_ISOLATION_DONE__", ""),
            self.isolation + delivery(),
            self.isolation + "domain halted by capability fault",
            self.isolation.replace("FAULT_ISOLATION HEALTHY\n", "", 1)
            + "FAULT_ISOLATION HEALTHY\n",
        ):
            with self.subTest(text=text):
                self.assertFalse(checks.isolation_passed(text, 0))


if __name__ == "__main__":
    unittest.main()
