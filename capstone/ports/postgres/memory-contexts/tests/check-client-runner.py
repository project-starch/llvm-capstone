#!/usr/bin/env python3
"""Prove the example runner rejects failures and incomplete guest runs."""

import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "client_runner", Path(__file__).resolve().parents[1] / "examples/run-qemu.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ClientVerdicts(unittest.TestCase):
    def test_controls(self):
        good = "PG_CLIENT slab RESULT 0\n__CAPSTONE_PG_HOST_DONE__\n"
        self.assertTrue(runner.passed(good, "slab", 0))
        self.assertFalse(runner.passed(good, "slab", 1))
        self.assertFalse(runner.passed(good, "bump", 0))
        for bad in (
            "",
            good.replace("RESULT 0", "RESULT 1"),
            good.replace("__CAPSTONE_PG_HOST_DONE__", ""),
            good + good,
            good + "__CAPSTONE_PG_CLIENT_FAILED__",
            good + "domain halted by capability fault: cause = 24",
            good + "domain capability fault delivered: cause = 24",
        ):
            with self.subTest(serial=bad):
                self.assertFalse(runner.passed(bad, "slab", 0))


unittest.main()
