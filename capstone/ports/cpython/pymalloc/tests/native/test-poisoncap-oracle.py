#!/usr/bin/env python3
"""Reject incomplete/wrong-mode replays without requiring cross-ABI layouts."""

import importlib.util
from pathlib import Path
import struct
import unittest

path = Path(__file__).resolve().parents[2] / "host/cheribsd/poisoncap/run.py"
spec = importlib.util.spec_from_file_location("poisoncap_runner", path)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReplayOracleTests(unittest.TestCase):
    def setUp(self):
        self.native = [runner.MAGIC, 10, 0, 0, 10, 4, 4, 1, 1, 0, 1024, 73]
        self.actual = self.native.copy()
        self.actual[2] = 1

    def check(self, actual):
        return runner.validate_replay(
            struct.pack("<12Q", *actual), struct.pack("<12Q", *self.native), 1
        )

    def test_layout_counters_may_differ(self):
        self.actual[8:11] = [2, 1, 8192]
        self.assertTrue(self.check(self.actual)["logical_fields_match"])

    def test_wrong_mode_failure_counts_checksum_and_completion_rejected(self):
        for field in (0, 1, 2, 3, 4, 5, 6, 7, 11):
            broken = self.actual.copy()
            broken[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(broken)

    def test_short_or_failed_reference_rejected(self):
        with self.assertRaises(ValueError):
            runner.validate_replay(b"short", struct.pack("<12Q", *self.native), 1)
        self.native[3] = 1
        with self.assertRaises(ValueError):
            self.check(self.actual)


if __name__ == "__main__":
    unittest.main()
