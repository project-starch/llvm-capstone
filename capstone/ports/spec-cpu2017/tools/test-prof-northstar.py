#!/usr/bin/env python3
"""Accounting and refusal regressions; run with python3 on the measurement host."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "northstar", Path(__file__).with_name("prof-northstar.py"))
ns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ns)


class NorthstarTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('CAPSTONE_TMP_ROOT'))
        self.addCleanup(self.tmp.cleanup)
        self.a = Path(self.tmp.name) / "a.json"
        self.b = Path(self.tmp.name) / "b.json"
        self.write(self.a, 100, 50, 25)
        self.write(self.b, 130, 65, 900)

    def write(self, path, p, k, h, **extra):
        path.write_text(json.dumps(dict(total_attempts=p+k+h,
            all_functions=[["program", "main", p], ["module", "cv_step", k],
                           ["kernel", "do_idle", h]], **extra)))

    def test_denominator_excludes_harness(self):
        r = ns.compare(self.a, self.b)
        self.assertEqual((r["program_pct"], r["boundary_pct"]), (20, 10))
        self.assertEqual(r["b"]["H"], 900)

    def test_missing_and_incomplete_data(self):
        self.a.unlink()
        with self.assertRaisesRegex(ValueError, "missing bucket"):
            ns.compare(self.a, self.b)
        self.a.write_text('{"total_attempts": 10, "all_functions": []}')
        with self.assertRaisesRegex(ValueError, "sum"):
            ns.compare(self.a, self.b)

    def test_slice_work_must_match(self):
        self.write(self.a, 100, 50, 25, slice={"symbol": "step", "count": 2})
        with self.assertRaisesRegex(ValueError, "unequal work"):
            ns.compare(self.a, self.b)
        self.write(self.b, 130, 65, 25, slice={"symbol": "step", "count": 3})
        with self.assertRaisesRegex(ValueError, "unequal work"):
            ns.compare(self.a, self.b)
        self.write(self.b, 130, 65, 25, slice={"symbol": "step", "count": 2})
        self.assertEqual(ns.compare(self.a, self.b)["work"]["count"], 2)

    def test_failed_run_and_empty_denominator(self):
        result = self.a.with_name("result.json")
        result.write_text('{"verdict": "FAIL"}')
        with self.assertRaisesRegex(ValueError, "verdict FAIL"):
            ns.compare(self.a, self.b)
        result.unlink()
        self.write(self.a, 0, 0, 25)
        with self.assertRaisesRegex(ValueError, "no program"):
            ns.compare(self.a, self.b)


if __name__ == "__main__":
    unittest.main()
