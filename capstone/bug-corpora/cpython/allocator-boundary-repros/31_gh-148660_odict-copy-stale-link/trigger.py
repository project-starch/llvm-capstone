#!/usr/bin/env python3
"""Trigger for gh148660 -- the upstream fix's own regression test.

Upstream fix: 9dc1ae091838
Test file at the fix: Lib/test/test_ordered_dict.py
Test: (whole file)

Run with the pinned 3.13.7 host-oracle interpreter. The harness runs the
upstream file as a whole. No method is named, because this case was migrated
without one, so a verdict here is about the whole file and not about one test.
Selecting a method would re-measure the case, and the earlier claim that the
harness "requires the interpreter to confirm Ran 1 test" is withdrawn rather
than enforced. The method that reproduces the defect is
CPythonOrderedDictTests.test_issue148660_copy_clear_in_key_eq and
CPythonOrderedDictTests.test_issue148660_copy_clear_in_subclass_getitem
(upstream_test.py:877, :899); which of the two produces the ASan report is
not settled. A re-run naming it is pending.
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v"]
runpy.run_path("upstream_test.py", run_name="__main__")
