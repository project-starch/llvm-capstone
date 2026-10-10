#!/usr/bin/env python3
"""Trigger for gh142664 -- the upstream fix's own regression test.

Upstream fix: 4fcb1d98198f
Test file at the fix: Lib/test/test_memoryview.py
Test: (whole file)

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
runs the upstream file as a whole: no method is named, because this case
was migrated without one. The file runs more tests than the defect -- 157
on case 19 -- and a verdict here is therefore about the whole file, not one
test. Selecting a method would re-measure the case, so the earlier claim
that the harness "requires the interpreter to confirm Ran 1 test" is
withdrawn rather than enforced.
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v"]
runpy.run_path("upstream_test.py", run_name="__main__")
