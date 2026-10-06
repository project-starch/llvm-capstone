#!/usr/bin/env python3
"""Trigger for gh146613 -- the upstream fix's own regression test.

Upstream fix: 7fc8507f3d52
Test file at the fix: Lib/test/test_itertools.py
Test: TestBasicOps.test_grouper_reentrant_eq_does_not_crash

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "TestBasicOps.test_grouper_reentrant_eq_does_not_crash"]
runpy.run_path("upstream_test.py", run_name="__main__")
