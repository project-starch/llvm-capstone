#!/usr/bin/env python3
"""Trigger for gh143543 -- the upstream fix's own regression test.

Upstream fix: d177460b4317
Test file at the fix: Lib/test/test_itertools.py
Test: TestBasicOps.test_groupby_reentrant_eq_does_not_crash

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "TestBasicOps.test_groupby_reentrant_eq_does_not_crash"]
runpy.run_path("upstream_test.py", run_name="__main__")
