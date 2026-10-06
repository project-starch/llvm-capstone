#!/usr/bin/env python3
"""Trigger for gh143004 -- the upstream fix's own regression test.

Upstream fix: 4dd9d94619be
Test file at the fix: Lib/test/test_collections.py
Test: TestCounter.test_update_reentrant_add_clears_counter

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "TestCounter.test_update_reentrant_add_clears_counter"]
runpy.run_path("upstream_test.py", run_name="__main__")
