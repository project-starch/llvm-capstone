#!/usr/bin/env python3
"""Trigger for gh157335 -- the upstream fix's own regression test.

Upstream fix: edabbff90bc6
Test file at the fix: Lib/test/test_mmap.py
Test: test_setitem_resize_reentrancy  (selected with -k so it is found in whichever
concrete TestCase subclass defines or inherits it)

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
selected by pattern so the run cannot quietly measure some other test. Nothing
in probe/ checks that at least one test ran, although an earlier version of
this docstring said the harness did. A skipped method shows only in the case's
own log.
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "-k", "test_setitem_resize_reentrancy"]
runpy.run_path("upstream_test.py", run_name="__main__")
