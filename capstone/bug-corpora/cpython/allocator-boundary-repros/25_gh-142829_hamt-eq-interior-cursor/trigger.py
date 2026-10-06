#!/usr/bin/env python3
"""Trigger for gh142829 -- the upstream fix's own regression test.

Upstream fix: 149ecbb9a9e6
Test file at the fix: Lib/test/test_context.py
Test: ContextTest.test_context_eq_reentrant_contextvar_set

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "ContextTest.test_context_eq_reentrant_contextvar_set"]
runpy.run_path("upstream_test.py", run_name="__main__")
