#!/usr/bin/env python3
"""Trigger for gh148973 -- the upstream fix's own regression test.

Upstream fix: 8da3d39c510d
Test file at the fix: Lib/test/test_peepholer.py
Test: DirectCfgOptimizerTests.test_optimize_cfg_const_index_out_of_range

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "DirectCfgOptimizerTests.test_optimize_cfg_const_index_out_of_range"]
runpy.run_path("upstream_test.py", run_name="__main__")
