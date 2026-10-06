#!/usr/bin/env python3
"""Trigger for gh157325 -- the upstream fix's own regression test.

Upstream fix: af1be50e4cfd
Test file at the fix: Lib/test/test_multibytecodec.py
Test: Test_IncrementalDecoder.test_hz_keep_buffer

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "Test_IncrementalDecoder.test_hz_keep_buffer"]
runpy.run_path("upstream_test.py", run_name="__main__")
