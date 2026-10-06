#!/usr/bin/env python3
"""Trigger for gh144169 -- the upstream fix's own regression test.

Upstream fix: bc92e7878f25
Test file at the fix: Lib/test/test_ast/test_ast.py
Test: ASTConstructorTests.test_non_str_kwarg

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "ASTConstructorTests.test_non_str_kwarg"]
runpy.run_path("upstream_test.py", run_name="__main__")
