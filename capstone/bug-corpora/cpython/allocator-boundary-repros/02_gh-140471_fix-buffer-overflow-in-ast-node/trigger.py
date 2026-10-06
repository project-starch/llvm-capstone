#!/usr/bin/env python3
"""Trigger for gh140471 -- the upstream fix's own regression test.

Upstream fix: 1cc2c954d6b5
Test file at the fix: Lib/test/test_ast/test_ast.py
Test: ASTConstructorTests.test_malformed_fields_with_bytes

Run with the pinned 3.13.7 host-oracle interpreter. The test METHOD is
named explicitly so the run cannot pass by skipping it; the harness
requires the interpreter to confirm "Ran 1 test".
"""
import runpy, sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "ASTConstructorTests.test_malformed_fields_with_bytes"]
runpy.run_path("upstream_test.py", run_name="__main__")
