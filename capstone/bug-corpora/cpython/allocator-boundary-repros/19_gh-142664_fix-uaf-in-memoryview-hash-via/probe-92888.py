#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

OtherTest.test_use_released_memory is gh-92888's regression test: a memoryview
whose backing bytearray is freed from inside __index__, then read through.
It PASSES on the pinned host build, and the sublet arm faults on it at
pc=0xc034a960 with cause 24.

The nesting test decides whether that is the arm seeing a defect the host
sanitizer cannot: run this on the ASan build twice, once with pymalloc active
and once with PYTHONMALLOC=malloc. A report that appears only without pymalloc
means the freed block was reused from a pool, which is exactly what hides a
nested use-after-free from ASan and what sublet's per-block revocation exposes.
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "-k", "test_use_released_memory"]
try:
    runpy.run_path("upstream_test.py", run_name="__main__")
except SystemExit as exc:
    print("PROBE-DONE gh-92888 only, unittest rc=%r" % (exc.code,))
else:
    print("PROBE-DONE gh-92888 only, unittest rc=0")
