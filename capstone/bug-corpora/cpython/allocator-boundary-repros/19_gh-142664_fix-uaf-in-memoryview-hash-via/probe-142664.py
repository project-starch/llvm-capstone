#!/usr/bin/env python3
"""Diagnostic, not a trigger and not a control.

This case's trigger runs the whole upstream file, which carries regression
tests for two live defects: gh-142664, which the case declares, and gh-143195,
a use-after-free in memoryview.hex(sep) via a re-entrant sep.__len__. A
whole-file fault therefore does not say which one it belongs to.

This selects gh-142664's tests only, by name, so a fault here is that defect's.
The companion probe-143195.py does the converse. Neither replaces trigger.py:
run them with TRIGGER=, which the harness records in run.meta.
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ["upstream_test.py", "-v", "-k", "hash_use_after_free"]
try:
    runpy.run_path("upstream_test.py", run_name="__main__")
except SystemExit as exc:
    print("PROBE-DONE gh-142664 only, unittest rc=%r" % (exc.code,))
else:
    print("PROBE-DONE gh-142664 only, unittest rc=0")
