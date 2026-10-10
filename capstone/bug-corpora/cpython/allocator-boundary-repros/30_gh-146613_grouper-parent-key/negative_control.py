#!/usr/bin/env python3
"""Strong negative control for case 30.

Same allocation and free traffic as trigger.py: the same upstream test file is
run through the same runpy/unittest path with the same test method selected. The one
difference is that no key arms do_advance, so __eq__ is still called on every comparison and still allocates, but never advances the grouper from inside the comparison; groupby, the Key class and the grouper are otherwise unchanged.

A fault here is a fault that does not depend on the defect, so it disqualifies
this case's detection rather than confirming it. The control self-tests: if
unittest does not report success the control is wrong, not the arm, and it says
so and exits 3.
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['negative_control_test.py', '-v', 'TestBasicOps.test_grouper_reentrant_eq_does_not_crash']
code = 0
try:
    runpy.run_path("negative_control_test.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
print('NEGATIVE-CONTROL no defect performed')
if code:
    print("NEGATIVE-CONTROL SELFTEST-FAILED unittest rc=%r" % (code,))
    sys.exit(3)
