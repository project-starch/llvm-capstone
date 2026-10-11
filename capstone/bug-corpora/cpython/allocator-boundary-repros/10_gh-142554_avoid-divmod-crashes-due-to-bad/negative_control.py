#!/usr/bin/env python3
"""Strong negative control for case 10.

Same allocation and free traffic as trigger.py: the same upstream test file is
run through the same runpy/unittest path with the same test method selected. The one
difference is that the replacement _pylong.int_divmod returns (1, 2) instead of (1,), so CPython's read of element [1] is in bounds; the monkeypatch, the 2*10000-bit operands and the _pylong path are unchanged.

A fault here is a fault that does not depend on the defect, so it disqualifies
this case's detection rather than confirming it. The control self-tests: if
unittest does not report success the control is wrong, not the arm, and it says
so and exits 3.
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['negative_control_test.py', '-v', 'PyLongModuleTests.test_pylong_int_divmod_crash']
code = 0
try:
    runpy.run_path("negative_control_test.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
print('NEGATIVE-CONTROL no defect performed')
if code:
    print("NEGATIVE-CONTROL SELFTEST-FAILED unittest rc=%r" % (code,))
    sys.exit(3)
