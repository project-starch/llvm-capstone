#!/usr/bin/env python3
"""Strong negative control for case 15.

Same allocation and free traffic as trigger.py: the same upstream test file is
run through the same runpy/unittest path with the same test method selected. The one
difference is that LOAD_CONST names index 1 instead of index 2, which is inside the two-element consts list; the same optimize_cfg call on the same instruction sequence still runs.

A fault here is a fault that does not depend on the defect, so it disqualifies
this case's detection rather than confirming it. The control self-tests: if
unittest does not report success the control is wrong, not the arm, and it says
so and exits 3.
"""
import runpy, sys, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['negative_control_test.py', '-v', 'DirectCfgOptimizerTests.test_optimize_cfg_const_index_out_of_range']
code = 0
try:
    runpy.run_path("negative_control_test.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
print('NEGATIVE-CONTROL no defect performed')
if code:
    print("NEGATIVE-CONTROL SELFTEST-FAILED unittest rc=%r" % (code,))
    sys.exit(3)
