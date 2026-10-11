#!/usr/bin/env python3
"""Strong negative control for case 28.

Same allocation and free traffic as trigger.py: the same upstream test file is
run through the same runpy/unittest path with the same test method selected. The one
difference is that no key arms do_advance, so __eq__ is still called on every comparison but never advances the groupby from inside it; the same generator keyfunc, the same Key objects and the same two next() calls still run.

A fault here is a fault that does not depend on the defect, so it disqualifies
this case's detection rather than confirming it. The control self-tests: if
unittest reports a failure this control did not expect, the control is wrong,
not the arm, and it says so and exits 3.
"""
import io, runpy, sys, os

ALLOWED = ()


class Tee(io.TextIOBase):
    """unittest writes to stderr; keep every line in the log and read it back."""

    def __init__(self, real):
        self.real = real
        self.buf = []

    def write(self, s):
        self.real.write(s)
        self.buf.append(s)
        return len(s)

    def flush(self):
        self.real.flush()


os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv = ['negative_control_test.py', '-v', 'TestBasicOps.test_groupby_reentrant_eq_does_not_crash']

tee = Tee(sys.stderr)
saved = sys.stderr
sys.stderr = tee
code = 0
try:
    runpy.run_path("negative_control_test.py", run_name="__main__")
except SystemExit as exc:
    code = exc.code or 0
finally:
    sys.stderr = saved
    tee.flush()

text = "".join(tee.buf)
print('NEGATIVE-CONTROL no defect performed')

bad = [ln for ln in text.splitlines()
       if ln.startswith(("FAIL: ", "ERROR: "))
       and not any(a in ln for a in ALLOWED)]
if "Ran " not in text:
    bad.append("unittest never reported how many tests it ran")
if bad:
    print("NEGATIVE-CONTROL SELFTEST-FAILED unittest rc=%r" % (code,))
    for ln in bad[:10]:
        print("  " + ln)
    sys.exit(3)
