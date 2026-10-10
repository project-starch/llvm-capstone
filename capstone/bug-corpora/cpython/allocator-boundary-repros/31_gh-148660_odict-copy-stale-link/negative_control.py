#!/usr/bin/env python3
"""Strong negative control for case 31.

Same allocation and free traffic as trigger.py: the same upstream test file is
run through the same runpy/unittest path with no method named, as the trigger
names none. The one difference is that neither gh-148660 test clears the
OrderedDict from inside __eq__ or __getitem__, so copy() walks a linked list
that stays live; the same two dicts are built, the same copy() is called, and
the other tests in the file run exactly as the trigger runs them.

A fault here is a fault that does not depend on the defect, so it disqualifies
this case's detection rather than confirming it. The control self-tests: if
unittest reports a failure this control did not expect, the control is wrong,
not the arm, and it says so and exits 3.

Two failures ARE expected inside the Capstone domain. test_sizeof_exact
asserts exact sys.getsizeof values, and capability pointers widen every object,
so it fails on this substrate for reasons that have nothing to do with the
control; the trigger's own run of this file fails them too. They are allowed by
name rather than ignored by rule.
"""
import io, runpy, sys, os

ALLOWED = ("test_sizeof_exact",)


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
sys.argv = ["negative_control_test.py", "-v"]

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
print("NEGATIVE-CONTROL no defect performed")

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
