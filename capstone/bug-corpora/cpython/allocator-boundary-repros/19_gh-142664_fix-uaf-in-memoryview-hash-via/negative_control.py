#!/usr/bin/env python3
"""Strong negative control for case 19, scoped to the case's own defect.

A whole-file control is NOT achievable here, and that is a measured fact
rather than a judgement. trigger.py runs the whole upstream file, and the
sublet arm faults at three distinct sites inside it:

  pc=0xc05c3f70  gh-142664, this case's defect (memoryview.__hash__)
  pc=0xc06226cc  gh-143195, a use-after-free in memoryview.hex(sep) via a
                 re-entrant sep.__len__ -- a different issue, also live here
  pc=0xc034a960  OtherTest.test_use_released_memory, gh-92888's regression
                 test, which PASSES on the pinned host build and produces no
                 ASan report with pymalloc or with PYTHONMALLOC=malloc

The third site is not a defect this corpus holds and no host oracle flags it,
so no amount of editing the file's defects makes a whole-file control silent.

This control is therefore paired with probe-142664.py instead of with
trigger.py: both select the same six tests by name, the probe with the
re-entrant release in place and this control with it removed. A fault in the
probe and silence here is a complete qualification of the detection, and it
does not depend on the whole-file run at all.
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
sys.argv = ["negative_control_test.py", "-v", "-k", "hash_use_after_free"]

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
if "Ran 6 tests" not in text:
    bad.append("expected exactly the six hash_use_after_free tests")
if bad:
    print("NEGATIVE-CONTROL SELFTEST-FAILED unittest rc=%r" % (code,))
    for ln in bad[:10]:
        print("  " + ln)
    sys.exit(3)
