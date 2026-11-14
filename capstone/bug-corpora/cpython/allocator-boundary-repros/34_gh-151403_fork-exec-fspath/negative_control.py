#!/usr/bin/env python3
"""Strong negative control for case 34: the same call, with the Evil item kept alive.

__fspath__ still clears args and still returns a non-path, so PyOS_FSPath takes the same error
path and fork_exec raises the same TypeError; the one difference is that `keep` holds a second
reference, so clearing args frees nothing. A fault here does not depend on the defect.
"""
import _posixsubprocess


class Evil:
    def __fspath__(self):
        args.clear()
        return 12345


keep = Evil()
args = [keep]
try:
    _posixsubprocess.fork_exec(
        args, [b"false"],
        True, (), None, [b"env"],
        -1, -1, -1, -1,
        1, 2, 3, 4,
        True, True, 0,
        None, None, None, -1,
        None, False)   # 3.13 takes allow_vfork as a 23rd argument, which main later dropped
except TypeError:
    print("NEGATIVE-CONTROL no defect performed")
else:
    print("NEGATIVE-CONTROL SELFTEST-FAILED fork_exec returned")
    raise SystemExit(3)
