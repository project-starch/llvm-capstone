#!/usr/bin/env python3
"""Trigger for gh-151403 -- the reproducer from upstream issue #151403.

One change: the issue was written against main, whose fork_exec takes 22 arguments; 3.13.7 takes
23 (allow_vfork last, Lib/subprocess.py:1902 at v3.13.7), so a False is appended. Without it the
call fails its argument count before any argument is converted.

Upstream fix: 5810def6dd ([3.13] back-port of 6679ac07d8); the fix adds no test.
The conversion of args happens before any fork, in the parent: the error path in PyOS_FSPath reads
the type of the item __fspath__ just freed, and fork_exec then raises TypeError.
"""
import _posixsubprocess


class Evil:
    def __fspath__(self):
        args.clear()      # drop the args list's last reference to self
        return 12345      # non-str/bytes -> takes the error path in PyOS_FSPath


args = [Evil()]
try:
    _posixsubprocess.fork_exec(
        args, [b"false"],
        True, (), None, [b"env"],
        -1, -1, -1, -1,
        1, 2, 3, 4,
        True, True, 0,
        None, None, None, -1,
        None, False)   # 3.13 takes allow_vfork as a 23rd argument, which main later dropped
except TypeError as exc:
    print("TRIGGER TypeError:", exc)
else:
    raise SystemExit("TRIGGER fork_exec returned: the defect path was not taken")
