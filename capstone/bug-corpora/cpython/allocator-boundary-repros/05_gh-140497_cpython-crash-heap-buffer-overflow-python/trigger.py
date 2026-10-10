#!/usr/bin/env python3
"""Trigger for gh140497 -- the reproducer from upstream issue #140497.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (200-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
def f():
    pass
f.__code__ = f.__code__.replace(co_code=b"")
f()

