#!/usr/bin/env python3
"""Trigger for gh156075 -- the reproducer from upstream issue #156075.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (37-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import types
def f():
    raise ValueError("x")
c = f.__code__.replace(co_linetable=b'\xf0\xff\xff\xff')
g = types.FunctionType(c, {})
try:
    g()
except ValueError:
    import traceback; traceback.print_exc()
