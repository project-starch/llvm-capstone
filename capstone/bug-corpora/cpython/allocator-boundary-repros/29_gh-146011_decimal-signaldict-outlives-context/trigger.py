#!/usr/bin/env python3
"""Trigger for gh146011 -- the reproducer from upstream issue #146011.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (112-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import decimal

ctx = decimal.Context(prec=7)
mapping = ctx.flags
del ctx
print(mapping)

