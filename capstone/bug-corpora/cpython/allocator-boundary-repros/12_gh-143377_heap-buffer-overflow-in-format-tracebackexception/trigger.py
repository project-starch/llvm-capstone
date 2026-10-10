#!/usr/bin/env python3
"""Trigger for gh143377 -- the reproducer from upstream issue #143377.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (1-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import traceback
import _interpreters

orig_format = traceback.TracebackException.format

def empty_format(self):
    return []

traceback.TracebackException.format = empty_format

try:
    raise ValueError("boom")
except Exception as exc:
    _interpreters.capture_exception(exc)
finally:
    traceback.TracebackException.format = orig_format

