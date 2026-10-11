#!/usr/bin/env python3
"""Trigger for gh140551 -- the reproducer from upstream issue #140551.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (24-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import unittest
class Test(unittest.TestCase):
    def test(self):
        class X(object):
            def __hash__(self):
                return -1
            def __eq__(self, other):
                    d.clear()
        d = {}
        d[X()] = 3
        d[X()] = 4
        d[0.0] = 6
        self.assertIn(9, d)
if __name__ == "__main__":
    unittest.main()

