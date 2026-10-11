#!/usr/bin/env python3
"""Trigger for gh140607 -- the reproducer from upstream issue #140607.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (8193-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.

The upstream reproducer has readinto return 2147483647. That number is not the
defect -- the defect is that BufferedReader trusts whatever readinto returns --
and in the Capstone domain it asks for 2 GiB, gets MemoryError, and the overflow
never happens: the row read CAPACITY on both arms and was excluded as not a
measurement. The 192 MiB image did not help and could not have.

1048576 was chosen by measuring both builds, because the two answer different
questions and only one of them is the domain:

  lie        ASan build            plain build (what the domain runs)
  2147483647 heap-buffer-overflow  SIGSEGV
  16777216   heap-buffer-overflow  SIGSEGV
  1048576    heap-buffer-overflow  SIGSEGV
  65536      heap-buffer-overflow  HANGS
  8194       heap-buffer-overflow  HANGS

Every value overflows, and every value "terminates" under ASan because ASan
aborts at the first overflowing byte. Without bounds checking the loop only
ends if the claimed copy runs off into unmapped memory, which needs about a
megabyte. 8194 was tried first on the ASan evidence alone and would have
traded CAPACITY for TIMEOUT.

So 1048576: it overflows the same 8193-byte region, it terminates with or
without bounds checking, and it asks the domain for 1 MiB of a 48 MiB heap.
The report on the pinned ASan build is unchanged --

  AddressSanitizer: heap-buffer-overflow   (8193-byte region)
  in _io__Buffered_read_impl

-- so nothing about the defect is given up. The upstream value is kept in the
source line so the departure is visible.
"""
import io
import unittest
class CTestCase(unittest.TestCase):
    pass
class BufferedReaderTest:
    read_mode = 'rb'
class MockRaw(io.RawIOBase):
    def __init__(self, data=r'\n\r\t'):
        self._buf = memoryview(data)
        self._pos = 0
    def readable(self):
        return True
    def readinto(self, b):
        if self._pos >= len(self._buf):
            return 1048576   # upstream: 2147483647; see the note above
        n = min(len(b), len(self._buf) - self._pos)
        self._pos += n
        return n
class CBufferedReaderTest(BufferedReaderTest, CTestCase):
    tp = io.BufferedReader
    def test_initialization(self):
        rawio = MockRaw(b'abc')
        bufio = self.tp(rawio)
        self.assertEqual(bufio.read(), b'abc')
if __name__ == "__main__":
    unittest.main()

