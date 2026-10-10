#!/usr/bin/env python3
"""Trigger for gh143308 -- the reproducer from upstream issue #143308.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (4097-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import io
import pickle

base = bytearray(b'A' * 0x1000)
pb = pickle.PickleBuffer(base)

class Evil:
    def __bool__(self):
        global base
        pb.release()
        base = None
        return True

def callback(pb):
    return Evil()

pickle.dumps(pb, protocol=5, buffer_callback=callback)

