#!/usr/bin/env python3
"""Trigger for gh142831 -- the reproducer from upstream issue #142831.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (16-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import json

items_list = [("1337", object())]

class Dict(dict):
   def items(self):
       return items_list

def encode_str(_):
   items_list.clear()
   return ''

encoder = json.encoder.c_make_encoder(
   None,
   lambda o: 0,
   encode_str,
   None, ": ", ", ", False, False, True,
)

encoder(Dict(a=1), 0)

