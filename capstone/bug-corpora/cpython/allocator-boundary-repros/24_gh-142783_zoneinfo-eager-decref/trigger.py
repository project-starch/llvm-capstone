#!/usr/bin/env python3
"""Trigger for gh142783 -- the reproducer from upstream issue #142783.

Run on the pinned build it reports:
  AddressSanitizer: heap-use-after-free   (328-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
from zoneinfo import ZoneInfo

class Cache:
    def get(self, *args, **kwargs):
        return None
    def setdefault(self, *args, **kwargs):
        return None
    def clear(self, *args, **kwargs):
        pass

class BombDescriptor:
    def __get__(self, obj, owner):
        return Cache()

class EvilZoneInfo(ZoneInfo):
    pass

EvilZoneInfo._weak_cache = BombDescriptor()

EvilZoneInfo("UTC")

