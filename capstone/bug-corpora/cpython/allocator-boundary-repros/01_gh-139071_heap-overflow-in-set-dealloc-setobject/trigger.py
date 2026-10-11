#!/usr/bin/env python3
"""Trigger for gh139071 -- the reproducer from upstream issue #139071.

Run on the pinned build it reports:
  AddressSanitizer: heap-buffer-overflow   (216-byte region)
Needs ASAN_OPTIONS=detect_leaks=0 -- with leak checking on, the leak
summary buries the report.
"""
import _asyncio

class dummy(object):
    def __hash__(self):
        return 0
class CorrupTrigger():
    def __hash__(self):
        return 0
    def __eq__(self,other):
        try:
            _asyncio._register_task(self)
        except:
            pass
        return False
a = [dummy(),dummy()]
_asyncio._register_task(a[0])
_asyncio._register_task(a[1])
del a # make dummy entries in set 
print (_asyncio._scheduled_tasks)
a = [CorrupTrigger() for i in range(2)]
_asyncio._register_task(a[0])
_asyncio._register_task(a[1]) # make recursive comparision


