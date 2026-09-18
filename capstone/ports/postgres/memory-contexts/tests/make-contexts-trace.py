#!/usr/bin/env python3
"""Synthetic A11 replay covering all four managers, including direct delete."""

import importlib.util
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location(
    "fixture", Path(__file__).with_name("make-fixture-trace.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class Trace(fixture.Trace):
    def manager(self, kind, parent):
        c = self.ctx("mixed-context", parent)
        op, i, name, parent, minimum, initial, maximum = self.recs[-1]
        if kind == 8:  # Slab: block size, fixed chunk size, unused.
            minimum, initial, maximum = 8192, 64, 0
        self.recs[-1] = (kind, i, name, parent, minimum, initial, maximum)
        return c


t = Trace()
top = t.ctx("TopMemoryContext")
for kind in (7, 8, 9):
    c = t.manager(kind, top)
    sibling = t.manager(kind, top)
    held = t.alloc(sibling, 64)
    for cycle in range(4):
        objects = [t.alloc(c, 64) for _ in range(256)]
        if kind != 9:
            for obj in objects[::2]:
                t.free(obj)
            for _ in range(128):
                t.alloc(c, 64)
            t.realloc(objects[1], 128 if kind == 7 else 64)
        if kind != 8:
            large = t.alloc(c, 40000)
            if kind == 7:
                t.realloc(large, 60000)
        t.reset(c)
    t.alloc(c, 64)
    t.delete(c)  # live direct destroy, not a preceding reset
    if kind != 9:
        t.free(held)  # sibling still usable after unrelated delete
    t.delete(sibling)
t.delete(top)
# PostgreSQL 17.0, native release layout; check-native.py mutates this oracle
# to prove that backing-count and realloc drift are both rejected.
blocks = (49, 48, 0, 7)
if len(sys.argv) > 2:
    blocks = tuple(map(int, sys.argv[2].split(",")))
print(f"mixed contexts: {t.write(sys.argv[1], blocks)} records")
