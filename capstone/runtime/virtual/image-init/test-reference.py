#!/usr/bin/env python3
"""Executable contract tests, with deliberately broken reference variants."""
import argparse
from contextlib import contextmanager
import subprocess
from reference import Capability, Grant, Image, Refused, Relocation, Symbol

p = argparse.ArgumentParser()
p.add_argument('--mutant', choices=('writable-table', 'weak-rebase', 'cursor-window',
                                  'destination-span', 'unlocked-publication'))
p.add_argument('--fault-peer', help='executable compiled against the fault lane header')
a = p.parse_args()
checks = 0

def check(ok):
    global checks
    checks += 1
    assert ok, f'check {checks}'

def refused(call):
    try:
        call()
    except Refused:
        check(True)
    else:
        check(False)

destination = Grant(7, 1, 1, 0x10000, 8192, 'rw')
source = Grant(7, 2, 1, 0x20000, 4096, 'r')
live = {destination, source}
held = set()
@contextmanager
def guard(namespace):
    check(namespace not in held)
    held.add(namespace)
    try:
        yield
    finally:
        held.remove(namespace)

def admit(g, base, size, access):
    check(g.namespace in held)
    return g in live and access in g.rights and g.base <= base and base + size <= g.base + g.size

def classify(event, cause):
    if a.fault_peer:
        value = subprocess.check_output([a.fault_peer, str(event), str(cause)], text=True).strip()
        return {'0': 'denied', '1': 'read', '2': 'write', '3': 'exec'}[value]
    return {13: 'read', 15: 'write', 12: 'exec'}.get(cause, 'denied') if event == 4 else 'denied'

symbols = [Symbol(source, 16, 16), Symbol(None, weak=True)]
records = [Relocation(0, 0), Relocation(4096, 1, -0x100000)]
def image(rs=None, ss=None, **kw):
    options = dict(version=1, destination=destination, symbols=symbols if ss is None else ss,
                   records=records if rs is None else rs, metadata_writable=False,
                   admit=admit, guard=guard, mutant=a.mutant)
    options.update(kw)
    return Image(**options)
def touch(i, page=0, **kw):
    opts = dict(namespace=7, page=page, event=4, cause=13, classify=classify)
    opts.update(kw)
    return i.touch(**opts)

i = image()
records[0] = Relocation(16, 0)  # Mutating the input cannot rewrite the protected plan.
symbols[0] = Symbol(None, weak=True)
check(touch(i) == 1 and list(i.memory) == [0])
check(i.memory[0] == Capability(source, 0x20010, 16, 0x20010))
check(touch(i) == 0)
check(touch(i, 1) == 1 and i.memory[4096] is None)
try:
    i.memory[0] = None
except TypeError:
    check(True)
else:
    check(False)
symbols[:] = [Symbol(source, 16, 16), Symbol(None, weak=True)]
records[:] = [Relocation(0, 0), Relocation(4096, 1)]
refused(lambda: image(version=2))
refused(lambda: image(metadata_writable=True))
for rs in ([Relocation(0, 0), Relocation(0, 0)], [Relocation(4096, 0), Relocation(0, 0)],
           [Relocation(-16, 0)], [Relocation(1, 0)], [Relocation(8192, 0)],
           [Relocation(0, 2)], [Relocation(0, 0, 1 << 63)]):
    refused(lambda: image(rs=rs))
refused(lambda: image(ss=[Symbol(None), symbols[1]]))
refused(lambda: image(ss=[Symbol(source, 4090, 16), symbols[1]]))
refused(lambda: image(ss=[Symbol(Grant(8, 2, 1, 0x20000, 4096, 'r'), 0, 16), symbols[1]]))
refused(lambda: image(ss=[Symbol(source, 0, 0), symbols[1]]))
large = Grant(7, 3, 1, 0x30000, 8192, 'r')
refused(lambda: image(ss=[Symbol(large, 1, 4096), symbols[1]]))
# R-60 exact small-object window: [0x1f800, 0x23800).
for cursor in (0x1f800, 0x237ff):
    j = image(rs=[Relocation(0, 0, cursor - 0x20010)])
    check(touch(j) == 1 and j.memory[0].cursor == cursor)
for cursor in (0x1f7ff, 0x23800):
    refused(lambda: image(rs=[Relocation(0, 0, cursor - 0x20010)]))
for cause in (12, 24, 25, 28, 29):
    refused(lambda: touch(image(), cause=cause))
refused(lambda: touch(image(), event=2))
refused(lambda: touch(image(), namespace=8))
refused(lambda: touch(image(), page=2))
live.remove(source)
j = image()
refused(lambda: touch(j))
check(not j.memory)
live.add(source)
# One valid relocation followed by a retired source must publish neither.
j = image(rs=[Relocation(0, 0), Relocation(16, 1)],
          ss=[symbols[0], Symbol(large, 0, 16)])
refused(lambda: touch(j))
check(not j.memory)
live.remove(destination)
refused(lambda: touch(i))  # An initialized page still requires live authority.
live.add(destination)
print(f'image-init-double checks={checks} mutant={a.mutant or "none"} peer={bool(a.fault_peer)} PASS')
