#!/usr/bin/env python3
"""Negative test for gp-initdesc-blocks.py. No board, no QEMU, no build, and no forged ELF either.

It drives the block walker over bytes it writes itself, because that walker is the whole of the
gate: the rest is readelf. Forging an ELF well enough for readelf was tried first and tested the
forgery rather than the tool.

A gate that has only ever accepted is unproven, and this one guards a fault that every other gate
in the build passes, so it is what stands between the next port and half a day.
"""
import importlib.util
import pathlib
import struct
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gate", HERE / "gp-initdesc-blocks.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def blob_of(counts):
    """The section a linker would produce for translation units with these many globals each."""
    # The header is THIRTY-TWO bytes, of which the tool reads the first sixteen. Writing sixteen
    # here is the mistake this comment exists to prevent: every block after the first then lands
    # at the wrong offset and the walker reports nonsense that looks like a finding.
    return b"".join(struct.pack("<QQ", 1, c) + b"\0" * 16 + b"\0" * (24 * c) for c in counts)


def run(raw):
    """What the gate says about a section holding exactly these bytes."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(raw)
        path = f.name
    gate.section = lambda p, n: (0x20000, 0, len(raw))
    found = gate.blocks(path)
    pathlib.Path(path).unlink()
    return found


CASES = [
    ([7],        1, "one block is what the ABI wants"),
    ([0],        1, "one block with no globals at all"),
    ([4, 9],     2, "two blocks, which is the defect"),
    ([4, 2, 9],  3, "three blocks, the nginx image"),
    ([0, 5],     2, "a first block of zero still counts as two"),
]

fail = 0
for counts, want, label in CASES:
    found = run(blob_of(counts))
    got = None if found is None else len(found)
    if got == want and [c for _, c in found] == counts:
        print(f"  ok    {label:<44} {got} block(s)")
    else:
        print(f"  FAIL  {label:<44} got {found}, wanted {want} blocks of {counts}")
        fail = 1

# A header claiming more globals than the bytes can hold is not a block, and must not be read as
# one: inventing a block here would turn a truncated section into a pass.
raw = bytearray(blob_of([4]))
struct.pack_into("<Q", raw, 8, 9999)
if run(bytes(raw)) is None:
    print(f"  ok    {'an impossible count reads as unreadable':<44} None")
else:
    print(f"  FAIL  {'an impossible count reads as unreadable':<44} got a block list")
    fail = 1

# Too short to hold even a header: an empty list, which main() must turn into "broken" rather than
# an IndexError. That was a real bug in the first version of the gate and this case is why.
if run(b"\0" * 8) == []:
    print(f"  ok    {'a section too short for a header':<44} []")
else:
    print(f"  FAIL  {'a section too short for a header':<44} got something else")
    fail = 1
if gate.main([]) == gate.BROKEN:
    print(f"  ok    {'no arguments is refused, not a pass':<44} {gate.BROKEN}")
else:
    print(f"  FAIL  {'no arguments is refused, not a pass':<44}")
    fail = 1

print("gp-initdesc-blocks-selftest: " + ("every case behaved" if not fail else "FAILED"))
sys.exit(fail)
