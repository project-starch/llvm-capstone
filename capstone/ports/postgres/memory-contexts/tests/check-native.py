#!/usr/bin/env python3
"""Check the generated workload and prove its backing-count oracle rejects drift."""

from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary, fixture = map(Path, sys.argv[1:])
subprocess.run([binary, fixture], check=True)
raw = bytearray(fixture.read_bytes())
# A11_BLOCKS records have op=10, then four identities and three counters.
head = struct.Struct("<8sIIQIIQII32s")
record = struct.Struct("<IIIIQQQ")
with tempfile.TemporaryDirectory(dir=fixture.parent) as temporary:
    # Reject both a changed block count and an independently changed realloc
    # count: checking only malloc/free would miss the second mutation.
    for name, field in (("blocks", 4), ("realloc", 6)):
        data = bytearray(raw)
        for offset in range(head.size, len(data), record.size):
            values = list(record.unpack_from(data, offset))
            if values[0] == 10:
                values[field] += 1
                record.pack_into(data, offset, *values)
                break
        else:
            raise SystemExit("fixture has no backing-count oracle")
        path = Path(temporary) / (name + ".a11")
        path.write_bytes(data)
        run = subprocess.run([binary, path], capture_output=True, text=True)
        if run.returncode != 1 or "same thing" not in run.stderr:
            raise SystemExit(f"native oracle failed its {name} negative control: {run}")
print("Native replay and backing-count negative controls passed")
