#!/usr/bin/env python3
"""Exercise all size classes, pool reassignment, raw fallback and arena release."""

from pathlib import Path
import struct
import subprocess
import sys

binary, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
magic = 0x31594C50524D5950
header = struct.Struct("<12Q")
event = struct.Struct("<4Q")
events = []
for i in range(12000):
    events.append((1, i, 512, i % 256))
for i in range(12000):
    events.append((4, i, 0, 0))
for round in range(4):
    sizes = [0, 1, 15, 16, 17, 31, 32, 33, 511, 512, 513, 4096] + list(
        range(16, 513, 16)
    )
    for i, size in enumerate(sizes):
        events.append((2, i, size, (i + round) % 256))
    for i in range(len(sizes)):
        for size in (500, 490, 33, 1024, 0, 17):
            events.append((3, i, size, i % 256))
        events.append((4, i, 0, 0))
events.append((5, 0, 0, 0))


def encode(items):
    return header.pack(magic, len(items), *([0] * 10)) + b"".join(
        event.pack(*e) for e in items
    )


trace = output / "exercise.bin"
trace.write_bytes(encode(events))
adapted = subprocess.run(
    [binary, trace, output / "report.bin"], check=True, capture_output=True
)
reference = subprocess.run(
    [sys.argv[3], trace, output / "reference.bin"], check=True, capture_output=True
)
assert adapted.stdout == reference.stdout, (adapted.stdout, reference.stdout)
assert (output / "report.bin").read_bytes() == (output / "reference.bin").read_bytes()
report = header.unpack((output / "report.bin").read_bytes())
assert report[0] == magic and report[3] == 0 and report[4] == len(events)
assert report[5] == report[6] and report[8] > 1 and report[9] > 0
assert report[5] == sum(e[0] in (1, 2) for e in events)
assert report[6] == sum(e[0] == 4 for e in events)
assert report[7] == sum(e[0] == 3 for e in events)
checksum = 0
for op, identity, size, value in events[:-1]:
    checksum = ((checksum * 33) ^ (op + 7 * identity + 13 * size + value)) & (
        (1 << 64) - 1
    )
assert report[11] == checksum
for name, data in {
    "truncated": encode(events)[:-1],
    "missing-end": encode(events[:-1]),
    "double-free": encode([(1, 0, 16, 1), (4, 0, 0, 0), (4, 0, 0, 0), (5, 0, 0, 0)]),
    "live-at-end": encode([(1, 0, 16, 1), (5, 0, 0, 0)]),
    "invalid-id": encode([(1, 65536, 16, 1), (5, 0, 0, 0)]),
}.items():
    path = output / f"{name}.bin"
    path.write_bytes(data)
    assert (
        subprocess.run(
            [binary, path, output / "bad-report.bin"], capture_output=True
        ).returncode
        != 0
    ), name
