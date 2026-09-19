import pathlib
import struct
import subprocess
import sys

ported, reference, directory = sys.argv[1:]
out = pathlib.Path(directory)
out.mkdir(parents=True, exist_ok=True)
magic = 0x315854434C4D4747
events = []


def add(op, c=0, b=0, n=0, t=0, arg=0):
    events.append((op, c, b, n, t, arg))


for epoch in range(100):
    add(1, 0, 0, 65536, 0, 0)
    add(1, 1, 1, 65536, 0, 1)
    for size in (0, 1, 15, 16, 17, 31, 32, 255, 256, 1024):
        add(2, 0, 0, size, epoch % 3, epoch % 256)
        add(2, 1, 1, size, epoch % 3, (epoch + 1) % 256)
    add(3, 0, 0)
    add(2, 0, 0, 256, 0, 71)
    add(4, 0, 0)
    add(4, 1, 1)
for ownership in (1, 0):
    add(1, 0, 0, 65536, 0, ownership)
    add(2, 0, 0, 64, 2, 91)
    add(4, 0, 0)
add(5)


def encode(es):
    return struct.pack("<16Q", magic, len(es), *([0] * 12), 32, 0) + b"".join(
        struct.pack("<6Q", *e) for e in es
    )


trace = out / "directed.bin"
trace.write_bytes(encode(events))
results = []
for name, exe in (("ported", ported), ("reference", reference)):
    report = out / f"{name}.bin"
    run = subprocess.run(
        [exe, str(trace), str(report)], check=True, capture_output=True
    )
    results.append(run.stdout)
assert results[0] == results[1], results
report = struct.unpack("<16Q", (out / "ported.bin").read_bytes())
assert report[4] == len(events) and report[6] == 2102 and report[7] == 100
assert report[8:11] == (101, 101, 200) and report[13] == 0
for name, raw in {
    "truncated": encode(events)[:-1],
    "missing-end": encode(events[:-1]),
    "bad-context": encode([(4, 0, 0, 0, 0, 0), (5, 0, 0, 0, 0, 0)]),
    "live-end": encode([(1, 0, 0, 4096, 0, 0), (5, 0, 0, 0, 0, 0)]),
    "overlap": encode(
        [(1, 0, 0, 4096, 0, 0), (1, 1, 0, 4096, 0, 0), (5, 2, 0, 0, 0, 0)]
    ),
    "exhausted": encode([(1, 0, 0, 16, 0, 0), (2, 0, 0, 32, 0, 0), (5, 1, 0, 0, 0, 0)]),
}.items():
    path = out / f"{name}.bin"
    path.write_bytes(raw)
    assert (
        subprocess.run(
            [ported, str(path), str(out / "rejected.bin")], capture_output=True
        ).returncode
        != 0
    ), name
print(f"PASS native/reference: {len(events)} events; malformed and exhaustion controls")
