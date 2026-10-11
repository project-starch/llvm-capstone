import pathlib
import struct
import subprocess
import sys

ported, reference, directory = sys.argv[1:]
out = pathlib.Path(directory)
out.mkdir(parents=True, exist_ok=True)
MAGIC = 0x31304D454D575357
NEW, ALLOC, FREE, REALLOC, FREE_ALL, GC, DESTROY, END = range(1, 9)
SIMPLE, BLOCK, STRICT, BLOCK_FAST = range(4)
events = []
counts = dict(news=0, allocs=0, frees=0, reallocs=0, free_alls=0, gcs=0, destroys=0)
next_object = 1


def add(op, pool=0, obj=0, size=0, kind=0, arg=0):
    events.append((op, pool, obj, size, kind, arg))


def alloc(pool, size, fill):
    global next_object
    obj = next_object
    next_object += 1
    add(ALLOC, pool, obj, size, 0, fill)
    counts["allocs"] += 1
    return obj


for epoch in range(20):
    for pool, kind in enumerate((SIMPLE, BLOCK, STRICT, BLOCK_FAST)):
        add(NEW, pool, 0, 0, kind)
        counts["news"] += 1
    for pool in range(4):
        objects = [
            (alloc(pool, size, (epoch + size) % 256), size)
            for size in (1, 15, 16, 17, 31, 32, 255, 256, 1024, 4096, 70000)
        ]
        add(REALLOC, pool, objects[0][0], 300, 0, 7)
        add(REALLOC, pool, objects[-1][0], 100, 0, 9)
        add(REALLOC, pool, objects[5][0], 33, 0, 11)
        counts["reallocs"] += 3
        add(FREE, pool, objects[3][0])
        counts["frees"] += 1
    # Jumbo objects exceed one block of their allocator: 8 MiB and 2 MiB.
    j1 = alloc(BLOCK, 9 << 20, 5)
    j2 = alloc(BLOCK_FAST, 3 << 20, 6)
    add(REALLOC, BLOCK, j1, 10 << 20, 0, 8)
    add(REALLOC, BLOCK_FAST, j2, 4 << 20, 0, 10)
    counts["reallocs"] += 2
    add(FREE, BLOCK, j1)
    counts["frees"] += 1
    add(GC, BLOCK)
    counts["gcs"] += 1
    for pool in range(4):
        add(FREE_ALL, pool)
        counts["free_alls"] += 1
    for pool in range(4):
        alloc(pool, 64, epoch % 256)
    add(GC, BLOCK)
    counts["gcs"] += 1
    for pool in range(4):
        add(DESTROY, pool)
        counts["destroys"] += 1
add(END)


def encode(es):
    return struct.pack("<16Q", MAGIC, len(es), *([0] * 14)) + b"".join(
        struct.pack("<6Q", *e) for e in es
    )


trace = out / "directed.bin"
trace.write_bytes(encode(events))
results = []
for name, exe in (("ported", ported), ("reference", reference)):
    report = out / f"{name}.bin"
    run = subprocess.run([exe, str(trace), str(report)], check=True, capture_output=True)
    results.append(run.stdout)
assert results[0] == results[1], results
report = struct.unpack("<16Q", (out / "ported.bin").read_bytes())
assert report[3] == 0 and report[4] == len(events), report
assert report[5:12] == tuple(
    counts[k] for k in ("news", "allocs", "frees", "reallocs", "free_alls", "gcs", "destroys")
), report
assert report[13] == 0 and report[14] > 0 and report[15] > 0, report
new = (NEW, 0, 0, 0, BLOCK, 0)
end = (END, 0, 0, 0, 0, 0)
for name, raw in {
    "truncated": encode(events)[:-1],
    "missing-end": encode(events[:-1]),
    "unknown-pool": encode([(FREE_ALL, 5, 0, 0, 0, 0), end]),
    "live-end": encode([new, end]),
    "double-new": encode([new, new, (END, 1, 0, 0, 0, 0)]),
    "bad-kind": encode([(NEW, 0, 0, 0, 4, 0), end]),
    "zero-size": encode([new, (ALLOC, 0, 1, 0, 0, 1), (END, 1, 0, 0, 0, 0)]),
    "live-object": encode(
        [new, (ALLOC, 0, 1, 8, 0, 1), (ALLOC, 0, 1, 8, 0, 1), (END, 1, 0, 0, 0, 0)]
    ),
    "unknown-object": encode([new, (FREE, 0, 7, 0, 0, 0), (END, 1, 0, 0, 0, 0)]),
    "payload-on-end": encode([new, (DESTROY, 0, 0, 0, 0, 0), (END, 0, 0, 1, 0, 0)]),
}.items():
    path = out / f"{name}.bin"
    path.write_bytes(raw)
    assert (
        subprocess.run([ported, str(path), str(out / "rejected.bin")], capture_output=True).returncode
        != 0
    ), name
print(f"PASS native/reference: {len(events)} events; malformed controls")
