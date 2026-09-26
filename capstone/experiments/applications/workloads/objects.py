"""Parse structured records, retain a working set, then process bounded batches."""
import gc
import json
import os
import sys

n, batches, retained = map(int, sys.argv[1:4])
def phase(name):
    os.write(2, ("MEMPHASE " + name + "\n").encode())

def records(count):
    # Each request exercises the real JSON parser and object allocator.
    return [json.loads('{"id":%d,"tags":["alpha","beta"],"body":"%s"}'
                       % (i, "x" * 96)) for i in range(count)]

keep = records(retained)
phase("baseline")
checksum = 0
for epoch in range(batches):
    batch = records(n * (4 if epoch == batches // 2 else 1))
    checksum += sum(record["id"] for record in batch)
    phase("live-%d" % epoch)
    del batch
    gc.collect()
    phase("released-%d" % epoch)
assert sum(x["id"] for x in keep) == retained * (retained - 1) // 2
burst = 4 * n
expected = (batches - 1) * n * (n - 1) // 2 + burst * (burst - 1) // 2
assert checksum == expected
print("EXP-OK cpython %d" % checksum)
