#!/usr/bin/env python3
"""A small trace, generated, so the port has a gate that needs no recording.

    make-fixture-trace.py <out.a11> [--blocks taken,given,grown,peak]

The measurements replay a recording of a real backend, and that file is large
and is not in this repository. A gate is a different thing: it has to run
without one, be the same every time, and fail when the manager stops behaving
the way it did. So this writes a trace of its own, in the same format, shaped
like what a backend does rather than like a random walk.

  one context that lives for the whole run and never frees, as a cache does
  a query loop that creates a context, allocates a mix of sizes, frees some,
    creates a child for the executor, deletes the child, resets its own
  one allocation above the chunk limit, so a block of its own is taken
  one realloc across the limit, so the path that copies is reached
  one context created and deleted without ever allocating, which is a third
    of what the real recording does
  a context that climbs the block ladder and is reset, which pins the
    doubling: the first block is initBlockSize and each one after it doubles

The oracle is what makes it a gate. The host driver compares what the manager
asked of the level below against the A11_BLOCKS record, so the numbers baked in
here are the ones a known-good build produced: 72 blocks taken, 68 given back,
one grown, twelve held at once. A change in the manager's block behaviour fails
the run rather than passing quietly. Move them with --blocks, from the driver's
own output, and say in the commit why they moved.

The file is not in the repository. The gate generates it, so there is one
source of truth for the workload and it is this script.
"""
import pathlib, struct, sys

HEAD = struct.Struct("<8sIIQIIQII32s")
REC = struct.Struct("<IIIIQQQ")
ALLOC, FREE, REALLOC, RESET, DELETE, CREATE_ASET = 1, 2, 3, 4, 5, 6
BLOCKS, END = 10, 0
MAGIC, VERSION, ENDIAN = b"A11TRACE", 1, 0x0102030405060708

# aset.c's default parameters, ALLOCSET_DEFAULT_SIZES
MIN, INIT, MAX = 0, 8 * 1024, 8 * 1024 * 1024
SMALL = (16, 24, 48, 64, 120, 200, 512, 1000)
CHUNK_LIMIT = 8 * 1024            # ALLOC_CHUNK_LIMIT at the shipped constants


def namehash(s):
    h = 1469598103934665603
    for ch in s.encode():
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return ((h ^ (h >> 32)) & 0xFFFFFFFF) | 1


class Trace:
    def __init__(self):
        self.recs = []
        self.nctx = 0
        self.nptr = 0
        self.live = {}            # ptr id -> ctx id, so nothing is freed twice
        self.of_ctx = {}          # ctx id -> set of its live ptr ids

    def ctx(self, name, parent=0):
        self.nctx += 1
        i = self.nctx
        self.recs.append((CREATE_ASET, i, namehash(name), parent, MIN, INIT, MAX))
        self.of_ctx[i] = set()
        return i

    def alloc(self, c, n):
        self.nptr += 1
        i = self.nptr
        self.recs.append((ALLOC, c, i, 0, n, 0, 0))
        self.live[i] = c
        self.of_ctx[c].add(i)
        return i

    def free(self, p):
        c = self.live.pop(p)
        self.of_ctx[c].discard(p)
        self.recs.append((FREE, c, p, 0, 0, 0, 0))

    def realloc(self, p, n):
        c = self.live.pop(p)
        self.of_ctx[c].discard(p)
        self.nptr += 1
        i = self.nptr
        self.recs.append((REALLOC, c, p, i, n, 0, 0))
        self.live[i] = c
        self.of_ctx[c].add(i)
        return i

    def _sweep(self, c):
        for p in list(self.of_ctx[c]):
            self.live.pop(p, None)
        self.of_ctx[c] = set()

    def reset(self, c):
        self.recs.append((RESET, c, 0, 0, 0, 0, 0))
        self._sweep(c)

    def delete(self, c):
        self.recs.append((DELETE, c, 0, 0, 0, 0, 0))
        self._sweep(c)

    def write(self, path, blocks):
        taken, given, grown, peak = blocks
        body = list(self.recs)
        body.append((BLOCKS, 0, 0, peak, taken, given, grown))
        body.append((END, 0, 0, 0, len(body), self.nctx, self.nptr))
        head = HEAD.pack(MAGIC, VERSION, REC.size, ENDIAN, 1, 0, 0,
                         len(b"fixture"), 0, b"fixture")
        with open(path, "wb") as f:
            f.write(head)
            for r in body:
                f.write(REC.pack(*r))
        return len(body)


def build(t):
    """The workload. Deterministic on purpose: no randomness anywhere."""
    top = t.ctx("TopMemoryContext")
    cache = t.ctx("CacheMemoryContext", top)
    for k in range(40):                       # a cache fills and never frees
        t.alloc(cache, SMALL[k % len(SMALL)])

    msg = t.ctx("MessageContext", top)
    for q in range(60):                       # the query loop
        held = [t.alloc(msg, SMALL[(q + i) % len(SMALL)]) for i in range(12)]
        for p in held[::3]:                   # a third are freed by hand
            t.free(p)

        ex = t.ctx("ExecutorState", msg)
        for i in range(8):
            t.alloc(ex, SMALL[i % len(SMALL)])
        if q % 10 == 0:                       # a chunk of its own block
            t.alloc(ex, CHUNK_LIMIT * 2)
        t.delete(ex)

        empty = t.ctx("ExprContext", msg)     # created and deleted, never used
        t.delete(empty)

        if q == 30:                           # the path that has to copy
            p = t.alloc(msg, CHUNK_LIMIT + 64)
            t.realloc(p, CHUNK_LIMIT * 3)

        # A context that climbs the block ladder and is then reset, which is
        # what pins the manager's policy: the first block is initBlockSize and
        # each one after it doubles, so this takes six or seven blocks and
        # gives them all back at once.
        if q % 12 == 0:
            srt = t.ctx("TupleSort", msg)
            for i in range(420):
                t.alloc(srt, 1000 + (i % 7) * 8)
            t.reset(srt)                      # the ladder starts over
            for i in range(60):
                t.alloc(srt, 2000)
            t.delete(srt)

        t.reset(msg)

    t.delete(msg)
    t.delete(cache)
    t.delete(top)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    # The golden values: what the host build of a known-good manager asked of
    # the level below over this trace. The driver fails if they differ, so a
    # change in the manager's block behaviour fails the gate rather than
    # passing quietly. Move them only with --blocks and say why in the commit.
    blocks = (72, 68, 1, 12)
    if "--blocks" in sys.argv:
        blocks = tuple(int(x) for x in
                       sys.argv[sys.argv.index("--blocks") + 1].split(","))
    t = Trace()
    build(t)
    n = t.write(out, blocks)
    print(f"{out}: {n} records, {t.nctx} contexts, {t.nptr} objects, "
          f"oracle taken/given/grown/peak = {'/'.join(str(b) for b in blocks)}")


main()
