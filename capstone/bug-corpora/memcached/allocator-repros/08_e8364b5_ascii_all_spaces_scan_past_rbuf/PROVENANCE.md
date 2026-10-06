# e8364b5 — a leading-space skip with no end, over a cache object

## The defect

`try_read_command_ascii`'s large-multiget branch skips leading whitespace before deciding whether the
buffered bytes are a `get`. The skip has **no end condition**. The branch's own premise is that there
is no `'\n'` in the first few kilobytes, so a buffer of nothing but spaces walks the cursor past the
read buffer — a `cache.c` object — and on into the next object of the same cache.

## Upstream defect

- **Fix:** `e8364b5` ("ascii: fix potential overrun when given all spaces"). It derives an end from
  the bytes actually read and stops there. It lands with a `t/getset.t` subtest named
  *"oops all spaces"*.
- **CVE:** none assigned.
- **Live at the `1.6.45` pin: NO — the fix is already in, so this is a fix-reversal.**

## The vulnerable code, quoted from upstream

`e8364b5^:proto_text.c:366-374`:

```c
            /*
             * We didn't have a '\n' in the first few k. This _has_ to be a
             * large multiget, if not we should just nuke the connection.
             */
            char *ptr = c->rcurr;
            while (*ptr == ' ') { /* ignore leading whitespaces */
                ++ptr;
            }
```

and the fix:

```diff
             char *ptr = c->rcurr;
-            while (*ptr == ' ') { /* ignore leading whitespaces */
+            char *end = c->rcurr + c->rbytes-6;
+            while (*ptr == ' ' && ptr != end) { /* ignore leading whitespaces */
                 ++ptr;
             }
```

The buffer is a cache object, `e8364b5^:memcached.c:406-415`:

```c
static bool rbuf_alloc(conn *c) {
    if (c->rbuf == NULL) {
        c->rbuf = do_cache_alloc(c->thread->rbuf_cache);
        ...
        c->rsize = READ_BUFFER_SIZE;
```

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry. Two-sided:
`1.6.45:proto_text.c` contains the fixed `ptr != end` **1 time** and the pre-fix unbounded
`while (*ptr == ' ') {` **0 times**.

## Why this is a NESTED row, and a different inner layer from cases 5-7

The read buffer comes from **`cache.c`**, the object cache, which carves objects out of pages the
slab allocator owns. Cases 5, 6 and 7 all cross a **slab chunk**. The two are different inner layers
over the same pages, and a port that narrows one does not thereby narrow the other — which is why
this row is worth having rather than being a fourth instance of the same boundary.

The crossing leaves the object and stays inside the page, which is what a page-granular bound cannot
see.

**Reachability note:** the ASCII protocol's large-multiget path, reached by sending more than a read
buffer's worth of bytes with no newline. Upstream's own added subtest sends exactly that.

## What is real here, and what is reduced

**Real:** the allocator. `cache_create`/`cache_alloc`/`cache_free` are upstream's own `cache.c`
through this corpus's seam, and `READ_BUFFER_SIZE` is 16384, unreduced. The two objects' **order is
checked, not assumed** — the free list promises no ascending addresses, and case 5 records that trap
costing it a refused control run.

**Reduced:** a step cap is added on the reduction's side so a runaway scan cannot hang the harness;
it is set well past the crossing so it cannot mask it. The successor object is filled with spaces
too, deliberately — that is what makes the scan **continue** past the end rather than stop at the
first byte, which is the difference between an overrun and a single stray read.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, two-sided with
the control arm first, and the case **asserts** both halves — the cursor past the object at the pin,
inside it under the fix — so a reduction whose arithmetic missed fails rather than reporting a
verdict about nothing.

**Does NOT measure** the Capstone, PoisonCap or CheriBSD arms for this case. Cases 0-7 were measured
under QEMU on 2026-10-05 and on stock CheriBSD on 2026-10-06; this one has had neither.

**N = 1 per cell.**
