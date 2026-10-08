# FFmpeg plain-temporal repros — lifetime defects on a DIRECT allocation

**The cell this corpus exists to fill.** Every temporal corpus in this tree sits on a *nested*
allocator — FFmpeg's `AVBufferPool`, Wireshark's wmem, memcached's slabs and `cache.c` — because
that is what the temporal hunts were aimed at. Recomputing the inventory's four cells from each
case's `nested` boolean therefore gave **temporal × plain allocator = 0 for all three target
programs**. That zero was a property of which corpora existed, not of the upstream software.

**What belongs here:** the freed object came straight from av_malloc / av_mallocz / av_calloc / av_malloc_array, freed with av_free / av_freep, with **no inner allocator
between it and the platform**. Its sibling is `../pool-repros`, and the difference is not which bound is
crossed but **who allocated the object**: that corpus's objects come from an AVBufferPool or an AVRefStructPool, which recycles them; the platform allocator only ever saw the pool's block. Here the object IS the platform allocation.

## The observable is ALIASING, not ordering

A read of freed memory is undefined but usually quiet, so a case that only recorded "the access
came after the free" would be asserting its own construction. Each case here instead frees the
object, takes a **fresh allocation of the same size** — which glibc's tcache satisfies from the
very chunk just freed — fills it with a marker, and then reads through the **stale** pointer and
finds the marker. The stale pointer now names a different live object.

That is deterministic, it is what makes a use-after-free exploitable rather than merely undefined,
and it is two-sided: under the upstream fix the stale pointer either is not retained or is not
followed, so no marker is seen.

**The plain and ASan builds measure different things, on purpose.** Under ASan the freed chunk goes
to quarantine, so the fresh allocation does not reuse it and the aliasing cannot happen — the buggy
arm aborts at the labelled probe instead, which is the `native-detect` reading. The aliasing is
observed in the plain build. Neither alone is the result, which is why `runners/run-native.sh`
reports both columns.

## Liveness

Recorded, never required. Most cases here are fix-reversals against the 9.0.1 pin, the convention
the rest of this tree already follows.

## Cases

| shape | cases |
|---|---|

Each row's `case.json` carries the upstream fix, the exact lifetime ender, the access that follows
it, and a `nested: false` with the reason — the field the inventory's cell counts are computed from.
