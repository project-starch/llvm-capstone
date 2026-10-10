# 495b402f27 — the edge-emulation base is sized for one sub-buffer and then carved into four

## The defect

Four sub-buffers are spaced i*ffalign(p->width, 16) apart inside an allocation of stride*max_blocksize, while each is used for up to max_blocksize rows of stride bytes -- room for one, not four.

## Upstream defect

- **Fix:** `495b402f27`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin, so this is a **fix-reversal**.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavcodec/diracdec.c` — both halves, the allocation and the carve:

```diff
-    s->edge_emu_buffer_base = av_malloc_array(stride, MAX_BLOCKSIZE);
+    s->edge_emu_buffer_base = av_malloc_array(stride, 4 * MAX_BLOCKSIZE);
...
         for (i = 0; i < 4; i++)
-            s->edge_emu_buffer[i] = s->edge_emu_buffer_base + i*FFALIGN(p->width, 16);
+            s->edge_emu_buffer[i] = s->edge_emu_buffer_base + i*s->buffer_stride*MAX_BLOCKSIZE;
```

with `MAX_BLOCKSIZE` 32 at `n9.0.1:libavcodec/diracdec.c:56` and the members at `:221-226`:

```c
    uint8_t *edge_emu_buffer[4];
    uint8_t *edge_emu_buffer_base;
    ...
    int buffer_stride;
```

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry — the latter has
called backported fixes live before. Two-sided, so the probe is known to fire: `n9.0.1:libavcodec/diracdec.c`
contains the fixed form (**1 occurrence**) and the pre-fix form (**0 occurrences**).

Liveness is **recorded, never required** — the convention is at
`../../memcached/allocator-repros/README.md:132-135`, and requiring it is what kept FFmpeg's spatial
count at four. All four rows in this corpus are fix-reversals.

**Why there is no inner bound to cross.** The four sub-buffers are spaced
`FFALIGN(p->width, 16)` apart but each is used for up to `MAX_BLOCKSIZE` rows of `stride` bytes, so
pre-fix they **overlap one another**. "Sub-buffer 3's bound" is therefore not a well-defined region,
and the only bound in existence is the `av_malloc_array`'s own — which sub-buffer 3's last row leaves.
That is why this row is in the plain-heap corpus and is filed as **not nested**: nothing
sub-allocated anything. The fix has to change both the size and the spacing, because either alone
leaves the four not fitting.

**Reachability note:** the Dirac/VC-2 decoder's motion-compensation path on any frame needing edge emulation.

## What is real here, and what is reduced

**Real:** the allocator. The platform's own `calloc`/`free`, for the reason `shared/corpus.h` states
at length — the buffer-pool port's `av_malloc` is an un-narrowed arena carve that rounds every request
to 64 bytes, so routing these cases through it would leave no per-allocation bound to cross and would
absorb every small crossing. The arms differ by exactly the upstream fix.

**Reduced:** `width = stride = 16`, so `FFALIGN(width, 16)` is the stride and the
arithmetic is the clearest it can be; the case **asserts** that sub-buffer 3's last row is past the end
at the pin and inside it under the fix, so a reduction with the wrong geometry fails rather than
measuring something else. Only the FIRST crossing byte is written; `extent` records that the unreduced
path writes a whole row.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control arm first; and **ASan sees it**, also two-sided, which is the contrast this
corpus draws against the sub-object corpora.

**Does NOT measure the Capstone, PoisonCap or CheriBSD arms.** Those are declared predictions in
`case.json`, recorded before any run so a refutation stays visible. The CheriBSD prediction is
deliberately **conditional**: that platform's `malloc` bounds to the allocator's *usable* size, not
the request, which is why `memcached/plain-heap-repros/00` predicted a catch and was refuted. This
case's request size and crossing distance are recorded in the arm so the reading can be checked
against them rather than assumed.

**Does not establish** upstream reachability. The note above names the entry point believed to reach
the consumer; that is a belief, not a measurement.

**N = 1 per cell.**
