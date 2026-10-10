# bcbf3a5630 — the colour-space compaction loop guards on j while reading j + 1

## The defect

The shift loop's guard is `j < nb_formats` while its body reads `formats[j + 1]`, so the last iteration reads index nb_formats.

## Upstream defect

- **Fix:** `bcbf3a5630`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin, so this is a **fix-reversal**.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavfilter/vf_scale.c` (the same change twice, for input and output):

```diff
         for (int i = 0; i < formats->nb_formats; i++) {
             if (!sws_test_colorspace(formats->formats[i], 0)) {
-                for (int j = i--; j < formats->nb_formats; j++)
+                for (int j = i--; j + 1 < formats->nb_formats; j++)
                     formats->formats[j] = formats->formats[j + 1];
                 formats->nb_formats--;
             }
         }
```

and the allocation, `n9.0.1:libavfilter/formats.c:424-429` (`MAKE_FORMAT_LIST`):

```c
    formats = av_mallocz(sizeof(*formats));
    ...
        formats->field = av_malloc_array(count, sizeof(*formats->field));
```

Upstream's own commit message: *"Results in over-read of the array. Fortunately, the excess element
was never actually used, but it still triggers ASAN (and could in theory trigger a segfault)."*

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry — the latter has
called backported fixes live before. Two-sided, so the probe is known to fire: `n9.0.1:libavfilter/vf_scale.c`
contains the fixed form (**1 occurrence**) and the pre-fix form (**0 occurrences**).

Liveness is **recorded, never required** — the convention is at
`../../memcached/allocator-repros/README.md:132-135`, and requiring it is what kept FFmpeg's spatial
count at four. All four rows in this corpus are fix-reversals.

**Why "never actually used" is not a defence, and why this row is worth having.** The
value read from `formats[nb_formats]` is written to `formats[nb_formats - 1]`, which the following
`nb_formats--` immediately puts out of range — so nothing downstream observes it. A test keyed to
*consequences* therefore cannot see this defect at all. A bounds check sees the **read**, which is why
ASan trips and why the fix was taken. This corpus's `damage` field is 0 on both arms for exactly that
reason, and the crossing alone is the measurement.

**Reachability note:** `scale`'s format negotiation, whenever any colour space fails `sws_test_colorspace` — i.e. ordinarily.

## What is real here, and what is reduced

**Real:** the allocator. The platform's own `calloc`/`free`, for the reason `shared/corpus.h` states
at length — the buffer-pool port's `av_malloc` is an un-narrowed arena carve that rounds every request
to 64 bytes, so routing these cases through it would leave no per-allocation bound to cross and would
absorb every small crossing. The arms differ by exactly the upstream fix.

**Reduced:** `sws_test_colorspace` is replaced by "the first element fails", which is the
arrangement that makes the shift loop run its full length and so reach the last index; the formats are
plain ints. The out-of-bounds read is probed one byte wide at the first byte past the array, which is
the narrowest possible crossing of that bound.

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
