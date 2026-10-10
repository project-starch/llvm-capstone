# ac59fc542f — uses a raw high-bit-depth sample as a histogram index, leaving the carved sub-slice

> **Moved 2026-10-10** from `../../subobject-repros/09_ac59fc542f_thumbnail_carved_histogram_slice`, where it was
> filed NOT NESTED as an 'inside one allocation' row. The region it leaves is a carve -- FFmpeg's arithmetic on one
> av_calloc -- and this corpus is where a carve is the inner allocator, so it is NESTED here, by the one rule
> `tools/catch-tables.py` states for carves. Its readings in the old corpus stay in that corpus's `results/`; this
> case is re-measured in this corpus's harness (the platform's calloc, `ffc_carve()`).

## The defect

`get_hist16`'s tail loop — the one reached when the width is not a multiple of 4, i.e. the
"odd sized HBD inputs" of the subject line — indexes the histogram with the raw 16-bit sample
`p16[x]`. The vectorised loop above it shifts and masks correctly; the tail does not. The histogram
has 256 entries.

## Upstream defect

- **Fix:** `ac59fc542f`. It shifts and masks the tail's index exactly as the vectorised body does: `hist[(uint8_t) (p16[x] >> shift)]++`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavfilter/vf_thumbnail.c`:

```diff
         /* handle tail */
         for (int x = width4; x < width; x++)
-            hist[p16[x]]++;
+            hist[(uint8_t) (p16[x] >> shift)]++;
```

The carve, twice over, `n9.0.1:libavfilter/vf_thumbnail.c:38`, `:206` and `:259`:

```c
#define HIST_SIZE (3*256)
...
    int *hist = s->thread_histogram + HIST_SIZE * jobnr;        /* :206 */
...
            int *hhist = hist + 256 * plane;                    /* :259 */
            if (s->bitdepth > 8) {
                get_hist16(hhist, p, linesize, planewidth, slice_end - slice_start,
                           s->bitdepth - 8);                    /* :262 */
```

**Liveness: a FIX-REVERSAL.** The fix is already in at the pin, read from the pinned
source rather than from ancestry — `git merge-base --is-ancestor` has called backported fixes live
before, so it is not used here. Two-sided, so the probe is known to fire:

- `n9.0.1:libavfilter/vf_thumbnail.c` contains the fixed `hist[(uint8_t) (p16[x] >> shift)]++` — **1 occurrence**
- the same file contains the pre-fix `hist[p16[x]]++` — **0 occurrences**

The case reconstructs the **pre-fix consumer shape against the shipped allocator**, which is this
tree's convention and not a weaker kind of case: it is stated at
`../../memcached/allocator-repros/README.md:132-135`, and most cases in this tree are fix-reversals.
Liveness is **recorded, never required**; requiring it is what left this cell nearly empty, and that
inference was retracted on `dev`.

**Why this row is in this corpus and not a sibling.** Its inner bound is a **carved slice**,
not a declared struct member — and two levels of it: one `av_calloc` holds `nb_threads` slices of
768 ints, and each slice holds three sub-slices of 256. Both boundaries are products of a
multiplication that **nothing declares**, which is exactly why no allocator can know them. The
corpus's boundary, "inside one allocation", covers it; there is no member to name.

**The magnitude is data-controlled, and the case says what it measures and what it does not.** A
16-bit sample reaches index 65535 — 65279 entries, 261 120 bytes, past the sub-slice — which would
leave the whole allocation and so would be measuring a *different* boundary. The case therefore uses
a **10-bit** sample, whose index 1023 is 768 entries past the sub-slice and still inside the
allocation, and **asserts** that containment. The larger magnitude is reported as a number, not
claimed as this row's reading.

**Reachability note:** the `thumbnail` filter on any high-bit-depth input whose plane width
is not a multiple of 4.

## What is real here, and what is reduced

**Real:** the allocator. `av_refstruct_allocz` over the pinned `libavutil/refstruct.c`, where
`refstruct.c:109` is `av_malloc(size + REFCOUNT_OFFSET)` — header and payload in **one**
allocation. So the fact the case turns on, that the crossed region shares an allocation with its
neighbour, is a property of FFmpeg's allocator and this struct's layout, not of the driver.

The case **asserts** the layout rather than assuming it, so a compiler or platform that laid the
members out differently would fail the case rather than quietly measure something else.

**Reduced:** `HIST_SIZE`, the 256-entry per-plane stride and the carve arithmetic are **not**
reduced. `get_hist16`'s vectorised body is left out because it is not the path that crosses — the
tail is, which is why the defect needs an odd width. The sample is 10-bit deliberately, as above.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, measured
two-sided with the control (fixed) arm run first.

**Does not establish** reachability upstream, anything about silicon, or the Capstone, PoisonCap,
CheriBSD and ASan readings. Those arms are **declared predictions** in `case.json` and are *not*
measured for this case. Measuring the Capstone arm needs a probe case in
`ports/ffmpeg/buffer-pool/security-tests` — the seam cases 0-2 use — and measuring ASan needs
`results/20261005-native-subobject/asan-probe.c` extended, with its positive control still firing.

**An arena caveat that bears on the Capstone and CheriBSD predictions.** This corpus's driver hands
the port's `av_malloc` **one** arena (`shared/driver.c` calls `ff2_memory_init`), and
`ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26` carves it by bumping a cursor,
returning a raw interior pointer and rounding every request up to 64 bytes;
`__builtin_capstone_cap_shrink` appears only in `src/capstone-domain/payload-capabilities.c:57`, on
pool *payload* blocks. So on this harness there is **no per-allocation bound on the struct at all**,
and a completion would be weaker evidence than "the bound is the whole allocation". The sub-object
claim does not rest on that — the crossing is interior by construction, which the case asserts on
the offsets — but the arm must not be read as a measured per-allocation bound.

**N = 1 per cell.**
