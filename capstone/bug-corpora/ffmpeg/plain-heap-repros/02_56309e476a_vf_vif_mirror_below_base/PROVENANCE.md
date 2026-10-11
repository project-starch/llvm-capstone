# 56309e476a — the hand-rolled filter mirror reflects a tap index of 2*w to -1 and never re-checks it

## The defect

`jj = jj < 0 ? -jj : (jj >= w ? 2*w - jj - 1 : jj)` folds once; a tap index of 2*w reflects to exactly -1, and the negative branch has already been passed over.

## Upstream defect

- **Fix:** `56309e476a`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin, so this is a **fix-reversal**.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavfilter/vf_vif.c` (the same change for both axes):

```diff
-                    ii = ii < 0 ? -ii : (ii >= h ? 2 * h - ii - 1 : ii);
+                    ii = avpriv_mirror(ii, h - 1);
...
-                    jj = jj < 0 ? -jj : (jj >= w ? 2 * w - jj - 1 : jj);
+                    jj = avpriv_mirror(jj, w - 1);
```

the tap index it is applied to, `n9.0.1:libavfilter/vf_vif.c:268`:

```c
                    int jj = j - filt_w / 2 + filt_j;
```

and the allocation, `n9.0.1:libavfilter/vf_vif.c:513-518`:

```c
    if (!(s->temp = av_calloc(s->nb_threads, sizeof(s->temp[0]))))
        return AVERROR(ENOMEM);

    for (int i = 0; i < s->nb_threads; i++) {
        if (!(s->temp[i] = av_calloc(s->width, sizeof(float))))
```

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry — the latter has
called backported fixes live before. Two-sided, so the probe is known to fire: `n9.0.1:libavfilter/vf_vif.c`
contains the fixed form (**1 occurrence**) and the pre-fix form (**0 occurrences**).

Liveness is **recorded, never required** — the convention is at
`../../memcached/allocator-repros/README.md:132-135`, and requiring it is what kept FFmpeg's spatial
count at four. All four rows in this corpus are fix-reversals.

**Why the mirror fails, precisely.** The reflection `2*w - jj - 1` maps `[w, 2*w-1]` into
`[0, w-1]` correctly. At `jj == 2*w` it yields `-1`, and at larger `jj` it yields more negative values
still. The `jj < 0` branch of the ternary was already evaluated and not taken, so nothing re-checks the
result — the expression folds **once**, and one fold is not enough when the tap reaches twice the row
length. `jj = j - filt_w/2 + filt_j` reaches `w - 1 + filt_w/2`, so `jj >= 2*w` needs
`filt_w >= 2*w + 2`: a filter wide relative to the row, which is the "small dimensions" of the fix's
subject line. The fix's `avpriv_mirror` folds repeatedly.

**Reachability note:** the `vif` quality-metric filter on small frames, where the filter width exceeds twice the plane dimension.

## What is real here, and what is reduced

**Real:** the allocator. The platform's own `calloc`/`free`, for the reason `shared/corpus.h` states
at length — the buffer-pool port's `av_malloc` is an un-narrowed arena carve that rounds every request
to 64 bytes, so routing these cases through it would leave no per-allocation bound to cross and would
absorb every small crossing. The arms differ by exactly the upstream fix.

**Reduced:** `w = 4` and `filt_w = 17`, the smallest pair for which a tap reaches exactly
`2*w`; the case **asserts** `jj_raw == 2*w` so a reduction that missed the geometry would fail rather
than measure something else. The filter's coefficients and the accumulation are left out — they cannot
move the address. `avpriv_mirror` is reproduced as a repeated fold, which is its contract.

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
