# d133b4a231 — the backward kernel scan is inclusive of a bound the forward scan is exclusive of

## The defect

B is clamped to size + a and range is -a, so b + range == size; the forward loop is `n < b` and the backward one is `n >= a` starting at n = b, which reads tkernel[size].

## Upstream defect

- **Fix:** `d133b4a231`.
- **CVE:** none assigned.
- **Live at the `n9.0.1` pin: NO — the fix is already in at the pin, so this is a **fix-reversal**.**

## The vulnerable code, quoted from upstream

The fix's own diff, `libavfilter/avf_showcwt.c`:

```diff
-        for (int n = b; n >= a; n--) {
+        for (int n = b - 1; n >= a; n--) {
             if (tkernel[n+range] != 0.f) {
```

The allocation and the bounds, `n9.0.1:libavfilter/avf_showcwt.c:725` and `:736-738`:

```c
    tkernel = av_malloc_array(size, sizeof(*tkernel));
...
        const int a = FFMAX(frequency-12.f*sqrtf(1.f/deviation)-0.5f, -size);
        const int b = FFMIN(frequency+12.f*sqrtf(1.f/deviation)-0.5f, size+a);
        const int range = -a;
```

and the asymmetry, the forward pass being exclusive where the backward one is not:

```c
        for (int n = a; n < b; n++)        /* :740 -- writes, EXCLUSIVE of b  */
            tkernel[n+range] = ff;
        ...
        for (int n = b; n >= a; n--)       /* :755 -- reads,  INCLUSIVE of b  */
```

**Liveness: a FIX-REVERSAL**, read from the pinned source rather than from ancestry — the latter has
called backported fixes live before. Two-sided, so the probe is known to fire: `n9.0.1:libavfilter/avf_showcwt.c`
contains the fixed form (**1 occurrence**) and the pre-fix form (**0 occurrences**).

Liveness is **recorded, never required** — the convention is at
`../../memcached/allocator-repros/README.md:132-135`, and requiring it is what kept FFmpeg's spatial
count at four. All four rows in this corpus are fix-reversals.

**Why the index is exactly one past.** `b` is clamped to `size + a` and `range` is `-a`,
so `b + range` is at most `size + a - a` = `size`. The upstream report says it was "Reproduced with a
small output (e.g. size=2x2) under ASan", and a small output is what drives `b` onto that clamp.

**Reachability note:** the `showcwt` filter at a small output size.

## What is real here, and what is reduced

**Real:** the allocator. The platform's own `calloc`/`free`, for the reason `shared/corpus.h` states
at length — the buffer-pool port's `av_malloc` is an un-narrowed arena carve that rounds every request
to 64 bytes, so routing these cases through it would leave no per-allocation bound to cross and would
absorb every small crossing. The arms differ by exactly the upstream fix.

**Reduced:** the frequency/deviation arithmetic that produces `a` and `b` is replaced by
the clamped values it yields, and the case **asserts** `b + range == size` so a reduction that missed
the clamp would fail rather than measure something else. The kernel's float contents are reduced to
1.0f, since the scan only tests non-zero.

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
