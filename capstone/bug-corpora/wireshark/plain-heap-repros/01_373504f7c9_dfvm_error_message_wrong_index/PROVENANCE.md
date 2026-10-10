# 373504f7c9 — an error path indexes the second array with the first array's index

## The defect

`mk_binary_internal` walks two value arrays with two independent indices. The call that does the
work indexes correctly. The error message, reached only when that call fails, indexes the **second**
array with the **first** array's index. The arrays have independent lengths, so whenever `fv1` is
longer than `fv2` and the operation fails at an `i` beyond `fv2`'s end, the read is past
`fv2->pdata`.

## Upstream defect

- **Fix:** `373504f7c9` ("dfvm: Fix an error message to avoid an out-of-bounds read"). One
  character: `i` becomes `j`. It lands with a `test/suite_dfilter/group_syntax.py` case.
- **CVE:** none assigned.
- **Live at the `v4.6.8` pin: YES.**

## The vulnerable code, quoted from upstream

`373504f7c9^:epan/dfilter/dfvm.c:1382-1389`:

```c
	for (size_t i = 0; i < fv1->len; i++) {
		for (size_t j = 0; j < fv2->len; j++) {
			result = func(fv1->pdata[i], fv2->pdata[j], &err_msg);
			if (result == NULL) {
				debug_op_error(fv1->pdata[i], fv2->pdata[i], "&", err_msg);
```

and the fix:

```diff
-				debug_op_error(fv1->pdata[i], fv2->pdata[i], "&", err_msg);
+				debug_op_error(fv1->pdata[i], fv2->pdata[j], "&", err_msg);
```

**Liveness: LIVE AT THE PIN**, read from the pinned source rather than from ancestry, and keyed on
the **full expression** so the probe cannot match the correct call three lines above:
`v4.6.8:epan/dfilter/dfvm.c` contains `debug_op_error(fv1->pdata[i], fv2->pdata[i]` **1 time** and
the fixed `…fv2->pdata[j]` **0 times**. Two-sided.

## Why this is the NOT-NESTED row

A `GPtrArray`'s `pdata` is **one direct `g_malloc`**, grown with `g_realloc`, and nothing
sub-allocates it. There is no wmem scope and no chunk between the consumer and the platform
allocator, so the only bound in existence is the `malloc`'s own and the read leaves it.

## Why the row is worth having beyond the crossing

**The defect is on the error path only, and the correct index is three lines above it in the same
statement block.** That is why it survived review, and why a reduction exercising the success path
would come back clean and teach nothing — so this case deliberately forces the failure.

**The crossed value is only formatted into a debug string.** There is no corruption, no wrong answer,
nothing downstream: `damage` is 0 on both arms. A test keyed to *consequences* cannot see this defect
at all; a bounds check sees the read. `ffmpeg/plain-heap-repros` case 1 records the same asymmetry
from the other direction — upstream saying there "the excess element was never actually used, but it
still triggers ASAN" — and the two rows exist to make that class visible rather than to count twice.

**Reachability note:** any display filter whose binary operator is applied to two multi-value fields
of different cardinality, where the operation fails — upstream's own added test is
`test/suite_dfilter/group_syntax.py`.

## What is real here, and what is reduced

**Real:** the allocator. The platform's `malloc`/`free`. GLib is not linked — `g_malloc` is `malloc`
plus abort-on-failure, which is all this uses of it, and `shared/corpus.h` says so for the corpus.
The arms differ by exactly the upstream fix: the index, `i` against `j`.

**Reduced:** the arrays are 4 and 2 entries. The asymmetry **is** the defect's premise, so the case
asserts both that the error path's index is past `fv2` and that the working call's index is not; a
reduction that lost it fails rather than reporting a verdict about nothing. `func` and
`debug_op_error` are reduced to the read each performs. Nothing about *which* index is used is
reduced, because that is the defect.

## What the run establishes, and what it does not

**Establishes:** the defect is real and reproduces from the upstream fix differential, two-sided with
the control arm first; **ASan sees it**; and **stock CheriBSD catches it** — `si_code` 1, the fault
inside the labelled probe's extent, measured 2026-10-07 with both platform controls firing.

**Does NOT measure** the Capstone or PoisonCap arms. Those are declared predictions; PoisonCap is
unavailable on this host.

**N = 1 per cell.**
