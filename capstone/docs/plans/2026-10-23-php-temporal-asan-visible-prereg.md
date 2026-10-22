# Pre-registration — the temporal cases ASAN already sees

Filed BEFORE the run, per the root `CLAUDE.md` rule. The result commit cites this commit's hash.

## The population, and why it is interesting

`population(corpus, "temporal-asan-visible")`: the **29** `CWE-416` cases in the corpus's
default-build subset that ASAN detects **without** PHP's allocator being patched — the exact
complement, on the temporal axis, of the 31 this project has already measured.

They are not ASAN-blind for one reason: `efree` reaches `free()`. The block genuinely leaves
PHP's size-class cache, so ASAN's poisoning works. That single fact is what makes them a useful
control rather than more of the same, because it is also the fact our seam does **not** depend
on — the seam retires at `_efree`, above the cache, so it should not care either way.

**19 are out of the denominator**: native PHP dies on a signal, so there is no pair to read.
That is the CRASH-057 rule, and it is now applied on the corrected predicate (below).
**10 are scoreable**:

```
CRASH-062  CRASH-068  CRASH-070  CRASH-075  CRASH-078
CRASH-081  CRASH-083  CRASH-113  CRASH-117  CRASH-155
```

## A correction to the apparatus, made before the run and not after

The old test for "the oracle crashes, so there is no pair" was `pristine_outcome != "OK"`. That
conflates a crash with PHP ending itself: PHP exits **255** on its own fatal error — a parse
error, an uncaught exception, a failed include — after printing a deterministic diagnostic. Such
a case has a perfectly good oracle.

The old test never misfired on the ASAN-blind 31, where the only non-`OK` outcomes are two
`SIGNAL:11`. It was a condition satisfied by its input rather than by its logic. On **this**
population it misfires twice: `CRASH-062` and `CRASH-070` are `EXIT:255` from a PHP fatal error
and would have been discarded as "native PHP crashes on it".

The predicate is now `pristine_outcome.startswith("SIGNAL:")`, cross-checked per case against
the corpus's own `crashes_on_pristine_build`; a disagreement stops the run rather than being
resolved in our favour. The gate is negative-tested in the offline suite. The change is inert
for the ASAN-blind 31: that population is still 31, and `ran` still compares against exit 0 for
every case whose oracle is `OK`.

## Predictions, per arm

| | `stock` (0) | `block` (4) | `object` (6) | `exact` (7) |
|---|---|---|---|---|
| all 10 | MISSED | see below | **CAUGHT** | **CAUGHT** |

- **`stock` — MISSED for all 10.** Mode 0 takes one capability per block and never retires it.
  If any case reads CAUGHT here it is a case that faults on any build, as CRASH-022 and
  CRASH-102 do among the 31, and it says nothing about the protections.
- **`object` and `exact` — CAUGHT for all 10.** The seam retires at `_efree`, which these
  cases reach by construction. This is the prediction most likely to be wrong in an
  interesting way: a fault that lands outside the case's own ASAN site reads `unattributed`,
  which is what happened to CRASH-140 at `block` in the previous run.
- **`block` — the real question, and the one cell named in advance as most likely to move.**
  Mode 4 performs **no revocations** (measured: 0 / 0 / 8937 / 8937 in the previous run), so
  every catch it produces comes from a freed block being reused and overwritten with scalar
  data, which clears the tag. Because these blocks genuinely return to the heap and can be
  reused by *unrelated* allocations — rather than being parked for the same size class, as a
  cache-hidden block is — I predict **block catches a strictly higher fraction here than the
  6 of 21 it managed on the cache-hidden cases**. I am not predicting a number.

## Two cases carrying their own caveat

- **CRASH-070.** Its recorded pristine run prints `failed to open stream: soap12-test.inc`, so
  the oracle itself ran without the include the trigger needs. Whatever it reads, the reading
  is of a degraded oracle, and it is reported with that attached rather than quietly counted.
- **CRASH-062.** `EXIT:255`, PHP's own `Fatal error: Class declarations may not be nested`.
  The control arm must reproduce exit 255 **and** that text, which no run of this harness has
  ever been asked to do before.

## What would make this measurement void

The cross-arm gates are unchanged and all must pass: revoke instructions present in the image,
distinct mode lines per arm, strictly more revokes on the retiring arms, the bound probe
faulting as declared, and the benign workload completing with revocation on.

## Vehicle

The **identical image** as the ASAN-blind measurement — `php-real-seam.dom`, sha256
`b36d053690136b03f92dcb8ebb67bde2759168fd3167d17c6f3fdaeebebf8836` — on the merged QEMU at
submodule `1a6dd20732`, so the two tables are comparable without a second build to argue about.
