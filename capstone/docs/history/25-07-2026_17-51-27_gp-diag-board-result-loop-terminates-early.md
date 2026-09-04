# BOARD RESULT: a counted loop containing a CALL terminates after one iteration on silicon

> ## ⚠️ SUPERSEDED 25-07-2026 18:44 — the call-in-loop hypothesis is REFUTED.
>
> gp_diag **v2** ran a loop whose body calls a `noinline` function (`dbg9`) and it returned the
> correct 9. The hypothesis below is dead. v2 also showed **all 11 probes correct**, including
> the array-store-with-live-accumulator pattern (`dbg5`=28) that the original bug report was
> built on. The v1 OBSERVATION here (the loop ran exactly one iteration; retval ==
> FNV(H0,[0x5A5A]) exactly) stands and is still the key datum — only its explanation is
> withdrawn. Current leading hypothesis: iterated access through the SHARED-REGION capability.
> See `25-07-2026_18-44-49_gp-diag-v2-all-probes-correct-fault-is-shared-region-loop.md`.

**Date:** 2026-07-25 · **Lane:** B · **One board run** (gp_diag v1, one power-cycle).
Follows `25-07-2026_17-09-01_gp-captable-miscompute-shrink-theory-refuted.md`.

## The result

```
rung      retval      oracle      cycles   correct
gp_diag   542820029   875368783   1325     NO
gp_diag raw probes: dbg0=23130 dbg1=0 dbg2=0 dbg3=0 dbg4=0 dbg5=0 dbg6=0 dbg7=0 dbg8=0
```

- **`dbg0 = 23130 = 0x5A5A` — CORRECT.** Probe 0 (scalar global write→read through
  `ldc gp[i]`) works perfectly on silicon.
- **Every later slot is 0 — including `dbg8`, the canary**, which returns a literal
  constant and touches no global, no array, no call. A zero canary cannot be a
  data-corruption result; those slots were simply **never written**.
- **`retval = 542820029` is *exactly* `FNV(H0, [0x5A5A])`.** Verified with the inversion
  tooling. Not "close to", not one-word-off — the domain folded **exactly one value**.
- `cycles = 1325` corroborates: far too few for 9 probes.

## What this proves

The driving loop

```c
for (int p = 0; p < GPD_NPROBE; p++) {      /* 9 iterations */
    unsigned v = gpd_probe(p);              /* <-- a CALL */
    res[3 + p] = v;
    for (int b = 0; b < 4; b++) { ... }     /* inner 4-iteration fold */
}
```

**executed exactly ONE iteration on hardware, while being correct under QEMU.**

Crucially, the value that one iteration computed was **right**, and the **inner**
4-iteration byte-fold ran **all 4 iterations correctly** (that is what makes
`FNV(H0,[0x5A5A])` match exactly). So this is **not data corruption at all** — it is
**premature loop termination**, and it is selective: the inner loop survived, the outer
loop did not.

**This reframes the whole bug.** Every prior note described "wrong values" and hunted for
a corrupted datum. The mechanism is a control-flow one: loops exit early, so the values
they were supposed to accumulate are simply incomplete.

It also explains why the checksum inversion failed earlier: no *single-word* corruption of
the correct value list could match, because the folded values are themselves produced by
loops that ran short.

## The discriminator (hypothesis, NOT yet confirmed)

The outer and inner loops are both plain `-O0` counted loops over a stack variable, and
the disassembly of both is unremarkable (`lw` counter → `addiw` → `sw` → `blt`). The one
structural difference: **the outer loop's body contains a CALL; the inner loop's does
not.**

Working hypothesis: **a counted loop whose body contains a call terminates early on this
silicon.** Plausible mechanism — the loop counter lives on the stack at `-O0` and is
reloaded after the call; something about the call/return path leaves that reload wrong (the
loop then sees `p >= 9` and exits via `blt`).

**Do not treat this as established** — it rests on one observation. But it now survives the
obvious objection, and fits more of the rung data than anything before it.

### The `beebs_prime` counter-example is reconciled (it is insensitive, not contradictory)

`beebs_prime` PASSES on silicon even though `prime()`'s loop calls `divides()` in its body
at `-O0`. That looked fatal. It is not: **its output cannot distinguish the two cases.**
Both inputs (21649, 513239) are genuinely prime, so trial division never finds a divisor;
if the loop exits after one iteration it also finds no divisor and still returns "prime".
Checked numerically — correct result `0`, early-exit result `0`. The folded `px`/`py` come
from `swap()`, which has no loop. So `beebs_prime` is **blind to this bug by construction**
and is not evidence either way.

### How the six rungs line up

| rung | verdict | loop-with-call in the result path? |
|---|---|---|
| rv8_primes | PASS | **no** — the sieve's inner loops use macros (`RV8_ISCOMP`/`RV8_SETCOMP`), no calls. Consistent. |
| beebs_prime | PASS | yes, but **output insensitive** to early exit (above). No information. |
| matmult_int | FAIL | **yes** — the `mmC[i][j] = mm_cell(i,j)` loops call a `noinline` callee. Fits. |
| beebs_crc32 | FAIL | **yes** — `crc32pseudo`'s 1024-iteration loop calls `rand_beebs()`. Fits. |
| beebs_insertsort | FAIL | **no** — `is_sort`'s while-loops and the fold loop contain no calls. **Does not fit.** |
| beebs_recursion | FAIL | **no** — the 3-iteration fold loop contains no calls. **Does not fit.** |

So the call-in-loop hypothesis covers `matmult_int` and `beebs_crc32` and is consistent
with `rv8_primes`, but **two failures remain unexplained by it** — which is why v2 probes
several mechanisms rather than just this one. Note `beebs_insertsort` is the one rung using
an **initialized global** (`expected[]` → `.L__const.is_verify.expected`), which is exactly
probe P7; and `beebs_recursion`'s distinguishing feature is deep/mutual recursion, which is
P4/P5. **Expect more than one mechanism.**

Prefix-scan cross-check against the perf rungs was **negative**: no prefix of the correct
value list reproduces `beebs_recursion` (2095861164), `matmult_int` (1166210317), or
`beebs_insertsort` (255001740). That is expected if loops run short — the folded values are
computed by *earlier* loops, so they are wrong too, not merely truncated — but it means
those rungs are not yet explained by this finding either.

## gp_diag v2 (built, pending QEMU validation)

Two changes, both forced by the v1 result:

1. **The driver is now STRAIGHT-LINE.** v1 measured 9 probes through a loop — the exact
   construct that turned out to be broken — so 8 of 9 slots reported nothing about their
   own probe. v2 stores each slot with unrolled code, which cannot be truncated by the bug
   under investigation. The FNV fold is still a loop, but deliberately: it runs *after*
   every raw slot is stored, so if it truncates, only `res[0]` is wrong and `dbg0..dbg10`
   stay valid.
2. **Two new probes isolate the hypothesis directly** — identical loops differing only in
   whether the body calls:
   - `dbg8` P10: `for (i=0;i<9;i++) n++;` (no call) → expect 9
   - `dbg9` P11: `for (i=0;i<9;i++) n += gpd_one();` (call) → expect 9
   `gpd_one()` is `noinline` and returns a constant — no global, no array, no recursion.
   If P10 == 9 and P11 != 9, the hypothesis is confirmed and minimal.

Slot map is now dbg0..dbg10 (canary moved to dbg10); `LADDER_DBG_SLOTS` = 11.
New oracle: **3613869247**.

**Build constraint hit:** unrolling the 11-way FNV fold too pushed `.text` to 0x1317 and
the link failed — `link-gpfree.ld` hard-requires `.text` inside `[0, 0x1000)` (the
monitor's PCC window). Keeping the fold as a loop brings it to 0xcbc. Worth remembering:
the 4 KiB code window is a real budget, and unrolling burns it fast.

## Next

1. QEMU-validate gp_diag v2, then **one board run** — it should confirm or kill the
   call-in-loop hypothesis outright, and give clean per-probe values for the other 9.
2. If P11 fails and P10 passes: build the minimal reproducer (a loop calling a
   constant-returning `noinline` function) and take *that* to the board owner. It would be
   a far stronger artifact than anything we have had — tiny, `-O0`, QEMU-correct,
   silicon-wrong, no capability semantics involved at all.
3. Re-examine `beebs_prime` (PASSES, yet has a call in a loop) — it is the sharpest
   counter-example and must be reconciled before the story is trustworthy.
