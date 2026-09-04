# The overhead is ABI, not hardware — and fault 1 reproduces outside gp_diag

**Date:** 2026-07-26 · **Lane:** B · One board run (2 rungs, power-cycle each).
Board powered off + unlocked. Adds `minstret` to `ladder_perf_domain.h` so the
capability side reports retired instructions, which the baseline half already did.

Three findings. The first is the result we wanted; the second and third were not
expected and matter more for what comes next.

## 1. RESULT: the 5.6% is the ABI retiring more instructions, not slower instructions

`rv8_primes` is the only rung whose retval is correct on **both** sides, so it is
the only one this can be computed on.

| | cycles | instret | CPI |
|---|---:|---:|---:|
| capability domain | 17,375,220 | 8,773,753 | **1.98** |
| baseline (warm) | 16,459,057 | 7,960,829 | **2.07** |
| ratio | **1.056** | **1.102** | — |

- The capability build retires **10.2% more instructions** but costs only **5.6%
  more cycles**.
- Its **CPI is LOWER** (1.98 vs 2.07).
- The 812,924 extra instructions cost 916,163 extra cycles — **1.13 cycles each**,
  against a whole-program average of 2.07.

**Reading:** capability enforcement is essentially free per instruction on this
CVA6. The measured overhead is an **ABI/codegen cost** — the gp cap-table sends
every global through `ldc rd, i*16(gp)` — and those extra instructions are simple
loads that pipeline better than the average instruction, which is why the cycle
overhead is *half* the instruction overhead.

That is a mechanism a reviewer can check, and it implies a tuned ABI (fewer
cap-table indirections) would shrink the number. **One benchmark only** — the
caveat that matters most here.

## 2. Adding four instructions flipped a PASSING rung to miscomputing

`beebs_prime` passed on 25-07 and again on 26-07 (baseline pairing, 1.032×). With
the instrumentation added — two `csrr minstret` and two stores to a high region
slot, none of them inside the compute — it now returns **1087631800** against
oracle **582955588**.

The kernel is unchanged and deterministic. So **a rung that passes is not stable
ground**: a small codegen perturbation outside the computation flips a *scalar*
rung into miscomputing. Previously the split looked clean and mechanical — "4/4
array-store rungs fail, 2/2 scalar rungs pass". That framing is now too strong;
the scalar rungs were passing *for this exact codegen*, not because scalar code is
immune.

Consequences:
- The earlier `beebs_prime` 1.032× **still stands** — it was measured on the
  un-instrumented binary that returned the correct oracle. But it is a more
  fragile datum than it appeared.
- The wrong retval is **not** simply fault 2: the delta from the oracle is
  504,676,212, which is not a multiple of 2^27.
- `rv8_primes` was NOT flipped by the same instrumentation.

## 3. Fault 1 reproduces in ordinary perf rungs — a far better repro than gp_diag3

The phase slot `res[65]` was written twice with the constants 1 then 2. Both rungs
read it back as a **16-byte-aligned DRAM address**:

| rung | `res[65]` | expected |
|---|---|---|
| beebs_prime | `0x819BFF10` | 2 |
| rv8_primes | `0x819BCE80` | 2 |
| *(gp_diag3, 26-07)* | *`0x8197FE90`* | *32768* |

All three are `0x819B…`/`0x8197…` — the same 256 KB neighbourhood. The controller
`memset`s the region to zero before sharing, so a slot that was never written would
read **0**, not an address. Something wrote a capability-shaped value there.

This is **fault 1 of the two-fault model, outside `gp_diag3`**, in two ordinary
benchmark domains. It was invisible until now only because perf rungs write
`res[0..2]` and nobody had ever read a higher slot back.

Two further facts, both new:

- **`rv8_primes` returned the CORRECT retval while its region held a corrupt word.**
  Fault 1 can occur silently, without affecting the result. So the passing rungs
  were never clean — they were merely clean *where anyone looked*.
- **The clobbered slot is an EARLY-written one; the LAST-written slot survives.**
  `res[65]` (written first) is corrupt, `res[64]` (written last) holds a plausible
  instret. That matches gp_diag3 exactly, where `res[39]` from the straight-line
  window was corrupt while `res[40..47]`, written later by probe F's loop, were
  intact. Anything stored *after* the stray capability store survives it.

### Why this is the better reproduction

`gp_diag3` is a 9-probe diagnostic whose fault vanishes when perturbed (v4 is
clean). What just reproduced is **two stores of a constant to a high region slot**
inside an otherwise ordinary domain. If that is the minimal form, it is far cheaper
to iterate on than gp_diag3, and it suggests a sharp next experiment: **write a
sequence of distinct constants to many slots at known points in the execution and
see which are clobbered** — that timestamps the stray store, which no gp_diag
version could do.

Note this supersedes the plan in
`26-07-2026_02-21-26_...load-side-claim-retracted.md` step 1 (print the region's
address from the controller). That is still worth doing, but this is a stronger
lead: the fault is now reproducible in a domain we can freely modify.

### On the phase marker itself

The marker was added to detect a *gated `minstret`* (phase 1 = faulted, 2 = read).
It returned neither. `minstret` clearly **is** readable — `instret` came back with
plausible values on both rungs — so the diagnostic answered its own question
correctly while the slot holding the answer was corrupted by something else. Worth
keeping in mind: a marker slot in the shared region is not a trustworthy channel
on this silicon.

## Status of the numbers

- **Trustworthy:** `rv8_primes` cycle+instret, both sides, same session ⇒ the
  decomposition in §1.
- **Still valid:** `beebs_prime` 1.032× from the un-instrumented binary.
- **Discard:** `beebs_prime`'s instrumented run (wrong retval ⇒ wrong execution).

Next: `minstret` is confirmed readable, so the instrumentation stays; but any rung
carrying it must be re-gated on the oracle, because adding it can flip a rung.
