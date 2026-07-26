# The matmult/coremark hang is INSIDE THE COMPUTE, not at domain entry

**Date:** 2026-07-26 · **Lane:** B · 3 board boots. Board powered off + unlocked.

**This retracts the central claim of the 19-31-06 note from earlier today.**

## What was claimed, and why it was not earned

The 19-31-06 note ended with:

> ⇒ The hang is not reachable from the compiler side. It is a domain-entry fault
> needing monitor/RTL work; the domain-boundary `fence.i` patch is the standing
> candidate.

Three hypotheses had been killed with evidence (`-Os` codegen, code size, instruction
mnemonics), and the conclusion "domain-entry fault" was drawn from their failure. That
is not a valid inference. Ruling out three compiler-side explanations establishes only
that it is *not those three* — it does not localize the fault to a layer.

The specific gap: **a domain that enters correctly and then spins forever inside the
compute is externally identical to one that never enters.** No END marker either way.
Entry was assumed, never observed. Two further discriminators were also refuted while
chasing it — global *count* (`beebs_prime` 3 PASS vs `matmult_int` 3 HANG; `rv8_primes`
1 PASS vs `coremark_matrix` 1 HANG) and `.bss` *size* (`rv8_primes` carries the largest
`.bss` at 12,512 B and passes, while every hanging rung is under 800 B), which together
also kill the dom_data-exhaustion / `BUILD_GP_CAPTABLE`-zeroing-glue story.

## The experiment

`LADDER_INSTR_MODE=7` (`ladder_perf_domain.h`). The obvious probe — store a marker and
read it back — cannot work here, because the controller only reports `res[]` after the
`cscall` returns, so a hang hides the marker too. Mode 7 instead makes the answer be
**"does it return at all"**: the entire entry path runs unchanged (glue, gp-captable
build, region-cap delivery, `cscall`/`csreturn`), and only the compute is placed behind
a branch that is never taken.

The gate is the `func` argument because it is genuinely opaque — nothing passes it and
`domain_main` is called from assembly glue — so the optimizer cannot fold the branch and
delete the call. Keeping the compute *linked* is the point: its globals retain their
gp-captable slots and the image stays close to the hanging build. Gating on `res` would
have been unsafe (after the preceding stores, `-O1` may assume `res != NULL` and drop
the call).

Verified in the artifacts, not just the source: `matmult_int -O1` keeps `.bss` identical
at 768 B with `.text` 952 vs 992 B, and the disassembly shows `lui a2, 0xdeadc` /
`bne a1, a2` branching over a still-present compute. QEMU parity leg first: `phase=57623`
(0xE117), `ran=53406` (0xD09E).

## Result

| rung | opt | normally | mode 7 | attempt |
|---|---|---|---|---|
| `beebs_recursion` (control, normally PASSES) | −O1 | returns | **returns** | 1 |
| `matmult_int` | −O1 | **hangs** | **returns** | 1 |
| `coremark_matrix` @32 KiB | −O0 | **hangs** | **returns** | 1 |

All three on the first attempt, no retries, each with `ran=53406` and `phase=57623`.
`retval=0`/`cycles=0` are by construction (the compute never runs), so the oracle gate
reports NO — expected and irrelevant. END vs no END was the only signal.

**Both rungs that never produce a result completed a full domain round-trip.** Domain
entry, the entry glue, the gp-captable build, region-capability delivery and
`cscall`/`csreturn` all work for these exact binaries.

⇒ **The hang happens during the compute.** The "domain-entry fault" framing is retracted.

This also specifically kills the stale-icache-at-entry story: mode 7 fetches from the
same freshly-placed code at `0x10000` and executes it fine.

## What this means for the standing plan

**The domain-boundary `fence.i` patch is aimed at the wrong layer.** It is an entry-side
fix for a fault that is not at entry, and it was recorded as "the single highest-value
next action". It should not be built into board firmware on this rationale.

## What is NOT established

The mechanism inside the compute. The leading hypothesis is that this is the **same**
fault as the known miscompute, landing on a **loop bound** instead of a checksum:
`matmult_int` miscomputes at −O0 and hangs at −O1, which is what one bug with two
different victims looks like. That is a hypothesis, not a result.

Note "in the compute" does **not** by itself mean "compiler bug" — the known miscompute
is a *silicon divergence* (QEMU computes correctly, silicon does not), whose root cause
remains UNKNOWN with five hypotheses already dead. What has moved is the **location**:
the load/store path during compute, not the domain switch.

One caveat on strength: the mode-7 image is near-identical to the hanging build, not
byte-identical (`.text` differs by 40 B). `.bss`, global count and captable layout are
identical, and the skipped region is exactly the compute, so the inference is strong —
but it rests on a near-identical image.

## Next diagnostic

Bisect *within* the compute rather than around it: shrink the iteration counts /
matrix dimensions until `matmult_int -O1` returns, and find the threshold. A hang that
disappears below some trip count points at a corrupted loop bound; one that persists at
a trivial size points elsewhere. That is cheap and needs no firmware change.
