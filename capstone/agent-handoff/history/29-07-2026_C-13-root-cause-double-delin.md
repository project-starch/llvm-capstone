# C-13 root cause: `delin` is NOT idempotent on silicon, and the interp glue delins four times

**Date:** 2026-07-29
**Status:** A real defect, found and fixed. It explains the stage-1 vs stage-2 result
completely — but it did **NOT** close C-13: with the fix in place the **real** interp path
still failed on hardware (`beebs_primer1`, 2 attempts, 2026-07-29). Either the fix is
insufficient or there is a second independent failure, the prime suspect being the
descriptor read out of the monitor-copied blob — the one part of the design never checked
on silicon. Next step is stage 2 (fix, no descriptor read) x4 to tell those apart.
Read the "The bug" section as established, and the closing claims as not yet earned.
**Blocks:** SQLite on hardware (the descriptor-driven glue is the only path that can
express SQLite's globals, so a defect in it blocks the benchmark and nothing else).

## The bug

The RTL (`capstone-ariane/core/anvil_build/capstone_dyn_unit.anvil`, `func DELIN`) accepts
**`CAP_TYPE_LINEAR` only**:

```
} else if(rs1.metadata.cap_type != cap_type_t::CAP_TYPE_LINEAR){
    call raise_exception(data.trans_id, ex_code::UNEXPECTED_CAP_TYPE)
}
```

Our QEMU was patched to do the opposite (`capstone-qemu/target/riscv/op_helper.c:900`,
`helper_csdelin`):

```c
/* Already NONLIN — goal achieved, nothing to do. ...
 * treat delin as idempotent rather than faulting. */
if (rd_v->val.cap.type == CAP_TYPE_NONLIN) return;
```

So a double `delin` is **silent under emulation and fatal on silicon**. That divergence
hid this for the entire gp-captable bring-up.

`SPLIT` **preserves `cap_type`** — `modify_cap_start` / `modify_cap_end` /
`modify_cap_revnode` all copy the field through. So once `sp` is delin'd, *every*
capability split from it is already NONLIN.

The interp glue delin'd `sp` at entry (to derive the blob view), and then delin'd three
more capabilities derived from it:

| site | capability | after entry delin | on silicon |
|---|---|---|---|
| entry | `sp` | LIN → NONLIN | OK |
| `split(gp, sp, t1)` | `gp` | already NONLIN | **fault** (first to execute) |
| `split(t2, sp, t1)` | `t2` | already NONLIN | **fault** |
| tail | `sp` | already NONLIN | **fault** |

The generated glue never delins `sp` early, so all of its delins land on LINEAR
capabilities — which is exactly why `generated` passes on silicon and `interp` does not.

## Evidence

One fixed configuration, repeated — not a single sample:

- **stage 1** (entry delin omitted, `sp` stays LIN): **4/4 PASS** on hardware.
  `beebs_primer1..r4`, retval `582955588` == oracle, ~9722 cycles, instret 2708.
- **stage 2** (entry delin present): **3/3 FAIL** (4th cut off by a harness timeout),
  including at VA 0x10000 where an earlier one-off run had passed.

Stage 1 and stage 2 differ by exactly the entry block, so the attribution is sound.

## The fix

`delin` **exactly once**. A new `INTERP_SP_LINEAR` macro records whether the entry delin
ran; when it did, the three later delins are gated off as both redundant and fatal.

Verified by disassembly, not by preprocessor reasoning:

| build | delins per entry path |
|---|---|
| real (no `INTERP_FAKE_COUNT`) | **1** |
| stage 2 diagnostic | **1** (was 4) |
| stage 1 diagnostic | **3** — unchanged, so the 4/4 control still holds |

## Consequences beyond C-13

1. **R-2 ("delin in domain code wedges the board") is very likely this same bug**, not a
   property of `delin` in domain code at all. Re-test it.
2. **R-9 and every other interp-glue result is suspect** — they were single samples taken
   with a glue that faulted non-deterministically in practice. Re-measure.
3. **The earlier C-13 bisection that blamed `RUN_CAP_INIT`'s `jalr`, then `lla`/`auipc`,
   was invalid** and is fully retracted; both claims are void. The process error was
   bisecting before establishing that the failure was deterministic.

## Recommended follow-up (NOT done here)

**Make QEMU's `delin` strict so it matches the RTL.** The current leniency is a
QEMU-vs-hardware divergence that converts a hard silicon fault into a silent no-op — it
cost this project a multi-week blocker on its critical-path benchmark. Aligning it (or
putting the leniency behind an off-by-default flag) would turn this whole class into a
QEMU-visible failure. Deferred only because it may surface other latent double-delins and
the SQLite deadline is imminent; it should be done immediately after.

**General lesson:** the `.anvil` RTL sources are a readable, authoritative spec of what
this silicon actually does. Three competing hypotheses (`lcc` off-by-one, `lcc` cap-type
legality, byte-load `lb`) were killed by reading them in minutes, with no board time spent.
Consult them before theorising about a silicon-only failure.
