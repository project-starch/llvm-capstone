# Two new silicon-correct rungs, one publishable row — and an −O-level procedure bug that nearly produced a false finding

**Date:** 2026-07-27
**Lane:** C (primary)
**Cost:** 13 board boots. Board powered off + unlocked after every run.
**Net:** silicon-correct rungs 3 → 5; publishable overhead rows 3 → 4.

---

## Headline

| rung | capability | baseline (−O1, warm) | **cycles** | **instr** | status |
|---|---|---|---:|---:|---|
| `beebs_bs` | 2,258 cyc / 875 instr | 1,912 cyc / 827 instr | **1.181×** | **1.058×** | ✅ **publishable** |
| `beebs_cnt` | 128,178 cyc / 76,429 instr | 187,313 cyc / 67,140 instr | 0.684× ⚠ | 1.138× | ⚠ **instr only** |
| `beebs_fibcall` | 2,022,152,645 (oracle 2,518,333,445) | — | — | — | ❌ miscompute |
| `beebs_fac` | hang, both attempts | — | — | — | ❌ hang |
| `beebs_duff` | hang, both attempts | — | — | — | ❌ hang |

`beebs_bs` is a clean fourth row: capability CPI rises 2.31 → 2.58, cycles up 18.1 %,
instructions up 5.8 % — the same *more instructions, ABI not enforcement* shape as the
sieve.

**`beebs_cnt` PASSES silicon correctness but its cycle ratio is NOT publishable.**
0.684× would mean pervasive capability safety makes code **32 % faster**, which is not a
result, it is an uncontrolled confound. Its instruction ratio (1.138×) is credible and
sits in family with the other rungs. See "The `cnt` cycle anomaly" below — this needs
resolving before `cnt` can be a row, and it may affect the existing rows too.

## The procedure bug — read this before running any sweep

`run_ladder_perf_fpga.py` **rebuilds every artifact by default** (the 25-07 anti-stale
fix) by shelling out to `build-ladder-fpga.sh` with the inherited environment. I
pre-built the domains with `LADDER_OPT=-O1` and then **omitted `LADDER_OPT` from the
sweep command itself**. The runner therefore rebuilt all five at its `-O0` default,
discarded my `-O1` binaries, and measured those — against baselines I had specified at
`-O1`.

Consequences, all of which were briefly believed and reported:

1. **All five rungs "failed"**, including `beebs_bs`, which had passed in an earlier
   session.
2. Because the three deliberately-chosen **cross-object** controls all failed, I
   concluded **"R-1's same-object clause is refuted"** and stated that the repro package
   going to the board owner was misleading and needed correcting.
3. Because the previously-passing rung now failed, I nearly wrote up **"an ordinary
   rebuild flips a passing rung"** as a hardened §5 finding — a claim that would have
   made the hardware look markedly flakier than the evidence supports.

**All three were wrong.** Re-run correctly at `-O1`, `beebs_cnt` — the sharpest
cross-object control there is — **passes**, exactly as R-1 predicts. R-1 stands as
documented and the repro package needs **no** correction.

### What caught it

Only the control. `beebs_bs` was in the sweep purely as a stability check, and its
failure is what made the whole sweep suspect rather than informative. Without it, three
`-O0` artifacts would have gone into the registry as evidence against R-1's scope, and
from there into a corrected bug report sent to the board owner.

Two attempts were needed to make the control work, and the first was itself invalid:
pointing `LADDER_FPGA_DIR` at the preserved binary **did not run that binary**, because
the runner rebuilt over it. `LADDER_REBUILD=0` is required to run a pre-built set. The
original artifact survived only by luck, at
`/tmp/capstone/capstone-runtime-qemu-share/beebs_bs.dom`.

### The −O0/−O1 difference is large and visible statically

| build | `.text` | `ldc gp[i]` | globals in cap table | board |
|---|---:|---:|---:|---|
| `beebs_bs` −O0 | 2,100 B | 4 | 2 (2 initialized) | ❌ 1677761900 |
| `beebs_bs` −O1 | 1,408 B | 2 | 1 | ✅ 887447230 |

`bs_compute` holds a function-local `static const int probes[18]` beside `bs_data[15]`.
At −O1 it stays a private constant; at −O0 it becomes a **delivered cap-table global**,
adding an entry, two more `ldc gp[i]`, and ~700 B of initialization-copy code. This is
the C-4 boundary moving under an optimization flag.

**−O1 reproduces the passing binary byte-for-byte** (`cmp` clean), which is how the
level was identified after the fact.

### Rules that follow

- **`LADDER_OPT` must be set on the RUNNER invocation, not just on a pre-build.** The
  runner rebuilds; a pre-built set is discarded silently.
- **Always keep a known-good rung in every sweep.** It is the only thing that
  distinguishes "informative failures" from "my harness is misconfigured".
- **`LADDER_REBUILD=0` is required to run a specific pre-built binary.** Pointing the
  artifact dir at it is not enough.
- Check the static shape (`.text`, `ldc-gp` count) against the known-good build before
  believing a flipped result.

## The `cnt` cycle anomaly — open, and it may not be `cnt`-specific

`beebs_cnt` retires **13.8 % more instructions** in the capability build (expected: that
is the gp-captable ABI) but takes **32 % fewer cycles**. Capability CPI is 1.68 against a
baseline CPI of 2.79.

The baseline runs as a Linux userspace process; the capability domain runs bare-metal
with a clean icache and no OS. For a rung with a 400 B working set and a tight
init/sum loop, the baseline may be paying for OS interference the domain never sees.
That is the same *class* of confound as the cold-vs-warm paging one already documented
(which produced "capabilities are 1.8× faster" for `beebs_prime`) — and the warm-baseline
rule was written to control exactly that, so it is not sufficient here.

`beebs_bs` does not show it (1.181×, coherent). Neither does the sieve. So it is not
universal, but **it is not established that the existing rows are free of it** —
`beebs_prime` at 1.032× would be the one to re-examine, since a confound of this
direction would *understate* capability overhead.

Do not quote a `cnt` cycle ratio. Its instruction ratio is fine.

## R-1's predictive record, scored at the intended −O level

**2 hits, 3 misses, 1 partial.**

| rung | predicted | actual |
|---|---|---|
| `beebs_bs` | PASS | ✅ PASS |
| `beebs_cnt` | PASS | ✅ PASS — **the cross-object control R-1 needed** |
| `beebs_fibcall` | PASS | ❌ miscomputes |
| `beebs_fac` | PASS | ❌ hangs |
| `beebs_duff` | PASS | ❌ hangs |
| `beebs_janne` | PASS | ❌ hangs (R-6) |

R-1 predicts the **memory-shape** outcomes well: both rungs whose failure mode it could
speak to came out as predicted, and `cnt` in particular confirms the same-object clause
that had never been tested. It does **not** explain the hangs, which is consistent with
the standing "≥2 independent faults" position rather than a mark against it.

`beebs_fibcall` is worth its own note: at −O1 it retires **166,539** instructions against
a baseline of **177,855** — it runs ~94 % of the workload and still returns a wrong
answer. That is a third distinct signature, different from both the hangs and from the
−O0 "compute never ran" pattern (18,118 instructions). It has no arrays at all.

## Board health

Not in question. The same `beebs_bs` binary returned the same value across two sessions
and a power cycle, 2,264 → 2,258 cycles (0.3 %). Every baseline retval matched its
oracle in both sweeps, and the second sample is byte-identical in instret on all five
rungs.

## Files

Kernels/apps/hosts for the four new rungs are in
`tests/runtime-qemu/silicon-ladder/beebs_{fibcall,fac,cnt,duff}_*`; all four are
QEMU-green through the identical controller. Both halves are registered in
`build-ladder-{fpga,base-fpga}.sh` and in `ladder_base_ctl.c`'s dispatch table.
