# I-2 CONFIRMED: the Linux baseline is ~1.21× slow on identical work, so every published overhead is UNDERSTATED

**Date:** 2026-07-28
**Lane:** C (primary)
**Cost:** 3 board boots. Board off + unlocked.
**Bottom line:** a domain cycle is not comparable to a Linux-baseline cycle. The error is
**proportional to work** and always in the direction that **flatters us**.

---

## The measurement

Two probes (`ctrsanity`, `ctrsanity4` = 4× the work) whose measured region is a
**5-instruction register-only loop** — no loads, no stores, no pointers, no globals inside
the loop. A capability target and a plain riscv64 target have nothing to differ about there.
Verified in the disassembly: both emit `srai / xor / addi / add / bne`.

| probe | cap cyc | base cyc | **cyc ratio** | ins ratio | cap CPI | base CPI |
|---|---:|---:|---:|---:|---:|---:|
| `ctrsanity` (100 k iters) | 600,309 | 728,727 | **0.824** | 0.982 | 1.201 | 1.431 |
| `ctrsanity4` (400 k iters) | 2,400,310 | 2,884,826 | **0.832** | 0.982 | 1.200 | 1.417 |

**The probe is valid.** The capability side retires **500,033** instructions = 5 × 100 000 + 33,
i.e. exactly the loop and nothing else, and 2,000,033 at 4×. Instruction ratio 0.982 at both
lengths; the ~1.8 % excess is on the **baseline** side.

**Cycles per loop iteration — the cleanest way to see it:**

| | 1× | 4× |
|---|---:|---:|
| capability domain | **6.003** | **6.001** |
| Linux baseline | 7.287 | 7.212 |

The domain is metronomic. The baseline is ~1.21× slower **executing the same five
instructions**.

## It is timer interrupts, and the evidence is quantitative

The baseline's excess over the domain scales with work:

| | 1× | 4× | scaling |
|---|---:|---:|---|
| excess instructions | 9,145 | 35,837 | **3.92×** |
| excess cycles | 128,418 | 484,516 | **3.77×** |

Both ≈ 4.0, i.e. **proportional to elapsed work**, which is what periodic interrupts look like
and what a fixed entry cost does not. The excess runs at **14 cycles per excess instruction** —
far above this core's ~1.2–1.4 CPI, exactly as expected for interrupt entry/exit plus the cache
and pipeline disruption it causes.

So the Linux baseline is being charged, inside our measurement bracket, for work the
bare-metal domain never does.

## Why this matters more than a lost benchmark row

`ratio = capability ÷ baseline`. Inflating the **denominator** makes the ratio **smaller**.
Every affected row therefore **understates capability overhead** — the error flatters our own
result, which is the worst direction for it to run.

Scale of the effect on a controlled workload: **1.214×**. If it applied uniformly:

| row | published | ×1.214 |
|---|---:|---:|
| `beebs_prime` | 1.032× | 1.253× |
| `rv8_primes` | 1.050× | 1.275× |
| `beebs_bs` | 1.181× | 1.434× |
| `beebs_recursion` | 1.801× | 2.186× |

**DO NOT APPLY THAT CORRECTION.** It is an illustration of magnitude, not a result. Two
reasons it is not uniform:

1. **Short kernels may take no interrupts at all.** `beebs_bs`'s baseline is 1,912 cycles
   (~40 µs); at a 1 kHz tick most passes contain zero interrupts. Its two passes retired
   **byte-identical** instruction counts (827/827), as did `beebs_recursion` (2,019/2,019).
   Those rows are probably clean and must not be inflated.
2. **The 14 cyc/instr figure includes cache disruption specific to this loop.** A kernel with
   a different footprint will pay a different amount per interrupt.

## The existing "certified clean" test is necessary but NOT sufficient

The project already treats *byte-identical instret across the cold and warm passes* as
certification that no interrupt landed. `ctrsanity`'s passes **differed** (510,224 vs 509,178),
so it fails that test and is correctly flagged. But `beebs_cnt`'s second sample was
**identical** (67,140/67,140) and its ratio is still an impossible **0.684×** — two passes can
take the *same* number of interrupts and both be contaminated. Identical instret proves
*reproducibility*, not *absence*.

`beebs_cnt` is worse than this probe (1.46× implied environment factor vs 1.214×), so
interrupts do not fully explain it either. **`cnt` remains unexplained beyond this.**

## What to do

Ranked, and none of these is "adjust the numbers":

1. **Run the baseline bare-metal**, in the same environment as the domain. This is the
   principled fix and it removes the confound rather than modelling it. It is real work: the
   baseline kernels currently link into a Linux userspace binary.
2. **Lead with instruction ratios.** They are far less contaminated (0.982 vs 1.000 here,
   against 0.824 for cycles), and the paper's central claim — *the cost is the ABI, not the
   enforcement* — is an instruction-count argument already.
3. **Per-row triage before publishing any cycle ratio.** A row is safe only if its baseline is
   short enough to be plausibly interrupt-free, and "identical instret" alone does not
   establish that.

## Consequence for the paper

The four-row cycle table cannot be published as it stands without at least a per-row
justification, and the honest headline may be **larger** overheads than we have been claiming,
not smaller. The instruction-ratio story is unaffected and is the stronger claim anyway.

This is the second time a baseline-environment confound has bitten this measurement — the
first was cold-vs-warm paging, which produced "capabilities are 1.8× faster" for `beebs_prime`
and led to the warm-baseline rule. That rule does not cover this one.
