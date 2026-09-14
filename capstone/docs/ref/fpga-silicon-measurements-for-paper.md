# Silicon results for the paper

Consolidated, paper-facing extract of every measurement taken on the CapliFive CVA6
FPGA. The dated notes in `history/` are the investigation trail; **this file is what
a paper author lifts from**. Each entry states the number, the exact conditions, what
it supersedes in the current draft, and what it does **not** establish.

Vehicle throughout: Genesys 2 CVA6, bitstream `working-caplifive-captype-fixed.bit`
(**superseded 2026-08-04** by `caplifive_fixed_forward.bit`; every number below predates
that reflash and must be re-measured before it is compared with anything taken after),
`mcycle`/`minstret` read in-domain (the unprivileged counters are gated for the
domain). Single core.

Last updated 2026-07-27.

**Benchmark scope is capped by a 4 KiB code window — but the cap is liftable.**
`link-gpfree.ld` forces globals to image offset `0x1000`, so every domain's `.text`
must fit 4096 bytes. That is why the benchmarks are kernel *slices*: `coremark_matrix`
is CoreMark's matrix phase only, deliberately excluding its linked-list phase, and at
~56k cycles it should NOT be called CoreMark without scaling.

The limit is one hardcoded number, not a hardware constraint — the monitor splits at
a **runtime** `code_size` (the whole image size), `gp` is carved from `dom_data`'s
end, and `GPFREE_GLOBALS_OFFSET` appears only in comments. QEMU-validated at 16 KiB
(with a rung having an **initialized** global, exercising the large-RO delivery path)
and at 32 KiB. **SILICON-VALIDATED at 32 KiB as of 2026-09-05** — corrected 2026-09-10; this
line previously read "Not yet silicon-validated" and was overtaken by `board-results/2026-09-05.tsv`,
which carries three passing 32 KiB-window rows (`b2d072e8cbac2b41`, `ffedb2e78f650c94`,
`2fc5b371cc8c7b2a`). Lifting it is what would make full CoreMark
and Dhrystone buildable.

---

## 0. Terms, and how to read every table below

Read this once; every number in this file is one of these.

**The two hardware counters.** RISC-V CPUs expose free-running 64-bit counters. We read
each one immediately before and immediately after the code we care about and subtract,
so every number below is a *delta*, not an absolute.

| term | what it is |
|---|---|
| **cycle** / `mcycle` | Clock ticks elapsed. This is **time** (at a fixed clock, 1 cycle = 1 tick). The thing we ultimately care about. |
| **instret** / `minstret` | **Instr**uctions **ret**ired = how many instructions actually completed. This is **work done**, independent of how fast the machine ran them. |
| **CPI** | **C**ycles **P**er **I**nstruction = `cycles ÷ instret`. How expensive the average instruction was. CPI 1 = one instruction finishes per tick (a perfect pipeline). CPI 2 = each instruction costs two ticks on average — stalls, cache misses, multi-cycle ops. **⚠ CORRECTED 2026-09-10: this CVA6 measures 1.13–6.44 across §4's own table, and two rows ARE near 1.**
The previous wording here said "2.0–3.2, never near 1", and the table it summarises contradicts both
halves: `beebs_aha_mont64` is 1.13 and `ctrsanity` is 1.20, while `beebs_recursion` is 6.44, twice the
old upper bound. Computed straight from the capability column's `cycles / instret` pairs. **The spread
is workload-dependent by a factor of nearly six, so quoting a single CPI range as a property of the
core is the error; quote the row you mean.** This was found when the range was passed to another lane
as an authoritative bound for a runtime estimate — it came from this glossary line rather than from the
data eight lines below it. |

Why both counters matter: cycles alone cannot tell you *why* something is slower. A build
can be slower because it **executes more instructions** (an ABI/codegen cost) or because
**each instruction costs more** (a hardware cost). `instret` separates those two. That
separation is the whole of §3.

**The two builds being compared.** Every overhead number is a ratio of two runs of the
*same source*:

| term | what it is |
|---|---|
| **capability** build | Compiled `-target capstone64` as a pure-capability Capstone **domain** — every pointer is a 128-bit bounded capability. This is the thing being priced. |
| **baseline** build | The identical source, same clang, same `-O`, compiled `-target riscv64` with **no** capability flags. Ordinary RISC-V. This is the denominator. A static gate fails the build if a capability instruction leaks into it. |
| **overhead ratio** | `capability ÷ baseline`. `1.032×` means the capability build cost 3.2% more. Reported separately for cycles and for instructions. |

**Warm vs cold** — applies to the **baseline** only, and it is load-bearing:

| term | what it is |
|---|---|
| **cold pass** | The baseline's *first* run. It pays first-touch **page faults** — the Linux kernel maps each page on first access — *inside* our measurement bracket. |
| **warm pass** | A *second* run of the same code in the same process. Pages are already mapped, so no fault cost. |
| **why warm** | The capability domain has **no paging at all**, so a cold baseline would be charged for something the capability side never pays. Using cold instead of warm gives `beebs_prime` = 0.544×, i.e. "capabilities are 1.8× *faster*" — which is how the confound was caught. **Always warm.** |

So **"warm cycles" / "warm instret"** in §4 just mean: those counters, measured on the
warm (second) baseline pass.

**Other vocabulary used below**

| term | what it is |
|---|---|
| **rung** | One benchmark in the "silicon ladder" — a small kernel (`<name>_kernel.h`) shared verbatim by the capability domain, the baseline, and a native host oracle. |
| **domain** | A Capstone protection context. Entered via `cscall`, left via `csreturn`. |
| **oracle** | The expected answer, computed natively on the host from the identical source. A rung is only counted if the board returns exactly this. |
| **bracket the compute only** | Both counter reads sit *inside* `domain_main`, around the kernel — so domain entry/exit is excluded from both halves and needs no correction. |
| **gp cap-table** | The silicon ABI in use: globals are reached indirectly through a table of capabilities based at `gp` (`ldc rd, i*16(gp)`), rather than by direct address. Source of most of the measured overhead — see §3. |
| **`ldc`** | Load capability (128-bit). The cap-table indirection instruction. |

---

## 1. Primitive costs — cycle-accurate (already in the draft)

Feeds `tab:primcost-rtl`. Method: each operation is an `N`-iteration inner loop
bracketed by two `mcycle` reads with an empty calibration loop subtracted. `cyc/op` =
**cycles per single operation** (the loop total divided by `N`, calibration removed).

**The primitives, in words** — these are the Capstone ISA operations that implement
lending a pointer to another domain and later taking the authority back:

| primitive | what it does | cyc/op |
|---|---|---:|
| `load` | Load through a capability (an ordinary memory read, bounds-checked by hardware). | 2 |
| `shrink` | Narrow a capability's bounds to a sub-range. Pure register op. | 1 |
| `mrev` | **M**int a **rev**ocation node: create a handle that can later invalidate everything derived from this capability. The bookkeeping that makes revocation O(1) later. | 50 |
| `delin` | **Delin**earise: turn a linear (uniquely-owned) capability into a copyable one. | — |
| `revoke` | Invalidate every capability descended from a revocation node. | — |
| `delin` + `revoke` | Measured together. | 121 |
| **`mrev`+`delin`+`revoke`** | **Reclaim** — the full cost of taking authority back after one lend. | **171** |
| **borrow** | reclaim + load — the end-to-end cost of one lend/use/reclaim cycle. | **~173** |

`171` is the number §4's SQLite estimate multiplies by the borrow count.

Super-operation view — the same costs compared against an ordinary unprotected pointer,
to show what the safety actually buys and costs:

| operation | cyc/op | vs raw |
|---|---:|---:|
| raw pointer | 8 | 1.0× |
| **capability borrow** | **182** | **22.8×** |
| copy — 256 B | 902 | 112.8× |
| copy — 1024 B | 3611 | 451.4× |

Growth: **borrow(N) ≈ 75 + 3·(N/2)** cyc/op — base ≈75 cyc, ≈3 cyc per accumulated
revocation node. The temporal cost is the revocation tree (`mrev` + `revoke`);
`delin`, `load` and `shrink` are 1–2 cyc register ops.

Source: `history/21-07-2026_16-12-13_RESULTS-fpga-borrow-cost-cycle-accurate.md`.

---

> ## ✅ RESOLVED 2026-07-28 — measure against the BARE-METAL baseline
>
> The Linux baseline was serving timer interrupts inside the bracket (**I-2**). It is
> replaced by an S-mode OpenSBI payload with **no OS**. Proof it works: the `ctrsanity`
> control, whose clean value is known independently from the domain side, now reads
> **600,041 cyc bare vs 600,309 cyc capability — ratio 1.000** (was 728,727, i.e. 1.21x).
> Quality: **15/15 passes tied at min instret, spread 0** on nearly every rung.
>
> **Every overhead ratio rose. Use the table below; the old numbers are withdrawn.**
> Build: `build-ladder-base-bare.sh`; run: `fpga_driver/run_base_bare_fpga.py`.
> Trail: `history/28-07-2026_02-30-00_RESULTS-bare-metal-baseline-works-*.md`.

## 2. Pervasive spatial safety costs 0 %–96 % in cycles (bare-metal baseline)

The draft claims spatial safety is pervasive ("every pointer is a bounded
capability, always on"), demonstrates it is **correct**, and never prices it.
This is that price, measured.

> **IN THE PAPER as of 2026-07-27** — `old-parts/evaluation.tex`, new subsection
> `sec:eval-spatial-cost` + `tab:spatialcost` (paper commit `524f5d0`, **local
> only, not pushed** because that repo syncs with Overleaf). §3's ABI-not-hardware
> split is written up in the same subsection, and §4's measured CPI 2.0–3.2 now
> corrects `tab:appoverhead`'s CPI=1 footnote (the SQLite rows are ~2× what this
> hardware would pay; the conservative figures are kept deliberately, since the
> measured CPI comes from these kernels rather than from SQLite).
> **UPDATED 2026-07-28 (rungs 6 and 8).** `tab:spatialcost` now carries `cover` and
> `aha-mont64`, the reading paragraph says **0 %–96 %** rather than 5 %–96 %, and the
> mechanism sentence (*overhead is a property of data access, not execution*) is stated
> in the paper for the first time — it is the strongest claim the table supports and it
> was previously only in this document. The "hardware limitation" paragraph was also
> corrected: it attributed **all** unmeasured kernels to the register-indexed-load
> defect, which R-6 and R-9 refute.
> **When new rungs land, edit `tab:spatialcost` — it is built to take more rows.**

Method: each kernel compiled **twice from the identical source header** by the
**same clang at the same `-O`** — once for `-target capstone64` as a pure-capability
domain, once for `-target riscv64` with no capability flags — and run on the same
board. A static gate fails the build if a capability instruction reaches the
baseline. Baseline is the **warm** pass (the capability domain has no paging). Both
halves bracket the compute only, so domain entry/exit is excluded from both.

The `capability` and `baseline` columns are **cycles**. The two bold columns are the
overhead ratios (capability ÷ baseline) for cycles and for instructions respectively.

### FINAL — uniform −O1, bare-metal baseline, one harness, one session (2026-07-28)

| benchmark | opt | capability | baseline | **cycles** | **instr** | **CPI ratio** |
|---|---|---:|---:|---:|---:|---:|
| `beebs_prime` (pure scalar) | −O1 | 9,782 / 2,708 | 9,283 / 2,704 | **1.054×** | 1.001× | 1.052 |
| `rv8_primes` (sieve) | −O0 [*] | 17,283,292 / 8,773,753 | 13,679,903 / 7,764,899 | **1.263×** | 1.130× | 1.118 |
| `beebs_cnt` (matrix seed+sum) | −O1 | 128,175 / 76,429 | 94,736 / 57,949 | **1.353×** | 1.319× | 1.026 |
| `beebs_bs` (binary search) | −O1 | 2,259 / 875 | 1,470 / 827 | **1.537×** | 1.058× | 1.452 |
| `beebs_recursion` (deep+mutual) | −O1 | 18,971 / 2,944 | 9,696 / 2,019 | **1.957×** | 1.458× | 1.342 |
| **`beebs_cover`** (switch coverage, control-flow only) | −O1 | 159,952 / 76,471 | 167,778 / 76,513 | **0.953×** ⚠ | **0.999×** | 0.954 |
| **`beebs_aha_mont64`** (Montgomery modmul, no arrays) | −O1 | 290,071 / 256,699 | 283,612 / 256,622 | **1.023×** | **1.000×** | 1.023 |
| **`ctrsanity`** (**control**: identical code both sides) | −O1 | 600,309 / 500,033 | 600,041 / 500,022 | **1.000×** | 1.000× | 1.000 |

Cells are `cycles / instret`. **Pervasive spatial safety costs 0 %–96 % in cycles**,
depending entirely on kernel shape.

⚠ **`beebs_cover` reads 0.953×, i.e. nominally *faster*. Do NOT report it as a speedup.**
The instruction ratio is **0.999×** — both builds retire the same work (76,471 vs 76,513,
42 apart out of 76,000) — so there is no mechanism by which capabilities could make this
kernel faster. The 4.7 % cycle difference is code layout/alignment, and layout sensitivity
is documented here (2026-07-26: four added instructions flipped a passing rung). **Report it
as "no measurable overhead", not a negative cost.**

**Re-measured 2026-07-28 under the corrected ABI, and the row survives.** It was first
taken while GlobalMerge was packing this rung's 3 globals into one `.L_MergedGlobals`
container -- i.e. one union-bounds capability, not the per-object bounds the paper
claims. GlobalMerge is now disabled under gp-captable (an ABI invariant, not a tuning
knob) and the rung was re-run on the board: **290,071 cycles** against the previous
289,869, with instret byte-identical at 256,699. A 0.07 % cycle difference, so the
conclusion is unchanged -- but it is now measured in the configuration the paper
describes rather than a weaker one.

**`beebs_aha_mont64` is the independent confirmation `cover` needed** (added 2026-07-28).
It retires **256,699 vs 256,622** instructions — **77 apart out of 256,000, a ratio of
1.0003×** — for 1.023× the cycles. Its value is that it reaches the same place as `cover`
by a completely different route: `cover` is control-flow (180 switch dispatches, no data),
`mont64` is straight-line 64-bit arithmetic (a 64-iteration shift-and-subtract modulus, a
64-iteration binary GCD, 64×64→128 multiplies) with 24 B of scalar globals and no array of
any kind. Two unrelated execution profiles, both landing at a **1.000× instruction ratio**.
One rung at ~1.00× invites the reading that it was a lucky kernel; two, sharing only the
property of not touching data, make it the mechanism.

It also cleanly separates *cost* from *work*: mont64 is the second-longest rung in the
table by instructions (256 k, behind only `ctrsanity` and `rv8_primes`), so the ~0 %
overhead is not an artefact of a short run.

**It is the most informative row, not merely the sixth.** `cover` is control-flow dominated
— 180 switch dispatches per call, one global, essentially no data traffic — and its 0.999×
instruction ratio says the gp-captable ABI adds **nothing** when a kernel does not touch
data. Beside `cnt` (+31.9 % instructions, bulk array work) and `bs` (+44.6 % CPI,
dependency-chained loads), the table now separates the cost cleanly: **capability overhead
is a property of DATA ACCESS, not of execution.**

[*] **`rv8_primes` HANGS at −O1 on this silicon** and is measurable only at −O0 — a real
limitation, reported rather than hidden. Everything else is −O1. (Its −O0 pair is
internally consistent, so the ratio is valid; only cross-row `-O` comparison is affected.)

### Rungs that do NOT appear in the table, and why (2026-07-28)

Eight rows are measured. Coverage is bounded by silicon failures, not by effort, and the
bound is **not** a single cause — that is the honest statement and it differs from what
§5 said before today.

| rung | QEMU | baseline half | capability half on silicon | issue |
|---|---|---|---|---|
| `matmult_int` | pass | clean | `cscall` hangs at every reachable config | R-1 |
| `coremark_matrix` | pass | clean | `cscall` hangs at every reachable config | R-1 |
| `beebs_crc32` | pass | clean | hangs | R-1 |
| `beebs_insertsort` | pass | clean | wrong value, 560 instrs (compute never ran) | R-1 |
| `beebs_janne` | pass | clean | hangs — **R-1 predicts PASS** | R-6 |
| `rv8_sha512` | pass (oracle 1390718314) | 540,073 / 462,646 | hangs, both attempts | R-7 → R-1 |
| `rv8_sha512s` | pass (oracle 2842840124) | 117,035 / 69,108 | hangs (4 KiB control for R-7) | R-1 |
| `beebs_ns` | pass (oracle 1184999093) | 88,451 / 62,097 | hangs — **R-1 predicts PASS** | R-9 |

Every one of these has a **clean, correct baseline half**, so the failure is specific to
the capability build and not to the kernel or the harness. That is what makes them
reportable as a platform limitation rather than as "benchmarks we could not get working".

**Two of the eight are not explained by R-1** (`beebs_janne`, `beebs_ns`): neither writes
the object it indexes, so R-1's same-object load-with-intervening-store shape is absent.
Do not write "all remaining failures are the register-indexed-load defect" — it was
written once and R-9 falsified it the next day.

### Why the overheads are what they are

`cycles_ratio = instr_ratio × CPI_ratio`, and **which factor dominates is the finding**:

| benchmark | cycles | = instructions | × CPI | dominated by |
|---|---:|---:|---:|---|
| `beebs_prime` | 1.054 | 1.001 | 1.052 | neither — essentially free |
| `rv8_primes` | 1.263 | 1.130 | 1.118 | balanced |
| `beebs_cnt` | 1.353 | **1.319** | 1.026 | **extra instructions** |
| `beebs_bs` | 1.537 | 1.058 | **1.452** | **stalls** |
| `beebs_recursion` | 1.957 | **1.458** | **1.342** | both |
| `beebs_aha_mont64` | 1.023 | 1.000 | 1.023 | neither — no data to access |

- **`cnt` (bulk array work)** — nearly all cost is the +31.9 % instructions our globals
  ABI emits (`ldc gp[i]` per global reference); per-instruction speed barely moves.
- **`bs` (search)** — the opposite: only **+5.8 % instructions** but **+53.7 % cycles**.
  Each `bs_data[mid]` load depends on a `cincoffset` that depends on an `ldc` from the
  cap table — a serial dependency chain, and this is an **in-order** core, so it stalls.
- **`recursion`** — both, because the gp-free call/return sequence *and* 128-bit
  capability spills are paid on **every call**. Its absolute CPI is 6.44, by far the
  highest.
- **`prime`** — at −O1 the compiler keeps globals in registers, so the indirection is
  amortised almost entirely: **1.054×**. At −O0 the same kernel measured **1.683×**,
  because every access reloaded from the cap table.

**That last point matters for how the whole table is read.** Optimisation level changes
the answer by more than the capability mechanism does on scalar code, which is why the
table is now uniform −O1 (realistic) rather than mixed.

**`beebs_bs` added 2026-07-27 — four rows now.** Capability CPI rises 2.31 → 2.58; same
*more instructions, ABI not enforcement* shape as the sieve. The capability binary
reproduces across two sessions and a power cycle (2,264 → 2,258 cyc, 0.3 %), and both
halves are −O1. Trail:
`history/27-07-2026_22-40-00_RESULTS-two-new-silicon-rungs-and-an-O-level-procedure-bug.md`.

> **`beebs_cnt` is silicon-CORRECT but its cycle ratio is NOT publishable.** It returns
> its oracle exactly (2,356,896,837) and retires **1.138×** the baseline instructions —
> credible, in family. But it takes **0.684×** the cycles, i.e. it would claim pervasive
> capability safety makes code **32 % faster**. That is an uncontrolled confound, not a
> result. Capability CPI 1.68 vs baseline CPI 2.79: the baseline is a Linux userspace
> process while the domain is bare-metal with a clean icache and no OS, and for a 400 B
> working set the baseline may be charged for interference the domain never sees. This
> is the same *class* as the cold/warm paging confound that once produced "capabilities
> are 1.8× faster" for `beebs_prime` — so the warm-baseline rule does **not** cover it.
> **`beebs_bs` and the sieve do not show it, but it is NOT established that the existing
> rows are free of it.** `beebs_prime` (1.032×) is the one to re-examine, because a
> confound in this direction would *understate* capability overhead. Quote `cnt`'s
> instruction ratio only, or hold the rung back entirely.

> **[†] Why `beebs_prime` has no instruction ratio — this is a finding, not a gap.**
> The ratio needs `instret` from *both* halves. The baseline half has it (14,680, see §4).
> The **capability** half does not, and cannot be measured today: reading `minstret`
> inside the domain requires adding instrumentation to `domain_main`, and **adding that
> instrumentation changes the answer this rung computes.** A controlled A/B on the board
> settled it — the instrumented build (`LADDER_INSTR_MODE=4`) returns a *wrong*,
> deterministic value; the un-instrumented build (`mode 0`, `mcycle` only) returns the
> oracle. Four instructions, none inside the computation.
> So the trustworthy `beebs_prime` capability run is the un-instrumented one, which reads
> **cycles only**. Publishing an instruction count for it would mean publishing a number
> taken from a run that is known to compute the wrong result.
> Two consequences worth carrying: the missing cell is **evidence for the miscompute
> bug**, and it means **a passing rung is not stable ground** — re-gate on the oracle
> after *any* change to `domain_main`. (`rv8_primes` and `beebs_recursion` tolerate the
> instrumentation and still return their oracles, which is why they have both columns.)

**The spread is the result.** Report the range and the mechanism, not an average.

Recursion is the outlier for a legible reason, visible in the counters: it retires
45.8% more instructions (against 10.2% for the sieve) *and* its CPI rises from 5.21
to 6.44. A gp-free call/return plus capability spills to the stack are paid **per
call**, and `beebs_recursion` is nothing but calls; the sieve amortises its `ldc`
cap-table indirections over long straight-line loops.

`beebs_recursion` is certified clean: its two baseline passes retired byte-identical
instruction counts (2,019/2,019), so neither counted a page fault or interrupt.

Two conditions on citing this table:
- Each pair is internally consistent (same compiler, same level, both sides), but the
  **set mixes levels** — `beebs_recursion` is at −O1 because that is the level at
  which it computes correctly on silicon.
- Using the *cold* baseline instead of warm gives `beebs_prime` = 0.544×, i.e.
  "capabilities are 1.8× faster". That is how the paging confound was caught.

Sources: `history/26-07-2026_14-46-43_RESULTS-silicon-spatial-safety-overhead-baseline.md`,
`history/26-07-2026_19-31-06_RESULTS-three-benchmarks-on-silicon-and-the-hang-blocker.md`.

---

## 3. "That overhead is ABI, not hardware" — REFUTED (2026-07-28): CPI RISES

> ### ❌ REFUTED 2026-07-28 — measured, not estimated
>
> The claim rested on `rv8_primes` retiring **more instructions than it cost cycles**,
> CPI *falling* 2.07 -> 1.98, read as "bounds enforcement is near-free per instruction;
> the cost is the ABI". Against the bare-metal baseline it **reverses**:
>
> | | cycles | instructions | CPI |
> |---|---:|---:|---:|
> | baseline (bare) | 13,679,903 | 7,764,899 | **1.762** |
> | capability | 17,283,292 | 8,773,753 | **1.970** |
> | ratio | **1.263x** | **1.130x** | CPI **RISES** |
>
> Cycles grow **faster** than instructions. The old "CPI falls" was an artifact of
> interrupts inflating the *baseline's* CPI — interrupt handling runs at ~14 cycles per
> instruction against real code's ~1.8, i.e. it inflated exactly the quantity the
> argument turned on. (The pre-registered estimate said baseline CPI ~1.742; measured
> 1.762.)
>
> **Do not claim capability enforcement is free per instruction.** On every rung with
> both counters, cycles grow faster than instruction count. Some of the cost is the ABI's
> extra instructions, but it is NOT the whole story and this section must be rewritten
> rather than re-cited.

### The question this answers

§2 says the capability build costs 5.6% more cycles on `rv8_primes`. That alone does
not say **why**, and the "why" decides whether the number is a property of *Capstone as
a hardware design* or of *our current compiler ABI*. Exactly two things can make a
program take more cycles:

1. it **executes more instructions** — a codegen/ABI cost, fixable in software; or
2. **each instruction costs more** — a hardware cost: extra pipeline stalls to check
   bounds, wider 128-bit operands, more cache pressure.

Reading `instret` as well as `mcycle` separates them, because
`cycles = instret × CPI`. If the extra cycles come with proportionally more
instructions, it is (1). If CPI rises, it is (2).

### The measurement

Both counters, `rv8_primes`, same session (so same clock, same board state):

| | cycles | instret | CPI = cyc/instr |
|---|---:|---:|---:|
| capability domain | 17,375,220 | 8,773,753 | **1.98** |
| baseline (warm) | 16,459,057 | 7,960,829 | **2.07** |
| **ratio (cap ÷ base)** | **1.056** | **1.102** | — |

> **Why the capability cycles here (17,375,220) differ from §2's (17,283,292) — same
> rung, same board.** They are different builds. Reading `minstret` from inside the
> domain requires instrumentation in `domain_main`, and **that instrumentation costs
> cycles**: +91,928, or +0.53%. §2 quotes the *un-instrumented* run; §3 must use the
> instrumented one, because it is the only one that reports instructions at all.
> Consequence: **§3's cycle ratio (1.056) is very slightly inflated by the measuring
> instrument; §2's (1.050) is the one to cite** for the headline overhead. The
> instruction ratio (1.102) and CPI are unaffected — both halves of §3 come from the
> same instrumented pairing.
> The same observer effect appears in §2's `[†]` footnote in a far more severe form:
> on `beebs_prime` the instrumentation does not merely cost 0.5%, it changes the
> computed *result*. Here it is benign and quantified; there it is disqualifying.

### Reading it

- **+10.2% instructions, but only +5.6% cycles.** The instruction count grew *nearly
  twice as fast* as the time did.
- **CPI went DOWN**, 2.07 → 1.98. The average capability-build instruction was
  *cheaper* than the average baseline instruction.
- Cost of the extra work, isolated: the capability build retires
  `8,773,753 − 7,960,829 = 812,924` extra instructions and spends
  `17,375,220 − 16,459,057 = 916,163` extra cycles on them
  ⇒ **916,163 / 812,924 = 1.13 cycles per extra instruction**, against a program
  average of 2.07.

So the added instructions are **roughly half as expensive as a typical instruction in
this program**. That is the signature of simple, independent, well-pipelined loads —
not of stalls.

### Why that is the expected shape

The extra instructions are the **gp cap-table indirections**. Under this ABI a global
is not addressed directly; the domain loads its capability out of a table first:

```
ldc  rd, i*16(gp)      # fetch the capability for global #i
<use rd>               # then access through it
```

These are independent loads off a hot, tightly-packed table — ideal for the pipeline,
hence 1.13 cyc each. Nothing in the measurement is attributable to bounds checking.

### Claim this supports

**Capability enforcement is essentially free per instruction on this CVA6.** The
measured overhead is an **ABI cost**, not a hardware cost — so it is the kind of number
a better compiler reduces, and it should not be presented as the intrinsic price of
capability hardware. A tuned ABI (caching cap-table entries, hoisting `ldc` out of
loops, or addressing globals through `PCC` where provenance allows) would shrink it.

### Caveats — do not drop these

- **One benchmark.** This is the caveat that matters most. `rv8_primes` is a sieve:
  long straight-line loops over arrays, which is precisely the shape that amortises
  cap-table indirection best. It is the *friendliest* case for this claim.
- **The counter-example is in §2.** `beebs_recursion` goes the other way — instructions
  +45.8% **and** CPI 5.21 → 6.44. Per-call costs (gp-free call/return, capability
  spills to the stack) are not amortisable. So "CPI does not rise" is a property of
  *this workload shape*, not a universal result.
- CPI here is being used descriptively (a measured ratio), not as a model of the
  microarchitecture.

Source: `history/26-07-2026_15-58-45_overhead-decomposed-and-fault1-reproduces-in-perf-rungs.md`.

---

## 4. NEW — Measured CPI replaces an assumption in the paper's SQLite estimate

### What `tab:appoverhead` is

A table in the paper draft that estimates **what fraction of SQLite's runtime would be
spent on Capstone's domain-boundary operations**. SQLite is too large to run on the
board today (§5), so unlike §§1–3 this row is **not** a measurement — it is an estimate,
and this section fixes one input to it.

### The estimate's arithmetic

The cost being estimated is *borrowing*: each time SQLite hands a buffer across a domain
boundary, the capability must be lent and later reclaimed. §1 measures that at
**171 cycles per borrow** (cycle-accurate, on this board). So:

```
                    borrows × 171 cycles          <-- cost added by Capstone
overhead fraction = --------------------------
                    total cycles SQLite would
                    have taken anyway             <-- the denominator
```

The numerator is measured. **The denominator is the problem:** we know roughly how many
*instructions* SQLite executes, but the formula needs *cycles*. The draft bridges that
with `cycles = instructions × CPI`, giving:

```
overhead fraction = (borrows × 171) / (instructions × CPI)
```

### The assumption, and why it was wrong in a specific direction

The draft sets **CPI = 1** and calls it "the conservative upper bound" (with a CPI = 1.5
sensitivity case). CPI = 1 is the *smallest physically plausible* value — one instruction
completing per clock tick. Since CPI sits in the **denominator**, the smallest CPI gives
the **smallest denominator**, hence the **largest** overhead fraction. That is what makes
it an upper bound: deliberately pessimistic, so the paper cannot be accused of
flattering itself.

The bound is honest but **loose**, because a real CVA6 does not run at CPI 1. So we
measured it — on the same board, on the warm baseline pass, five kernels:

| benchmark | warm cycles | warm instret | **CPI** |
|---|---:|---:|---:|
| `coremark_matrix` | 55,975 | 27,788 | **2.01** |
| `rv8_primes` | 16,459,057 | 7,960,829 | **2.07** |
| `matmult_int` | 71,860 | 29,661 | **2.42** |
| `beebs_insertsort` | 8,398 | 3,410 | **2.46** |
| `beebs_prime` | 46,306 | 14,680 | **3.15** |

*(These are the **baseline**, plain-RISC-V builds — the denominator represents SQLite
without Capstone. "Warm" = the second pass, no page-fault cost inside the bracket; see
§0. Every row here is just `cycles ÷ instret`.)*

**Measured CPI is 2.0–3.2 — never below 2, i.e. at least twice the assumed value.**
The spread is wide because CPI depends on the workload's memory behaviour, but the
*floor* is what matters here, and the floor is ~2.

### The consequence

Doubling CPI doubles the denominator and therefore **halves** the estimated overhead.
Worked at the low end of the measured range (CPI ≈ 2, the conservative choice within
the measurements):

```
overhead(CPI=2)      (borrows × 171) / (instr × 2)      1
---------------  =  ----------------------------  =  ---  ⇒ halve the draft's figures
overhead(CPI=1)      (borrows × 171) / (instr × 1)      2
```

| workload | draft (assumed CPI = 1) | **at measured CPI ≈ 2** |
|---|---:|---:|
| `speedtest1` (whole benchmark) | ~1% | **≈0.5%** |
| in-domain result scan (worst case) | ≤6% | **≈3%** |

*(The borrow counts and instruction counts behind these two rows live in the paper
draft; this section changes only the CPI input, so both rows scale by the same factor.)*

### Why this makes the paper stronger, not weaker

An **assumption becomes a measurement**. The old number was defensible but arbitrary —
a reviewer could ask "why 1?" and the honest answer was "because it is the worst case".
Now the answer is "because we measured 2.0–3.2 on the actual silicon, and used the low
end". The estimate gets *better* **and** more favourable at the same time, which is
rare; usually honesty costs you something.

### Caveats — do not drop these

- **This is still an estimate, not a measurement of SQLite.** Both rows depend on modelled
  borrow counts, which is the larger uncertainty — bigger than the CPI input this section
  fixes.
  *(UPDATED 2026-08-20. This caveat previously read "SQLite has not run on the board."
  That is no longer true — SQLite now runs its full self-checking workload on silicon and
  passes, 3/3, control green. **The caveat itself still stands**, for a different reason:
  what silicon has demonstrated is CORRECTNESS, not timing. No admissible cycle number
  exists from those runs, so these rows remain an estimate. Do not read the correctness
  result as a performance measurement.)*
- **CPI is workload-dependent**, 2.01–3.15 even across five small kernels. Using ≈2 is
  the conservative choice *within* the measured range; quoting 3.15 would halve the
  overhead again but is not defensible.
- **None of the five kernels chases pointers**, and SQLite does heavily. Pointer-chasing
  code typically has *higher* CPI (cache misses), which would push the estimate lower
  still — so ≈2 remains conservative for SQLite specifically.
- `cycles = instructions × CPI` is a definition, not a model; it is exact for whatever
  CPI the program actually exhibits. All the uncertainty is in *which* CPI to use.

---

## 4b. SQLite in the SILICON config — QEMU, 2026-07-28 (NEW)

SQLite had **never been compiled for the silicon ABI** before this date; the paper's
own text still says "SQLite has not run on the board." *(As of 2026-08-20 that sentence in
the paper is stale and needs the project lead's attention: SQLite has now run on the board
and passes its correctness workload. Flagged here rather than edited — `capstone/paper/` is
not ours to change.)* Both halves now run under QEMU
in the full silicon configuration (`-capstone-gp-captable` + gp-free call/ret +
shrink off + `-fno-jump-tables`, one module, descriptor-driven entry glue):

| | result |
|---|---|
| existence proof (CREATE/INSERT/SELECT) | all five markers: `alpha=11`, `beta=22`, `gamma=33`, `__CAPSTONE_SQLITE_EXTENDED_PASSED__`, `__CAPSTONE_SQLITE_MEMORY_PASSED__` |
| boundary workload | `rows=200 borrows=400 scan_instrs=790,003` → **1,975 instructions per borrow** |

Build shape, for the record: `.text` 1,308,416 B, globals offset `0x140000`,
**1,059 globals each behind its own bounded capability** (`ldc-gp` = 583 sites),
`cjalr` = 0, exactly one `.capstone_gp_table` header (i.e. no multi-TU index
collision). memsys5 arena reduced 1 MiB → 256 KiB because under gp-captable every
global's storage is carved from `dom_data` and is therefore charged against the
domain's stack budget, not just image space.

**Comparison, not replacement.** The previously reported ~2,863 instructions/borrow
came from the NON-silicon build (shrink on, gp-free off). The silicon figure is
**lower**, which is the expected direction: `-capstone-shrink-globals=false` removes
the per-access narrowing sequence. Do not present 1,975 as a correction of 2,863 —
they are different configurations, and the honest statement is that the boundary cost
is ~2,000 instructions/borrow in the configuration the silicon run uses.

**Still QEMU, not silicon.** This is the S5 gate of `plans/sqlite-on-silicon-scoping.md`,
not a board measurement. What remains before a board number: R-3's `fence.i` fix (each
run otherwise costs a full power-cycle plus a ~2 min firmware reload), and baking the
1.4 MB domain into the rootfs — UART transfer is ruled out by measurement at >= 63 min.

## 4c. SQLite ON THE BOARD — delivery works, the domain does not complete (2026-07-29)

First attempt to run SQLite on the FPGA. **Delivery is solved; execution is not.**

What worked, on real hardware:
- SQLite ships **inside the buildroot initramfs** and arrives with the firmware over
  JTAG — `ls /test-domains/` on the booted board lists `sqlite_silicon.dom`
  (1,623,008 B) with **no transfer step at all**. This is the route the board owner
  described, and it is now built end to end (17.5 MB `fw_payload`, FDT embedded,
  first 64 bytes matching the known-good firmware).
- The host loader parses the image and computes **`Globals offset = 0x140000`**
  correctly on silicon — so the C-12 plumbing works on hardware, not just QEMU.

What did not: after `Loadable size = 1389480` the domain produces nothing and the run
times out. No fault line, no output.

**Most likely cause, and it is a step that was skipped rather than a new unknown.**
The plan (`plans/sqlite-on-silicon-scoping.md`, S7) says to climb the code window with
a trivial rung first: 32 KiB → 256 KiB → 1 MiB → full size. C-5 is validated on
silicon only to **32 KiB**, and SQLite jumps straight to a **1.3 MB PCC** — a 40×
extrapolation, on a platform with documented layout sensitivity. Going straight to the
top was my shortcut and this is the predicted way for it to fail.

Second candidate, not yet excluded: on the caplifive-system monitor
`capstone_error(code)` is `#define ... while(1);` — the code is discarded and the
monitor **spins silently**. A blob-does-not-fit would therefore look identical to a
domain hang. The QEMU monitor prints; this one does not. Making them agree is cheap
and would have distinguished these two on the first run.

**Next, in order:** (1) give the FPGA monitor a real `capstone_error` so failures are
distinguishable; (2) climb the window with a trivial rung at 256 KiB and 1 MiB; (3)
re-run SQLite. Nothing here suggests the ABI work is wrong — every piece of it is green
under QEMU and the offset demonstrably reaches the monitor on silicon.

## 4e. SQLite RUNS ON THE BOARD — the basic workload completes, ~77% of the time (2026-08-14)

**This supersedes §4c's "the domain does not complete", which was written 2026-07-29 and is kept
above only as history.** SQLite executes in a pure-capability domain on the FPGA and returns
correct results.

Bitstream `caplifive_12august.bit`. Binary `G6.dom` (sha256 `f93a9188a9a4433c…`), **not rebuilt
between boots**, verified present in the initramfs by hashing the cpio members.

**Both S-06 workarounds are ON, confirmed from the binary rather than from a build log** (the log
did not survive a `/tmp` reset, and the manifest records names, not flags):

* `BEEBS_LDC_HIGH_HALF_FIXUP` — `memcpy` contains the fixup's source shape verbatim: the plain
  two-half copy loop (`sd zero` init, `bltu a1, a0` against 1, `slli 3` index, plain `sd`), then
  `ldc` of the granule, `lcc … 0x1`, compare against literal `7`, and a conditional `stc`. That is
  `BEEBS_CHUNK_COPY`'s guarded arm and nothing else produces it.
* `-capstone-guard-cap-granule-copies` — 511 `lcc` instructions in the image, of which exactly 1 is
  in `memcpy` (the fixup's). The other 510 are the compiler pass.

See `ref/S06-WORKAROUNDS-TO-REVERT.md`; the revert acceptance gates depend on which of these was
live, which is why it is established here from the artifact.

**Basic workload** = CREATE TABLE / three INSERTs / SELECT returning all three rows / finalize.
When it completes it returns `obs=0x5A6E0603` and prints `alpha 11`, `beta 22`, `gamma 33` — the
correct rows, byte-identical across every passing run.

| | genuine executions | completed | wedged |
|---|---|---|---|
| measured 2026-08-14, three boots + prior record | 13 | 10 | 3 |

**Completion rate ≈ 10/13 ≈ 77%.** Method: each boot ran a control domain followed by eight
repetitions of the same binary; all three controls passed, so no boot is void. Two R-16 entry
stalls are excluded from both numerator and denominator, an image that never entered being no
evidence about the code in it. Each boot stops at its first failure, so these are censored
run-lengths rather than 24 independent trials.

**The failures are one silicon defect, S-07, at one instruction.** All three wedges landed at
`output_text+0xdc` with mcause 25 (UNEXPECTED_OPERAND — a capability read back from memory arrives
untagged), from two different physical placements of the domain. `output_text` is our own domain
harness (`sqlite_boundary_cost_domain.c:48`), not SQLite: the fault is in the code that writes
result rows out through the shared region, not in the database engine.

**The full (extended) workload does not complete** — it wedges in `sqlite3DbMallocRawNN`, also
S-07, also mcause 25.

### What may and may not be claimed from this

- **May**: SQLite runs in a Capstone pure-capability domain on real silicon and produces correct
  results; the remaining failures are an identified hardware defect with a reproducer package
  (`tests/fpga-repros/S07-capability-untagged-on-reload/`), not an unknown.
- **May NOT**: that SQLite runs *reliably* on this silicon, or any timing number taken from these
  runs — cycle counts were not the object of this measurement and the S-06 workarounds add
  +33660 bytes of `.text` plus a branch per granule, so any performance figure from this binary
  measures the workaround as much as the workload.
- **Framing is the project lead's call.** The honest statements are "the basic workload completes,
  with p(failure) ≈ 23% per run, from one identified silicon defect" and "SQLite does not yet run
  reliably on silicon". Both are true; which belongs in the paper is not a lane's decision.

Full trail: `history/14-08-2026_18-30-00_s07-wedge-rate-and-fault-site.md`.

## 5. What is NOT established — read before citing anything above

- **PARTLY SUPERSEDED 2026-07-28 — read the correction first.** What follows was written when
  four rungs were unmeasured and all four fit the register-indexed-load shape. Since then
  **R-6 (`beebs_janne`) and R-9 (`beebs_ns`) hang although neither writes the object it
  indexes**, so R-1's shape is absent and cannot be the explanation for them. R-1 remains
  well-supported for the rungs it does describe, and the reproducer below is unaffected. But
  **"the mechanism is now known" is too strong as a blanket statement about non-measured
  rungs**, and the paper's corresponding paragraph was reworded on 2026-07-28. Treat the
  paragraph below as "the mechanism for the R-1 class", not "the mechanism, full stop".
- **THE MECHANISM FOR THE R-1 CLASS (2026-07-27).** Those rungs fail because of a
  characterised **hardware** fault: *a load whose address arrives through a register — a
  register-carried capability or a register-computed offset — does not observe pending stores to
  other addresses.* Isolated by a minimal failing case with controls on both sides: a register
  index alone is correct, a second store alone is correct, together they fail; store ordering and
  index arithmetic are irrelevant; it reproduces across boots. It is not loop-specific (a single
  such load returned 0 where 5 had just been stored). **Seven mitigations were tried and all
  failed** — fence before the load, fence after every store, register hoisting, making the other
  store register-indexed, 64 B cache-line separation, constant-offset pointer walk, both accesses
  via pointers — so there is **no general software workaround**: a dynamic array index cannot
  have a compile-time-constant base. QEMU executes every probe correctly.
  This explains the 3-pass/4-fail split exactly, including `rv8_primes`, whose passing had
  refuted several earlier theories: its inner loop touches one location per iteration, so a
  second store is never pending, while `matmult_int` and `coremark_matrix` do
  `C[i*N+j] += A[…]*B[…]` — register-indexed loads plus a store elsewhere.
  **For the paper this converts "an unexplained divergence" into "a documented hardware
  limitation", which is a citable claim.** Trail:
  `history/27-07-2026_17-05-00_RESULTS-culprit-found-register-indexed-load-misses-pending-stores.md`.
- **STALE COUNT — now eight measured, not three (2026-07-28).** The paragraph below dates from
  when three rows existed and predicted it would "stay three". It did not: the bare-metal
  baseline (I-2) plus rungs `cover`, `aha_mont64`, `bs`, `cnt`, `recursion` took it to eight.
  Kept for provenance because its per-rung failure detail is still accurate and still the best
  record of what was tried. `beebs_crc32` and
  `beebs_insertsort` were made *buildable* at −O1/−O2 on 2026-07-27 and are QEMU-correct, but
  when measured on the board **both failed**: crc32 hangs, insertsort returns a wrong value with
  only 560 retired instructions (the compute never ran). Both were already wrong on silicon at
  −O0, so this is the same divergence, not a new fault. `matmult_int` and
  `coremark_matrix` produce no
  result at ANY reachable configuration: they transfer cleanly, then the `cscall`
  hangs (matmult at −O1 and −O2; coremark at −Os and at −O0 with a 32 KiB window).
  It is not `-Os` codegen (coremark hangs at −O0), not code size (coremark hangs at
  1,988 B of text, smaller than rungs that pass), and no instruction is present in
  every hanging build and absent from every passing one. Global count and `.bss` size
  do not discriminate either (`rv8_primes` has the largest `.bss`, 12,512 B, and
  passes; every hanging rung is under 800 B).
  **The hang is INSIDE THE COMPUTE, not at domain entry** — established 2026-07-26 by
  `LADDER_INSTR_MODE=7`, which runs the whole entry path but branches over the compute:
  both hanging rungs then complete a full domain round-trip on silicon, first attempt.
  An earlier version of this document called it "a domain-entry fault"; that was
  inferred from the failure of three compiler-side hypotheses, which does not localize a
  layer, and it is **retracted**. Mechanism inside the compute is still unknown; the
  leading hypothesis is the known miscompute corrupting a **loop bound** rather than a
  checksum (`matmult_int` miscomputes at −O0 and hangs at −O1). Trail:
  `history/26-07-2026_23-56-07_the-hang-is-in-the-compute-not-at-domain-entry.md`.
  `beebs_crc32` and `beebs_insertsort` **no longer fail to build** (2026-07-27): the former was
  an optimizer/large-RO-delivery interaction, not a compiler bug (−O1+ constant-folds its
  runtime-generated CRC table into a 2048 B private constant the cap-table glue cannot deliver);
  the latter was CodeGenPrepare zero-extending a negative address offset into the 128-bit pointer
  carrier, plus an i128 ISel gap. Both build and pass the QEMU parity leg at −O0/−O1/−O2, so the
  measured set nevertheless stays at **three** — see
  `history/27-07-2026_15-48-02_RESULTS-the-two-newly-buildable-rungs-fail-on-silicon-too.md`.
  So 3 pass / 4 fail, and the four failures are one family: each either hangs or returns a value
  whose instruction count proves the compute never ran. No compiler-side property separates the
  two groups. Trail:
  `history/27-07-2026_12-59-35_three-codegen-fixes-unblock-two-ladder-rungs-and-rv8-at-O1.md`.
  **The loop-exit ("fragile `bne`") hypothesis is REFUTED on silicon (2026-07-27).** It
  was observed statically that `matmult_int` at −O1 emits 8 conditional branches, **all
  `bne`**, while the same source at −O0 emits 8, **all `blt`** — suggesting the hang and
  the miscompute were one fault whose symptom the branch kind selected (`bne` exits on
  exact equality and can be overshot; `blt` cannot). **Board test #65 killed it:** a −O1
  build with ordered exits forced (verified 0 fragile / 8 ordered, QEMU-correct through
  the same controller) **still hangs, identically.** The codegen split is real but is a
  correlate, not the mechanism. Do not repeat the "one fault, two symptoms" framing.
  **What #66 DID establish:** for `coremark_matrix`, the hang is inside
  **`core_init_matrix`**. Bisecting against mode 7 at the same −O0 @32 KiB config —
  entry-only RETURNS, entry + `core_init_matrix` HANGS, everything HANGS — narrows it
  from the whole benchmark to one ~40-line function. Two candidates remain inside it:
  the dimension loop `while (j < blksize) { j = i*i*2*4; }` (`bgeu` `0x10428` / `mulw`
  `0x10444`), and the N×N seeding loop doing `seed = ((order*seed) % 65536)` per element
  through the gp-delivered block cap. Not yet separated.
  **RESOLVED for coremark_matrix's FIRST fault (2026-07-27, board #67a-#67f): a `delin`
  executed in domain code wedges the RTL.** Bisected one instruction at a time, every build
  QEMU-correct through the identical controller: while-loop only RETURNS 9; **+ one `delin`
  HANGS**; the same image with `addi x0,x0,0` in the `delin`'s place (same position, same 4
  bytes, same register plumbing) **RETURNS 9**. Layout is controlled out -- mandatory, since
  4 added instructions previously flipped a passing rung. It is NOT "delin is unimplemented":
  the glue delins several caps in every domain and passing rungs work. The operand differs --
  the glue delins a cap *fresh from `split`*, domain code delins one *`ldc`-loaded from the
  cap-table*, which the glue already delin'd before `stc`, so on a type-preserving machine it
  is NONLIN->NONLIN -- exactly the case our QEMU fork was patched (`f4d416c265`) to treat as
  idempotent "rather than faulting". Caveat: instrumented QEMU reports that operand as LIN, so
  QEMU and the glue disagree about capability type after `stc`->`ldc`; which side is right is
  a board-owner question. **Removing the delin is safe but insufficient** -- the derivation it
  guarded works without it and QEMU still yields 14343, but the full rung still hangs, so there
  are **>=2 independent faults**; fault 2 lies in the seeding loop or later. `matmult_int`
  contains **no delin at all**, so this does not explain it. A minimal two-instruction silicon
  repro now exists. Trail:
  `history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`.
  Also corrected there: the "no discriminating instruction" sweep had been run with the
  Capstone-triple disassembler, which prints every M-extension op as `<unknown>`; re-run
  with `--triple=riscv64 --mattr=+m` the conclusion **stands** (`beebs_prime` passes
  with `mul`+`remu`), and the blind spot was only 2% of instructions, uniform across
  binaries. Trail:
  `history/27-07-2026_00-28-51_loop-exit-condition-splits-hang-from-miscompute.md`.
- **The pointer-chasing axis is missing entirely.** No measured kernel chases
  pointers, yet capabilities are 16 B against an 8-byte pointer, so linked structures
  double memory traffic — historically where capability machines hurt most. The set
  therefore likely **understates** overhead. Say so rather than implying coverage.
- **A passing rung is not stable ground — shown by controlled A/B.** Two builds of
  the same rung differing only in `domain_main`: *with* the minstret instrumentation
  it returns 1087631800 (wrong, and deterministic across two sessions); *without* it,
  582955588 = the oracle. Four instructions, none inside the computation. So the
  earlier "scalar rungs pass, array rungs fail" split is too strong — the scalar
  rungs passed *for that exact codegen*, not because scalar code is immune.
- **The domain uses the gp-captable *silicon workaround* config** (shrink off,
  `-fno-jump-tables`, gp cap-table), chosen for this RTL's constraints, not what a
  tuned Capstone ABI would emit. §2's figure plausibly **overstates** pervasive
  spatial safety. Do not present it as canonical.
- **Reproducibility: 0.05% on one rung.** Un-instrumented `beebs_prime` measured in
  two independent sessions a day apart, with a full power-cycle and firmware reload
  between: 47,804 (25-07) vs 47,780 (26-07), **-24 cycles / -0.05%**. That is one
  repeat, not a distribution, but the vehicle is stable.
- **Otherwise no error bars.** One measurement per condition. Interrupts land inside the
  bracket and cost ~16,000 cycles when they do, so **any benchmark under ~100k
  cycles is unreliable in a single pass** — that is 5 of the 7 rungs. The fix is
  to scale the kernels' iteration counts so each runs ≥1M cycles.
- **`coremark_matrix` at 56k cycles is a micro-slice, not CoreMark.** Real CoreMark
  runs ≥10 s precisely to swamp this noise. Do not call it CoreMark without scaling.
- **A region word gets silently corrupted.** `rv8_primes` returned the *correct*
  result while a word of its shared region held a stray DRAM address. The passing
  rungs were clean only where anyone looked.

---

## 6. Already in the draft, unchanged

- CHERI comparison (`tab:perftree`): CHERI-RISC-V purecap spatial 10,095 instr;
  async 19,281 (1.9×); eager 16.8 M (1,661×); ours +5 instructions, O(1).
  QEMU dynamic-instruction proxy, both vehicles.
- Compatibility: SQLite (open, CREATE/INSERT/SELECT, transactions, secondary index,
  prepared statements, UPDATE/DELETE, aggregates, sorter, JOIN, GROUP BY, string
  functions) plus CoreMark, the RV8 suite and 82 BEEBS kernels execute as
  pure-capability domains returning correct results. **QEMU-backed**, not silicon.

## §4d — R-12 CONFIRMED ON SILICON: the revocation-node pool wraps under SQLite (2026-07-31)

First hardware observation of the rev-node allocator overflow. Until now R-12 was predicted
from the RTL and never measured.

Method: run the SQLite domain on the board until it wedges, then read the allocator state
off the debug-LED mux WITHOUT resetting (a reset clears it). The mux is selected by the
board switches -- `debug_byte_sel = switches[7:5]`, `debug_reg_sel = switches[4:0]`
(cva6.sv:874-877); the rev-node registers are direct 8-bit assignments in the `debug_reg_sel`
case (cva6.sv:1184-1186), so only the low five switches matter.

    rev_node_head[7:0]                 switches 249  ->  0x4a  = 74
    {overflow, 5'b0, head[9:8]}        switches 250  ->  0x80  -> overflow = 1, head[9:8] = 0
    rev_node_serving_idx[7:0]          switches 251  ->  0x00
    stall flags                        switches 225  ->  0x84  -> stall_issue = 1

**overflow = 1, and head = 74.** The allocator's head is 10 bits initialised to 3
(capstone_rev_node.anvil:160,168) and bumped once per `split`/`mrev`. A final head of 74
after a wrap means 1095 allocations: SQLite's 1059 per-global carves, the cap-table split,
and ~35 for the monitor's create_domain / two create_regions / two map_regions / share. The
usable pool is 1021 (1024 minus the head's initial 3), so it wrapped by 74 and reused live
node ids.

Consequence, and why the failure has no trap: every `stc` blocks on a revocation-node query
with no timeout (capstone_dyn_unit.anvil:395-404, `recv rev_node_ep.query_res` with no abort
path), and id reuse can splice a node into the `next` chain twice. REVOKE_NODE
(capstone_rev_node.anvil:13-32) has no visit bound and no cycle detection, so it walks
forever, never re-enters IDLE_STAGE, and no later query is answered. The core stalls at
issue rather than faulting -- consistent with `stall_issue = 1` and a non-advancing
`serving_idx`, and with the board showing SHA5 then silence.

The overflow flag reaches only a debug LED: there is no CSR, no interrupt, and no watchdog
anywhere in the design (grep of core/ and corev_apu/ finds only testbench timeouts). So on
this silicon the condition is invisible to software by construction.

Provenance: the domain measured here is byte-identical to the one that passes end-to-end
under QEMU in the silicon config (__CAPSTONE_SQLITE_SILICON_PASSED__), hash-verified on the
board before execution (sqlite_silicon.dom 8e1cb920..., sqlite_host.user 052286d2...). No
diagnostic clamp and no feature trim. This is a silicon-only divergence.

Scope: this bounds any domain to ~1021 capability carves, i.e. ~1021 globals with one
capability per object. It is a property of the prototype board, not of the Capstone design.

---

## §4f — SQLite passes SQLLogicTest in a capability domain: 10,807 records, ZERO divergences (QEMU, 2026-08-21)

**READ THE SCOPE FIRST: this is QEMU, not silicon.** The board attempt is §4g below. Nothing
here is a silicon result and none of it should be cited as one.

### What was measured, and why it is a DIFFERENCE rather than a rate

`benchmarks/sqlite/slt/slt_runner.h` is a SQLLogicTest runner that compiles **unchanged** for
the host and for a capability domain, and `slt_native.c` links it against the **same** SQLite
3.53.3 amalgamation with the **same** semantic build configuration. The result is the
difference between the two sides.

That design is not tidiness. An absolute pass rate is contaminated by corpus-versus-engine
artifacts with nothing to do with capabilities, and this corpus contains several — one
evidence file produces eleven on any machine. With one runner they appear identically on both
sides and cancel. **It also caught a live instance during bring-up:** the domain build carries
`-DSQLITE_OMIT_FLOATING_POINT=1`, under which a decimal point is a syntax error, and the first
domain run failed five statements on `VALUES(3,'ccc',1.5)`. An ordinary configuration
difference was presenting as a capability defect.

### Result — every field of every summary EQUAL between domain and native

| file | records | queries | verdict |
|---|---|---|---|
| `negative-control.test` | 21 | 10 | identical, **including 2 stmt / 4 query deliberate failures, 2 skips, 1 parse error** |
| `select1.test` | 1031 | 1000 | identical, 0 failures |
| `select2.test` | 1031 | 1000 | identical, 0 failures |
| `select3.test` | 3351 | 3320 | identical, 0 failures |
| `select4.test` | 3857 | 2617 | identical, 0 failures, 215 skipped for size |
| `select5.test` | 1436 | 732 | identical, 0 failures (needs a 2 MiB arena — see below) |
| `evidence/slt_lang_aggfunc.test` | 80 | 67 | identical, including 11 shared corpus artifacts |
| **total** | **10,807** | **8,746** | **zero divergences** |

**7,393 of the query records state their expectation as an MD5 of the entire result set**, so
this is agreement over hashed full result sets, not over scalars.

**The negative-control row is the load-bearing one.** Six of its arms are wrong on purpose and
two more must be skipped rather than passed; the domain reproduces every one. Without it,
"zero failures" everywhere else would be equally consistent with a comparator that cannot fail.

### What this licenses, and what it does not

**Supported:** *"SQLite 3.53.3 executes 10,807 SQLLogicTest records inside a pure-capability
domain and produces results identical to the same SQLite built natively with the same
configuration, including MD5 hashes of full result sets, with zero divergences."*

**NOT supported — do not let any of these be dropped:**
* **QEMU, not silicon.**
* **A subset** — 7 files of 622.
* **17 features omitted**, floating point among them, so the R column type is never compared.
* **Result sets above 4096 values are not compared** — 215 records, all rowsort.
* **Arena-dependent:** `select1`/`select2`/the negative control fit the silicon 256 KiB arena;
  `select4` needs 1 MiB and `select5` 2 MiB. Below that they report a clean `oom` bucket, which
  is counted separately and never as a pass.

### One defect found and NOT attributed

At a 1 MiB arena `select5` faults on a `cincoffset` with an untagged operand at image VA
`0x644d4` inside **`sqlite3VdbeExec`** — SQLite's own interpreter, not the runner. The operand
is an all-zero word (`val=0x0`) loaded by the preceding `ldc`, which **excludes** both the S-07
tag-strip family and the linear move-out family. A poisoned-arena arm returned `0x0` rather
than `0xa5…`, so something wrote the zero rather than leaving it uninitialised. Whether an
allocation failed or a stray plain store zeroed a live capability slot is **open**; an earlier
"allocation failure" root cause was recorded and retracted the same day after audit. Full trail
in `plans/sqlite-regression-suite-proposal.md`.

---

## §4g — THE ALLOCATOR MATRIX: what each of SQLite's allocators costs under capabilities (QEMU, 2026-09-11)

**These are QEMU `-icount` instruction counts, NOT silicon cycles.** They are in this document
because they are the first numbers that exist for the question at all, and because one of them
cross-checks the silicon record — but the silicon run is still owed, and CPI is the term that
separates them (native CPI for `main` is 3.769 in §7k, so cycles and instructions part company badly
on hardware). Read the CONFIGURATION block before placing any of these beside a §7 row.

`main --size 1`, one image per cell, capability arms through `run-speedtest1-measure.sh` and native
arms as the matched `speedtest1_baseline` **warm** pass in the same emulator under the same
`-icount shift=0`.

| | native | capability | ABI ratio |
|---|---:|---:|---:|
| memsys5 alone | 539,412,660 | 685,213,794 | **1.2703** |
| lookaside over memsys5 | 529,112,120 | 670,803,165 | **1.2678** |
| Sublet (both allocators) | — *empty by construction* | 698,224,832 | — |

The sixth cell cannot exist: every primitive in `sublet.h` is opcode `0x5b`, which does not exist on
rv64imac. A revocable sub-lease has no meaning without capabilities.

**What the matrix says.**

- **The capability ABI costs ~1.27x and that is essentially allocator-INDEPENDENT** — 1.2703 against
  1.2678, a difference of 0.0025. Whatever the ABI is charging for, it does not scale with the
  allocation rate at this size. Nobody had measured this; it was reasonable to expect the overhead
  to track allocator traffic, and it does not.
- **Lookaside is ~2 % FASTER on both targets** (−1.91 % native, −2.10 % capability), which is the
  first cycle-level evidence either way. The whole recorded silicon corpus was taken with it OFF,
  by accident of a text harvest — so every §7 figure sits on the slightly slower side of this.
- **Sublet costs +4.09 % against the arm it actually replaces** (capability + lookaside, same
  target, same ABI). That is the price of temporal safety in the allocator: a `revoke` per free, an
  `mrev`+`delin` per hand-out, and a write-through of the block on reclaim.
- Composed end to end, Sublet against the unprotected native build is **+31.96 %** — but that
  bundles the ABI, the allocator rewrite and the revocation traffic, and the decomposition above is
  what separates them.

**CROSS-CHECK, and it is the reason to trust the apparatus rather than just the arithmetic.** The
measured ABI ratio for `memsys5`, 1.2703, agrees with §7k's independently recorded
*"instr ratio (predicted)"* for `main` — **1.270** — to **0.02 %**. That figure was derived on
silicon, from different arms, on a different day, by a different route.

**What these numbers are not.** Instruction counts, not cycles. Size 1 only. The Sublet cell reports
`split=8484 mrev=41115 delin=32638 revoke=41115 init=8477` — `init` non-zero beside a non-zero
`revoke`, so the reclaim genuinely executed rather than being skipped; on the currently flashed
silicon it would read `init = 0` and the cost would be understated by the whole write-through
(R-30/R-31). **A silicon Sublet number is not available until that bitstream is flashed**, and the
rev-node budget pins that cell to `--size 1` regardless: 43,407 nodes measured against 65,532
available, 66 % used.


### §4g.1 — THE FIVE IMAGES ARE BUILT AND THE CAPABILITY ARMS ARE QEMU-VALIDATED (2026-09-12, `--size 1`)

Staged so that the moment the R-30/R-31 flash lands, the board work is three boots and no builds.
Every arm below is `--testset main --size 1 --verify`, `-icount shift=0`.

| cell | | allocator | lookaside | image sha256 (16) | `HEAP` | QEMU cycles |
|---|---|---|---|---:|---:|---:|
| ① | native | memsys5 | off | `f4cf7caed144d952` | 2,097,152 | 545,623,496 |
| ② | native | memsys5 + lookaside | **on** | `24cb59fa7dbfb8fb` | 2,097,152 | 535,335,376 |
| ④ | domain | memsys5 | off | `2f4e6b73b85b569e` | 2,097,152 | 692,983,497 |
| ⑤ | domain | memsys5 + lookaside | **on** | `ccb73bc08db39990` | 2,097,152 | 678,572,868 |
| ⑥ | domain | Sublet (both, + discipline) | **on** | `ceeded2533a74bce` | **910,008** | 690,505,703 |
| ⑥′ | domain | Sublet, lookaside forced off — **control** | off | `5f5045dbe075d458` | **910,008** | 705,994,514 |
| ⑤ᴳ | domain | ⑤ rebuilt at ⑥'s heap — **does not complete** | on | `7cc434e807570136` | 910,008 | fault, cause 24 |

**READ THE `HEAP` COLUMN BEFORE ANY RATIO ACROSS IT.** ④/⑤ share a geometry and may be compared
directly; ⑥/⑥′ share one and may be compared directly. **⑥ against ⑤ does not**, and the ⑤ᴳ row is
the evidence that it cannot be made to.

**Post-#14 re-measurement of the pair (2.1, 2026-09-14).** ④ rebuilt on the #14 (S-14 spill fix)
toolchain with the recovered recipe — `run-speedtest1-measure.sh` defaults (memsys5, lookaside
`0,0`), `SPEEDTEST1_HEAP=2097152`, and `SPEEDTEST1_STACK=385024` so it loads under the #3 module's
one-region rule (declares 2,701,904 = the same 2,316,880 of data + the smaller stack figure; LOAD
headers identical to the archived ④; the knob is diagnostics-only) — image `6cf8edf637f72063`:
`112006 38bb59fd`, `HEAP 2097152 DROPPED 0 RC 0`, **692,983,497 — identical to the archived ④ to
the instruction** (pre-registered: identical within icount jitter; the 17 changed instructions all
sit in `__capstone_cap_init`, which runs once, and 13 of them are 1:1 width swaps). The ① environment
control — the ARCHIVED native binary `f4cf7caed144d952`, same arguments, today's rootfs, module and
QEMU pin — hashes to the oracle and reads `BASELINE-WARM CYCLES 545,617,503`, **5,993 below the
archived 545,623,496 (1.1e-5)**; the inner `SPEEDTEST1-CYCLES` moved by the same 5,993, so the
difference is inside the run (kernel activity under a native process), not in the wrapper. The
pre-registration said ~1e-7, which is §7l's size-20 two-run figure (1,847 instructions absolute on
14.15 G); on a 545 M count the same absolute noise is 1e-5, and the domain arm — which runs under
the monitor with no kernel interleaving — reproduces to zero. Both figures are far inside the 0.17 pp
the matrix ratios are quoted to. Reconciled while at it: the ① row's 545,623,496 is the outer
`BASELINE-WARM CYCLES` of the archived run and the 545,609,572 in the same log is speedtest1's inner
`SPEEDTEST1-CYCLES` (13,924 apart, the warm wrapper's own work); one run, two counters.

**The #3-module rows (2026-09-14, QEMU `-icount`, the same rootfs and pin as the post-#14 re-measurement above).**
Under the #3 module the region allocator rounds a non-representable request up (R-33), so the Sublet
cells' 1,419,584-byte arena is created at 1,421,312 and their `HEAP` reads **911104**, not 910,008;
the tables region (1,750,285) is rounded to 1,751,040 the same way. The archived images, byte for
byte, on this module:

| cell | image | `HEAP` | QEMU cycles, #3 module | archived (pre-rounding) | Δ |
|---|---|---:|---:|---:|---:|
| ⑥ Sublet, lookaside on | `ceeded2533a74bce` | 911,104 | **690,051,663** | 690,505,703 | −0.066 % |
| ⑥′ Sublet, lookaside off (control) | `5f5045dbe075d458` | 911,104 | **705,997,303** | 705,994,514 | +0.0004 % |
| ⑤ memsys5, lookaside on, stack declaration 385,024 | `e6ee5255c896aa21` | 2,097,152 | **678,572,868** | 678,572,868 | 0 |
| ② native, lookaside on (rebuilt, same recipe) | `d95dd98c0de73c68` | 2,097,152 | **535,283,834** | 535,335,376 | −0.0096 % |

⑥'s `sublet:` counters moved with the geometry — `split=5481 mrev=37874 delin=32565 revoke=37874
init=5309` against the archived 5508/37899/32565/37899/5334 — and ⑥′'s are 8610/41243/32638/41243/8605.
**The measure script rebuilt ⑥ and ⑥′ byte-identical to the archived images on the #14 toolchain**
(the S-14 spill fix changes nothing in these programs, as it changed nothing in ④), which is what
makes their QEMU pass records legitimate for a board boot. ⑤ needed only the diagnostics-only stack
declaration to load under the one-region rule and counts the archived value to the instruction. The
lookaside instrument, re-proven on these binaries in separate `--stats` runs: ⑥ `Successful
lookasides: 25010`, the rebuilt ② `25122` (the archived ②'s value). ⑥/⑤ on this module is
**1.0169** (was 1.0176) — still a two-geometry comparison, labelled "configuration" as before.

*The lookaside-ON arms of sw64's image family (for the size-20 pair, Boot D):* domain
`90ef29431abdbdee` (sw64's recipe + `SQLITE_DEFAULT_LOOKASIDE=1200,40`, host 2af56927): size 1
682,301,888 at `111130 1e792c9d`, size 20 **17,544,109,561** at `3807866 2738af78`, `HEAP 134217728
DROPPED 0 RC 0`, both with pass records; native `78e523cb05cbb91f` (sw64's baseline recipe +
lookaside): size 1 538,409,766, size 20 **14,003,848,190**, same oracles. Instruction ratio at size
20 with lookaside ON on both arms: **1.2528** (OFF/OFF was 1.2547, §7l); lookaside saves 1.04 % of
native instructions and 1.19 % of the domain's at this size.

The native rows are `speedtest1_baseline warm` — the **warm** subcommand, which the source names as
the denominator, not `run`. Their `sqlite_heap` is 2,097,152 bytes, read with `llvm-nm -S` and equal
to ④/⑤'s `HEAP`, so the two ABI ratios below are geometry-matched.

**THE LOOKASIDE INSTRUMENT IS PROVEN IN BOTH DIRECTIONS, on the same binaries, in a separate
`--stats` run kept apart from the measurement so the extra output cannot perturb it:**

    ② Successful lookasides: 25122      <- fires
    ① Successful lookasides: 0          <- and returns zero when it should

That is the positive *and* the negative control the CLEAN-result rule asks for, and it is what
licenses calling ② a lookaside arm at all — the hash differing from ① proves only that the define
reached the build, never that the pool is enabled at run time.

#### The capability-ABI cost is ~27 %, and it is very nearly the same under both allocators

| ratio | | value |
|---|---|---:|
| ④/① | memsys5, domain over native | **1.2701** |
| ⑤/② | lookaside, domain over native | **1.2676** |

**That the two agree to 0.25 pp is the result, not a null.** The plan asked whether the ABI cost
differs by allocator; on this evidence it barely does, so the capability overhead and the allocator
choice compose rather than interact. Under `-icount` these counts are deterministic, so 0.25 pp is a
real difference and not run-to-run spread — it is simply a small one. The PRECISION bands below
(0.027 pp / 0.171 pp) govern *board* readings and do not apply to these.

Both are instruction counts, not board cycles. The silicon ratio will differ — this core's measured
CPI spans 1.13 to 6.44 — and §7k's board pair ratio of 1.220 is the quantity to compare against once
the flash lands, not these.

**⑤ CARRIES ITS OWN POSITIVE CONTROL, and it validates a script change as well as the arm.** Its
image is hash-identical to `ccb73bc08db39990`, the image independently verified by *running* it —
`Successful lookasides: 25010`. That matters twice over: it confirms ⑤ is genuinely a lookaside arm,
and it confirms the new `SQLITE_LOOKASIDE` route through this runner reproduces the old hand-set
`DOMAIN_EXTRA_DEFS` image exactly. ④'s differing hash shows the define changes the build at all.

**A DEFECT FOUND AND FIXED IN THE SIXTH CELL: the runner would have built it lookaside-OFF while its
own comment said ON.** Stated as latent rather than observed: the only prior ⑥ on record, the 43,407
sweep, was lookaside-ON by the counter identity below, so something outside this runner — a hand-set
`DOMAIN_EXTRA_DEFS` in the operator's shell — supplied it that time. The defect is that the runner
did not, so the next person to run the documented invocation would have got the silent mismatch. Nothing on the `SPEEDTEST1_SUBLET` path set `SQLITE_DEFAULT_LOOKASIDE`, so ⑥ would have
been compared against a lookaside-ON ⑤ — making the one ratio the cell exists to produce, the cost
of the revocation discipline, a **two-variable** comparison, silently. The Sublet patch ports the
lookaside *code*; that is not the same as the pool being *enabled*, and at `0,0` the ported code is
compiled in and never used. Fixed in the runner and negative-tested by ⑥′ below.

**~~THE COST OF THE DISCIPLINE~~ — RETRACTED THE SAME DAY IT WAS WRITTEN, 2026-09-12. ⑥/⑤ = 1.0176
IS NOT THE COST OF THE DISCIPLINE, because the two arms do not share a geometry.** ⑤ runs
`HEAP 2,097,152`; ⑥ reports `HEAP 910,008`, since the Sublet port hands `CONFIG_HEAP` the *tables*
region rather than dom_data. A 2.3× difference in the arena memsys5 sees changes split and
fragmentation behaviour, so 1.76 % is **discipline + geometry** — the same two-variable error fixed
for lookaside two paragraphs above, one level down, and missed because the reading was taken with a
`grep -oE "SPEEDTEST1-CYCLES [0-9]+"` that cut the line before its `HEAP` field.

**THE GEOMETRY CANNOT BE EQUALISED WITH THE HEAP KNOB, AND THAT WAS TESTED RATHER THAN ASSUMED.** ⑤
rebuilt at `SPEEDTEST1_HEAP=910008` (image `7cc434e807570136`) does not complete: it enters and then
halts, `cause = 24, pc = 0x101c23394, badaddr = 0x101570000`, after `SQ: G/enter`. memsys5 alone
needs the documented ≥1.5 MiB for `main --size 1`; ⑥ clears the same workload at an effective 910,008
because lookaside absorbs the small allocations and the Sublet arena is carved differently.

Note this is **not** ⑥ using less memory. ⑥ holds arena 1,419,584 + tables 1,750,285 ≈ 3.17 MB of
host-shared regions against ⑤'s 2,097,152 of dom_data; `HEAP` names only the part memsys5 sees. The
two arms place their allocator arena in different kinds of memory, which is why no single knob
equalises them.

**Heap sweep, 2026-09-14 (QEMU, #14 toolchain, #3 module; lookaside `1200,40`, memsys5 static heap,
stack declaration 385,024 so the image loads under the one-region rule — diagnostics-only, the loaded
bytes do not change):** `SPEEDTEST1_HEAP` 910,008 (`f8a92251ec139724`, the ⑤ᴳ control) FAULTS;
1,048,576 (`8bde09363d2dcf03`) FAULTS; 1,310,720 (`6ca730db4883462a`) FAULTS; **1,572,864
(`e89ad882a93b9e01`) COMPLETES** at the off/off oracle `112006 38bb59fd`, `HEAP 1572864 DROPPED 0
RC 0`, **678,529,384 cycles** — 0.006 % below ⑤'s 678,572,868 at 2 MiB. So memsys5's minimum for
`main --size 1` in the domain is in (1.25, 1.5] MiB, where the native measurement put it (1.5 MiB,
`run-speedtest1-measure.sh` header), and ⑥'s 910,008 cannot be reached by the knob. The three faults
are one fault: `cause = 24, pc = 0x101c23394, badaddr = 0x101570000` at every size, and that pc is
`capstone_stdio_on_exit+0x6c` — the exit handler's DELIBERATE `ldc` through an integer
(`speedtest1_measure.c:340-341`, "rs1 holds an integer, so the capability load raises
UNEXPECTED_CAP_TYPE"), taken after `fatal_error` on memsys5 exhaustion writes the report. **⑤ᴳ's
"fault" is the abort path by design, not a capability defect and not an S-14 case**; the post-#14
rebuild reproduces it exactly, as predicted. ⑥/⑤ stays a two-variable comparison.

**What ⑥/⑤ = 1.0176 may be quoted as:** the end-to-end cost of the Sublet *configuration* against the
lookaside configuration at each one's own working geometry. Not as the cost of the revocation
discipline in isolation. Isolating that needs an arm differing from ⑥ in the discipline alone, which
this pair is not.

Two further caveats apply to it either way. These are `-icount` instruction counts and not board cycles, so the silicon figure will
differ — the reclaim is O(bytes), one store per 16 bytes, and that ratio is CPI-sensitive. And this
number exists on QEMU *because* QEMU implements REVOKE to spec: on pre-flash silicon R-31 returns
LINEAR, the port branches past both the write-through and the `init`, and the discipline's whole
reclaim is skipped. So **1.76 % is the prediction for what the post-flash board should show**, not a
figure the current board could reproduce.

#### The lookaside pool cuts revocation-node pressure by an eighth — a new result, from the control

|  | split | mrev | delin | revoke | init | **nodes** (`split+mrev`) | **of 65,532** |
|---|---:|---:|---:|---:|---:|---:|---:|
| ⑥ lookaside **on** | 5,508 | 37,899 | 32,565 | 37,899 | 5,334 | **43,407** | 66.2 % |
| ⑥′ lookaside **off** | 8,484 | 41,115 | 32,638 | 41,115 | 8,477 | **49,599** | 75.7 % |

Turning the pool off costs **6,192 extra revocation nodes, +14.3 %**, because lookaside absorbs the
small allocations that would otherwise each become their own sub-lease. Against a budget that
*deliberately stalls the core* on exhaustion, that is a safety margin and not a curiosity: 33.8 % of
headroom against 24.3 %.

**IT ALSO RESOLVES WHICH CONFIGURATION THE RECORDED 43,407 BELONGED TO.** ⑥ reproduces the earlier
sweep's counters *exactly* — `split=5,508 mrev=37,899 delin=32,565 revoke=37,899 init=5,334`, every
field. Since ⑥′ shows lookaside-off gives materially different counters, the earlier sweep was
**lookaside-ON**, and the headroom figure carried forward is for the configuration cell ⑥ actually
runs in. That had been an open ambiguity — the note beside it read "my sweep ran the plain arm",
which refers to the `memhook` instrument, not to the pool.

**`init` is non-zero (5,334) on every Sublet arm here, and that is expected on QEMU rather than
evidence about the flash.** QEMU implements REVOKE to spec, so the reclaim runs. The masking
signature to watch for on silicon is the opposite: `init = 0` beside a non-zero `revoke`.

### §4g.2 — BOOT sw59: the first boot on `caplifive_r30r31_1bfff7776` (2026-09-12)

Control first, `--tail --arena` last. **Control PASSED** — `k800 retval=4, cycles=4521, instret=1089`
— so the boot carries a verdict. Driver rc=0, all four arms ran, board restored.

| arm | | image (sha256, 16) | board cycles | board instret |
|---|---|---|---:|---:|
| 1 | control `k800` | — | 4,521 | 1,089 |
| 2 | domain, `main --size 1 --verify` | **`2f4e6b73b85b569e`** = cell ④ | **2,639,069,185** | — |
| 3 | native baseline, same workload | **`f4cf7caed144d952`** = cell ① | **2,176,757,779** | 578,533,909 |
| 4 | `--tail --arena 1419584 --tables 1750285` | `2f4e6b73b85b569e` | — | — |

*(Host `b576fd27f4efe6c3` on every arm. The hashes were added 2026-09-12 after an audit found this
table cited its arms by LABEL only — the exact breach of the cite-by-hash rule that R-29 exists to
prevent, and here it was load-bearing: without them the record could not distinguish "the bridge ran
and failed its band" from "this was never the bridge".)*

> **⚠ RETRACTED 2026-09-12: "THE BRIDGE PAIR HOLDS" — THIS WAS NOT THE BRIDGE ARM.** The agreed
> protocol was §7k's images unchanged, one post-flash boot, ratio against 1.220 within the 0.171 pp
> cross-boot band. **sw59 ran different images on both arms** — `2f4e6b73b85b569e` /
> `f4cf7caed144d952` here against sw56's `49994ed31852` / `072595ff0866` — so 1.2124 against 1.220
> is not a bridge measurement at all, and the 0.76 pp gap (4.5× the band) is a build difference
> rather than a band failure. The headline asserted a re-tie the arm could not deliver, while the
> body correctly caveated the build change: the caveat was right and the headline overrode it.
> (Found by the bench lane's audit; the hashes that settle it are in the table above.)

**What sw59 actually shows.** Ratio **1.2124** on a *different domain build* (this one carries the
lookaside fix), across a bitstream change — a term neither precision regime was ever measured over.
That is within a percent of the old build's relationship on the old bitstream, which is **a good
sign and not a re-tie**: it says the capability/native relationship did not move materially, and it
does not license carrying §7f–§7k forward. CPI is 3.81 on the domain arm and 3.76 on the baseline,
both inside the 1.13–6.44 band, so both arms did the work.

**THE BRIDGE IS STILL OWED, and it is the cheapest arm on the list:** §7k's own images unchanged —
`speedtest1_seven.dom` (`49994ed31852`) against `speedtest1_baseline` (`072595ff0866`) — one boot,
ratio against 1.220 in the 0.171 pp band, both absolutes recorded. Note the baseline currently
staged in the overlay is `24cb59fa7dbfb8fb`, the lookaside build, **not** §7k's.

**One discrepancy, recorded rather than smoothed — and RESOLVED 2026-09-12:** the baseline's board
instret is 578,533,909 where this session's QEMU `-icount` gave 545,623,496 for the *same binary*,
6.0 % apart on a quantity that should be deterministic. **It is the §7k tick asymmetry and nothing
else.** §7i derives that the board baseline's instret carries the timer tick and so lowers every
ratio by ~6 %, predicting a ticked count **6.06 %** higher for `main`; sw59 reads
578,533,909 / 545,623,496 = **6.03 %** higher. They agree to 0.02 pp. Cross-referenced here so the
next reader does not reopen it. It does not touch the ratio above, which is cycles/cycles.

#### THE R-30/R-31 ARM DID NOT TEST R-30/R-31, AND THE REASON IS STRUCTURAL

Arm 4 ran clean to `SQ: H/return`, then:

    SQ: released tables rc=0
    SQ: released pool  rc=1
    RCLM:00000000        (on all four shares, unchanged)

**`rc=1` IS NOT A FAILURE.** `libcapstone.c:577` — `/* 1: revoked, the slot kept */`. The pool *was*
revoked; only the region slot was not popped. This lane first read it as a regression against the
pre-flash `rc=0`, which was wrong, and is corrected here rather than left standing.

**`RCLM = 0` IS NOT THE MASKING SIGNATURE EITHER — the pre-registered reading was wrong about this
arm.** The reclaim is guarded at `sbi_capstone.c:1309`, `if (cap_type(r) == 3 /* UNINIT */)`, and it
sits on the **SHARE** path: it fires when a share finds a handle that a *previous* revoke left
UNINIT. Arm 4's structure is create → share → enter → return → **revoke at teardown**, and nothing
shares afterwards. So the reclaim was **unreachable by construction**, and `RCLM = 0` is the expected
reading for this arm rather than evidence about the silicon.

**What this costs and what it buys.** It costs the boot's headline: R-30/R-31 remain unverified on
silicon. It buys the design of the arm that *would* verify them, which the "revoked, the slot kept"
return makes possible: **revoke a REV_BORROWED region, then SHARE IT AGAIN in the same run.** The
second share is what reaches `:1309` with an UNINIT handle. Any arm whose only revoke is at teardown
cannot test this, which retires the `--tail --arena` invocation as the R-30/R-31 instrument.

**The positive control did fire, and it is the reason this is diagnosable at all.** `RCLM` is emitted
on every share, so the four `RCLM:00000000` lines prove the reporting path is live and readable —
a zero here is a real zero, not silence. Without that the same output would have been indistinguishable
from a counter that cannot be read.

### §4g.3 — THE R-30/R-31 FIX IS VERIFIED IN THE RTL THAT IS ON THE BOARD (2026-09-12, simulation)

Run at the **flashed revision** `1bfff7776` in a detached worktree, and against `66c4e7517` — the
revision that was resident until this session — as a matched negative control. The two test binaries
are byte-identical across the pair (`sha` checked, `ebac051fd47d3eb4` / `65f35ea4359e2c48`); the only
difference is the RTL, and between those revisions `core/` differs in exactly two files,
`capstone_dyn_unit.anvil` (+27) and `capstone_flu_unit.anvil` (+39).

| test | `66c4e7517` (pre-fix) | `1bfff7776` (**flashed**) |
|---|---|---|
| `r30-fill-init` | **FAILED**, tohost=11, 506 cyc, 1 `Exception:` | **SUCCESS**, 506 cyc |
| `r31-revoke-cursor` | **FAILED**, tohost=11, 501 cyc | **SUCCESS**, 490 cyc |

**The gate discriminates, which is what makes the PASS mean anything.** A clean result is not
evidence until the check is known to fire, and here it fires on exactly the RTL that lacks the fix.
Every cycle count is ~500 against a `+time_out=2000000` that reports ~2,000,013 on a hang, so none of
these is a timeout masquerading as a pass.

**NECESSARY, NOT SUFFICIENT, and the limit is specific.** `r30-fill-init.S` fabricates its UNINIT
capability with the Custom3 debug ops — R-30's own registry entry calls that "a test working around
the defect rather than reporting it" — and neither test runs through the monitor's real share/revoke
path or inside a capability domain. So this establishes that **the fix is present and works in the
RTL now flashed**, and does not yet establish that the monitor's reclaim fires on silicon. The board
probe in §4g.4 is what closes that.

### §4g.4 — BOOT sw60: **R-31 IS FIXED ON SILICON. R-30 IS NOT.** (2026-09-12)

The first arm ever to reach the monitor's reclaim on hardware. Control passed
(`k800 retval=4, cycles=4517`), so the boot carries a verdict.

**Read the LAST block of the capture, not the first.** The console replays ~548 KB of prior boots on
connect, and sw59 was also a four-arm boot with the same arm labels, so `### TEST 1/4 START` occurs
four times in this log. Scoping to the first occurrence yields sw59's numbers exactly — 2,639,069,185
and 2,176,757,779 — which look entirely plausible as sw60's. They are not.

#### The probe arm, verbatim

    SQ: released pool rc=1        <- precondition: revoked, slot kept (structural, pool released first)
    SQ: RR/share-A
    RGID:00000014  AREV:00000001  <- region 20, REV_BORROWED
    SHA2:00000003                 <- cap_type(r) = 3 = UNINIT
    BASE:AC100000  ALEN:0015A940  <- the 1,419,584-byte arena
    RCLM:00000000
    RCSH:000006C0                 <- fill shortfall, 1,728 bytes
    BASE:AC100000                 <- then the designed while(1) halt

**R-31 IS FIXED, and this is the direct evidence.** `SHA2` is `cap_type(r)`, emitted at
`sbi_capstone.c:1276` *before* the reclaim guard. It reads **3 = UNINIT**. On the previous bitstream
REVOKE returned LINEAR (0) — that is R-31 — and the whole reclaim was skipped. **`RCPR` did not
fire**, so `cap_cursor == cap_base` as well: revoke returns an UNINIT capability with its cursor at
base, exactly as the spec requires. Both halves of R-31's contract now hold on hardware.

**R-30 IS NOT FIXED, and the shortfall is not the documented one.** The fill ran and stopped
**1,728 bytes** short of `end` — `0x6C0`, which is 108 granules of 16 bytes — out of an arena of
1,419,584. *The unit is read off the macro, not off the tag's name,* because the two ends disagree:
`C_RECLAIM_FILL` (`sbi_capstone.c:259`) sets `n = (end - base) >> 4`, a GRANULE count, and
`C_RECLAIM` then reuses that same register for its result — `lcc` field 2 (cursor), `lcc` field 4
(end), `sub` — so the value that reaches `RCSH` is `end - cursor` in BYTES. In granules 1,728 would
read as 27,648 bytes and every ratio below would be wrong by 16×. R-30's registry entry describes a **one-byte** shortfall: filling an UNINIT region leaves
the cursor AT `end` where INIT requires PAST it. 1,728 bytes is a different quantity and is **not
explained here**. Stated as a measurement, not a diagnosis.

**What this does and does not close.** It closes the question the flash was for on the R-31 side, on
silicon, through the monitor's real share/revoke path rather than a fabricated capability. It leaves
R-30 open and re-characterised: the defect survives the fix, at a scale ~1,728× the documented one.
The RTL simulation in §4g.3 passed `r30-fill-init` at this same revision — and that test fabricates
its UNINIT with Custom3 debug ops on a small buffer, which is precisely the kind of gap between a
directed test and the real path that this arm exists to expose.

#### The lookaside pair, banked on silicon

| | image | board cycles | board instret |
|---|---|---:|---:|
| ⑤ domain, lookaside | `ccb73bc08db39990` | 2,551,506,640 | — |
| ② native, lookaside | `24cb59fa7dbfb8fb` | 2,107,533,496 | 567,065,689 |

**⑤/② = 1.2107**, against **④/① = 1.2124** from sw59. The capability-ABI cost on silicon is
**~1.21, and the two allocators are INDISTINGUISHABLE AT THIS PRECISION** — which is a weaker and
more defensible claim than the one first written here.

> **⚠ CORRECTED 2026-09-12: this said "does not depend on the allocator".** The separation is
> **0.17 pp against a 0.171 pp cross-boot band — 0.99× the band**, so these two silicon numbers
> cannot resolve a difference at all, in either direction. "Indistinguishable at this precision" is
> what they support; "independent" is a claim about the world that they do not. (Bench lane's audit.)
>
> **And the stronger evidence points the other way, slightly.** The QEMU pair — 1.2701 vs 1.2676,
> 0.25 pp, deterministic to 1.3e-7 — *is* resolvable, and it shows a small but real dependence, with
> lookaside costing marginally less. So across both platforms the supportable summary is
> **"barely depends"**, not "does not depend". The silicon pair is consistent with that and simply
> too coarse to see it.

### §4g.5 — BOOT sw61: the Sublet cell on silicon, and INIT is NOT unreachable (2026-09-12)

Control passed (`k800 retval=4`). Cell ⑥, image `ceeded2533a74bce`:

    SPEEDTEST1-CYCLES 2797516229  HEAP 910008  RC 0
    sublet: split=5508 mrev=37899 delin=32565 revoke=37899 init=5334

**`init = 5334`, NON-ZERO ON SILICON — this lane predicted 0 and was wrong.** After sw60 showed the
monitor's reclaim failing with `RCSH` (fill short of `end`), the expectation written down was that
R-30 would block Sublet's `init` too and the counter would read 0, the masking signature. It does
not. Every counter is **bit-identical to the QEMU run** — `5508 / 37899 / 32565 / 37899 / 5334`,
every field — so the Sublet port performs 5,334 successful INITs on this silicon and the discipline
is fully exercised.

**THIS RE-OPENS R-30'S CHARACTERISATION.** The registry entry says INIT is *unreachable* — filling an
UNINIT region leaves the cursor AT `end` where INIT requires PAST it. On the same bitstream, in
adjacent boots: the monitor's reclaim of a 1,419,584-byte region falls 1,728 bytes short (sw60,
`RCSH:000006C0`), while the domain's own 5,334 INITs succeed. **INIT is plainly reachable.** Whatever
sw60's shortfall is, it is not "INIT can never be satisfied", and the two readings have to be
reconciled before R-30 is described either way. Stated as the conflict it is, not resolved here.

**What that counter cannot say.** `init` increments when the domain's INIT returns without faulting,
so 5,334 is evidence about REACHABILITY and nothing else — it does not attest to the cursor or the
bounds INIT produced. A defect of the shape "INIT succeeds but yields wrong bounds" would print
exactly this number, and bit-identity with QEMU does not separate them either: both sides count the
same non-faulting returns. So **INIT is reachable** is the claim, and it is the only one the
instrument carries; *INIT is correct* is not measured here and must not be written anywhere from
this row.

#### The Sublet CONFIGURATION costs 9.6 % on silicon against 1.8 % on QEMU — and that gap is the point

> **⚠ THIS HEADING SAID "the discipline" until 2026-09-12, and that was the already-retracted
> claim reappearing in the summary.** The QEMU 1.8 % was retracted as a *discipline* cost in
> `ea64117a18f7` because the arms differ in heap geometry (HEAP 910,008 against 2,097,152). **The
> silicon pair sw61/sw60 carries the identical mismatch**, and the body below has always said so —
> but the heading, and both state-doc rows, asserted the retracted form. The body was right and the
> summary overrode it, which is the same failure shape as the bridge headline above. **The 5.5×
> figure inherits the caveat:** it is a ratio of two comparisons each of which carries a geometry
> term, so it is suggestive of the O(bytes) reclaim rather than a measurement of it.

| | domain cycles | vs ⑤ |
|---|---:|---:|
| ⑤ lookaside (`ccb73bc08db39990`) | 2,551,506,640 | — |
| ⑥ Sublet (`ceeded2533a74bce`) | 2,797,516,229 | **1.0964** |

QEMU put the same pair at 1.0176. The silicon figure is **5.5× larger**, which is what the reclaim
being O(bytes) predicts: one store per 16 bytes is memory-bound work that `-icount` counts as one
instruction each and silicon pays cache and memory latency for. It is the clearest case in this
corpus of an instruction count understating a cost, and it is why §7 is silicon-only.

**The geometry caveat carries over unchanged and is not a footnote.** ⑥ runs `HEAP 910,008` against
⑤'s `2,097,152`, so 1.0964 is the cost of the Sublet *configuration* against the lookaside
configuration, each at its own working geometry — not the discipline in isolation. ⑤ cannot be
rebuilt at ⑥'s heap (it faults; image `7cc434e807570136`).

### §4g.6 — BOOT sw62: the 1,728 bytes are BOUNDS RE-ENCODING, not a failed fill (2026-09-12)

On `caplifive_r30r31_1bfff7776`, monitor `d1bd7e4`, firmware `5c1d8fc7e40a`. Control `k800
retval=4`. Four arms, one image (`ccb73bc08db39990`, the same cell5 image sw60's probe used), three
arenas differing only in size — so arena size is the only variable against sw60.

**The predictions were written into the driver header and into `ISSUES.md` BEFORE the boot**, because
the two accounts on the table differed numerically and the result had to be unable to fit either
after the fact. Bounds compression predicts `round_up(N, 2^(E+3)) − N`; a proportional store-failure
rate predicts `1,728 × N/1,419,584`.

| arm | arena | granule | compression predicted | store-failure predicted | **measured** |
|---|---:|---:|---:|---:|---|
| 2 | 1,419,264 | 2,048 | 0 — no halt | ~1,728 — halt | **clean, no halt** |
| 3 | 709,632 | 1,024 | 0 — no halt | ~864 — halt | **clean, no halt** |
| 4 | 354,880 | 512 | **448** | 432 | **`RCSH:000001C0` = 448** |

Both granule-aligned arenas reclaimed cleanly and the monitor's reclaim counter advanced 0 → 1 → 2,
so the reclaim genuinely ran and completed on each rather than being skipped.

**Arm 4 is the finding in two numbers.** `RCCU:00056A40` = 354,880: the cursor reached the TRUE end,
so all 22,180 stores advanced and **none failed**. `RCEN:00056C00` = 355,328 =
`round_up(354,880, 512)`. Their difference is 448, equal to `RCSH`, so the instrument's self-check
holds and the shortfall is the rounding and nothing else.

**One capability reported two different `end` values inside one arm.** `ALEN` is traced on the share
path before the fill, cursor still at base: **354,880**, the exact requested size. `RCEN` read the
same region after the first store moved the cursor: **355,328**.

Mechanism, exposure and the firmware mitigation are in **ISSUES R-33**. In one line: `compress_bounds`
(`ariane_pkg.sv:787`) uses an exact form only while the cursor sits at the low bound, and otherwise
rounds the top up to a `2^(E+3)` granule (`:827-828`); `STC` is a DYN op (`decoder.sv:1309`) whose
`rs1` is re-compressed on writeback (`ex_stage.sv:1188`).

**Consequences for numbers already in this document.** §4g.4's 1,728 is
`round_up(1,419,584, 2048) − 1,419,584` exactly, and is not evidence of any failed store. §4g.5's
5,334 successful INITs are consistent rather than anomalous: the `E == 0 && len[12] == 0` sub-case
(`ariane_pkg.sv:818-821`) is exact, so small regions never round. **No performance figure in this
document is affected** — every cycle count here comes from arms that completed, and this defect
either halts the monitor or is absent.


## §7 — Timing closure across every routed build (2026-08-27)

**No bitstream this project has ever produced has closed timing.** Seven distinct commits, **eleven
routed builds**, every one negative; range **−10.629 to −16.400 ns**, best **−10.629 ns** against a
**40.000 ns** period (25 MHz,
confirmed from the report's own clock definition and from the MMCM's
`CLKOUT1_REQUESTED_OUT_FREQ` of 25).

Every figure read from that build's own post-route report,
`work-fpga/ariane_xilinx_timing_summary_routed.rpt` inside its archived tarball, **Intra Clock
Table row `clk_out1_xlnx_clk_gen`**.

| build | WNS (ns) | failing / total endpoints | tarball bytes |
|---|---:|---:|---:|
| `39b21639d` | −10.629 | 96,727 / 174,481 | 407,871,086 |
| `76b7f2afc` | −12.084 | 93,200 / 174,275 | 405,677,186 |
| `84ed6eafb` | −13.516 | 103,197 / 175,200 | 407,837,655 |
| `52fa06b9d` | −14.125 | 104,238 / 174,461 | 392,443,910 *(arm A, retiming OFF)* |
| `52fa06b9d` | −14.832 | 104,457 / 174,785 | 407,010,879 *(arm B, retiming ON)* |
| `80843404c` | −16.400 | 102,769 / 174,275 | 405,480,965 |
| `6f8345fdb` | −13.491 | 99,879 / 173,789 | ~396 MB *(S-12 fix, debug tree TIED OFF)* |
| `5097eb166` | −15.311 | 101,782 / 174,895 | ~400 MB *(S-12 fix, instrumented)* |
| `947327f6d` | −11.717 | 97,438 / 174,756 | 403,665,530 *(rebuilt `fpga-testing-dev-clean`: same fixes, instrument NEVER ADDED; 2026-09-07)* |
| `ef5a8eaf2` | −12.733 | 101,143 / 174,188 | 404,402,489 *(`947327f6d` + the registered switch-in-progress flag; 2026-09-07)* |
| `66c4e7517` | −12.425 | 102,508 / 174,960 | 399,051,406 *(`ef5a8eaf2` + the R-25, R-26 and R-27 RTL fixes; 2026-09-09)* |


### The R-25/R-26/R-27 build, `66c4e7517` (2026-09-09)

Three RTL fixes over `ef5a8eaf2`: the R-26 pipeline flush after a committed capability CSR write
(`csr_regfile.sv`), the R-25 `INIT` guard that nulls `rs1` when `rs1 != rd`
(`capstone_flu_unit.anvil`), and the R-27 drain that retires a revocation-node response whose
requester was flushed (`ex_stage.sv`). Synthesised at 40 ns with no flow edit; guard exit 0 in
1 h 19 m, synthesis peak 25.90 GB against a 100 GB ceiling. Bitstream 11,443,722 bytes, sha256
`b03bd9673b9a685c31dff541fce1e7ba07b7ce9901e51051a19ac31e21652da3`; the artifact is retained on the
synth machine under the synth machine account's scratch directory.

**Provenance of that hash, stated exactly.** It is the synthesis lane's computed value, checked by them
against both copies on that machine — the in-tree `work-fpga/ariane_xilinx.bit` and the same member inside
the run tarball — and confirmed identical, with no third copy under `ariane.runs/impl_1` (the flow's own
Makefile has already moved it). **Transferred and independently confirmed 2026-09-09**, on the
project lead's authorisation: the board lane streamed the tarball member to the working machine as
`~/capstone-artifacts/bitstreams/caplifive_r25r26r27_66c4e7517.bit` with a `.sha256` sidecar, and the value
now has three independent computations in agreement — the synthesis lane's on its own machine, the board
lane's on arrival, and a third by the RTL lane against the landed file, all `b03bd967…52da3` at 11,443,722
bytes. The transfer had to originate from the working machine: the synth machine has no route outward, no
name resolution and no outbound key. **Nothing has been uploaded to the board console and nothing flashed**
— the flash gate is unset and the decision is the project lead's.

The earlier version of this paragraph said the hash was unverified by transfer, which was true when
written; it is superseded rather than deleted because the sequence is the point. Read a mismatch against
this value as a transfer problem only after re-checking the source, since the first quote of it came from
the in-tree copy while being described as the tarball member's. The file carries its commit in its name and
its hash in a sidecar, which is the practice worth keeping: a bitstream named for a property rather than a
commit has twice on this project travelled further than the artifact that carried it.

**The prediction was written before the run and both halves held:** WNS in [−15.3, −11.7] read
−12.425, placed LUTs in [168.9 k, 170.5 k] read 169,207. The stated reasoning — one flush term, one
null write and about twelve flops of drain logic cannot move these measurably — is borne out.

| quantity | `66c4e7517` | `ef5a8eaf2` | delta |
|---|---:|---:|---:|
| WNS, `clk_out1_xlnx_clk_gen` (ns) | −12.425 | −12.733 | +0.308 |
| failing endpoints | 102,508 / 174,960 | 101,143 / 174,188 | +1,365 |
| Slice LUTs (placed) | 169,207 (83.03 %) | — | — |
| LUT as logic | 167,353 (82.12 %) | — | — |
| Slice registers | 93,145 (22.85 %) | 92,817 | +328 |

The register delta is the direction drain flops should push, and the opposite of the unexplained
−284 on the `ef5a8eaf2` build.

**This row does not license a flash, and must not be read as doing so.** It fails timing at 40 ns
like every build in the table; it is second-best of the family behind `947327f6d`'s −11.717 and
better than the flashed `5097eb166`'s −15.311 by 2.886 ns, but "better-timed than the one already on
the board" is not a licence — the shift rule puts closure for this design at roughly 53–55 ns.
Neither does the launch census: its worst launch is `dom_switcher/cur_idx_q_reg[0]`, which is inert,
but that is the same shape `5097eb166` showed before a second-launch query found a live cone behind
it on 99.8 % of its endpoints, and that query has NOT been run on this checkpoint. See the clean-tip
verdict and its 2026-09-08 retraction before citing the census for anything. A flash remains an
empirical risk decision and the project lead's.

Other gates on this build, for the record: `write_bitstream completed successfully` once against a
negative control of 0 and a positive control of 6; zero `DRC LUTLP-1` against 44 DRC mentions in the
same log; 100 `found timing loop` lines, identical to `ef5a8eaf2` and the whole lineage; routed
legally with zero illegal or unroutable nets.

**Read the right row.** `eth_rxck` is the *first* "Failing Endpoints" line in that report and it
reads healthy while the CPU clock fails. That trap has caught this project before.

**Deliberately absent, not counted as zero:** `1cb22e30a`, `c2211c9a8`, `eaa4e7984`, and the
killed retiming-ON attempt produced **no routed report** — they never routed, so they are outside
the claim rather than scored in it.

Provenance: read from the archived artifacts by the synthesis lane, 2026-08-27; artifacts
retained on that machine. Any figure can be re-verified against its source.

**Related correction, same date.** "Retiming-ON does not complete with this RTL" is **refuted**:
the run that appeared to die was killed by a memory ceiling that had counted a *second concurrent
run's* collector against it. Rerun with correct process scoping it peaked at **21.15 GB**,
completed the full flow, and did synthesis **41 min faster** than retiming-OFF (213 vs 254). The
flow deviation was never necessary: it bought nothing, cost 41 minutes a build, and was adopted
on the strength of a comment that disagreed with working code. On timing the two arms are a
**trade, not a dominance** — OFF routes marginally better (−14.125 vs −14.832, 104,238 vs 104,457
failing), ON synthesises 41 min faster, and 0.7 ns against ~174,000 endpoints is no meaningful
difference. This strengthens CLAUDE.md's "do not change the synthesis flow"; nothing there needs
changing.

> **CORRECTED 2026-09-04 — retained as the record of what was fixed.** The paragraph above
> originally read "it produced the worse-timed build", which contradicted this document's own
> table; the table is the sourced side and the text has been fixed. Both arms are `52fa06b9d`,
> read from their own
> post-route reports: retiming **OFF** is WNS **−14.125** with **104,238** failing endpoints;
> retiming **ON** is **−14.832** with **104,457**. Less-negative WNS is better — this document
> says so itself two paragraphs up ("best −10.629 ns") — so **OFF is the better-timed build on
> both metrics, not the worse one.**
>
> What the measurements actually support is a **trade, not a dominance**: retiming-ON synthesises
> **41 minutes faster** and retiming-OFF routes **marginally better-timed**. The differences in
> slack (0.7 ns) and endpoint count (219 of ~174,000) are small enough that the honest reading is
> "no meaningful timing difference, and ON is faster".
>
> **The conclusion survives; its stated cause does not.** "Do not change the synthesis flow" is
> still right, but not because turning retiming off produces worse timing — it does not. It is
> right because the deviation bought nothing, cost 41 minutes a build, and was adopted on the
> strength of a comment that disagreed with working code. Anyone re-deriving the rule from the
> "worse-timed" claim will find the numbers contradict them and may discard the rule with it,
> which is the specific harm this correction exists to prevent.

## Rate ladder for SQLite on silicon — started 2026-08-27, current compiler

**Why this exists.** The standing claim is "SQLite passes its correctness workload on silicon,
3/3". Two things make that insufficient. The rate was never established — three reps only — and
the extended workload **contains a two-table join**
(`sqlite_capstone_domain.c:1439`, `SELECT COUNT(*) FROM nums JOIN link ON nums.label = link.label`),
which is the construct S-12 triggers on at a measured **54% per draw**. If that rate applied here,
3/3 clean has probability ≈ 0.10. Either the 3/3 was luck, or the join SHAPE matters
(`JOIN ... ON` over an indexed column vs `qj2`'s cartesian `FROM t1, t2`).

**And the 3/3 was measured on the OLD compiler.** The `lcc`→`mv` change rewrites ~192 sites in the
`-O0` image, so nothing currently validates SQLite-on-silicon for the shipping toolchain.

### Boot 1 — resident `caplifive_s07clear_84ed6eafb.bit`, no reflash

Four DISTINCT draws (`CAPSTONE_TEXT_PAD` 0/32/64/96, sha256 verified 4-of-4 unique), extended
workload, no `--slt`.

| draw | outcome |
|---|---|
| `sqr0` | **returned** — `EXTENDED_PASSED`, `MEMORY_PASSED`, rc=0 |
| `sqr32` | **returned** — `EXTENDED_PASSED`, `MEMORY_PASSED`, rc=0 |
| `sqr64` | **INFRASTRUCTURE WEDGE**, monitor tag `SPLB:0000E010` — NO VERDICT |
| `sqr96` | never ran — collateral |

**`SPLB` is a MONITOR spin tag, not a domain result:** `split_out_cap`'s unimplemented exact-fit
case, an M-mode `while(1)`, which the monitor's own comment records as wedging runs 5-7 in 4 of 4
boots and as the source of a large share of this campaign's random wedges. Conflating it with a
domain wedge produced a confident false localization on 2026-08-06; the driver now separates them.

**Two clean draws on the current compiler.** That is the first board evidence that today's
toolchain changes did not break SQLite on silicon — nothing else covered it.

**Practical yield is ~2 big domains per boot**, because `split_out_cap` spins on the third. So
n ≈ 30 needs on the order of 15 boots, not the 8 a 4-slot budget would suggest. Worth knowing
before committing to the ladder.

### Boots 2-7 — completed 2026-08-27

Twelve further draws, every one a DISTINCT image (`CAPSTONE_TEXT_PAD` 32…416, sha256 checked
2-of-2 unique per boot before staging), extended workload, resident bitstream, no reflash.

**RESULT: 14 clean draws, 0 domain wedges.** Every boot: `EXTENDED_PASSED`, `MEMORY_PASSED`,
`rc=0`. Verified per boot with zero `SPLB`, zero infrastructure classifications and zero wedges,
so none of the fourteen is a mis-attributed monitor spin.

| if the extended workload's join behaved like `qj2` (54% wedge/draw) | probability |
|---|---|
| 4 clean | 4.5e-2 |
| 8 clean | 2.0e-3 |
| **14 clean** | **1.9e-5** |

**So it does not.** The extended workload's `JOIN ... ON nums.label = link.label` and `qj2`'s
cartesian `FROM t1, t2` are both two-level where-loops and they behave completely differently on
silicon. **The variable is not "two levels" — it is something that differs between these two join
forms.**

**What NOT to conclude, because it was already killed once.** The obvious guess is that SQLite
builds an automatic index for the equijoin, turning a repeated inner SCAN into a SEARCH. That
theory is on file and was refuted: `02eda1190ca4` found `p11_smalljoin` doing
`SCAN t1` + `SEARCH y USING AUTOMATIC COVERING INDEX`, but `4534e1d0f302` then killed it —
`q_two` has no WHERE clause, so the planner builds no automatic index and no bloom filter, **and it
wedges regardless**. Do not re-derive it.

**The rule that survives from that trail is the one to apply here:** *on a query engine, same shape
of SQL is not same execution plan, and the plan must be READ rather than inferred from the text*.
Neither plan has been read. That is the next step, and it is board-free — `EXPLAIN QUERY PLAN` on
both forms under the QEMU reference model.

### What this settles for the paper

* **SQLite's correctness workload runs reliably on silicon on the CURRENT toolchain: 14/14.**
  This also revalidates the 2026-08-27 compiler changes (128-bit store merging, `lcc`→`mv`,
  GEP speculation), which nothing else covered — the previous 3/3 was measured on the old compiler.
* **S-12 does not block it.** S-12 is specific to a query shape the workload does not use.
* Still **correctness, not performance**: no admissible timing number comes from these runs.

**Planning fact, measured:** only TWO large SQLite domains per boot carry a verdict. At slot 3 the
monitor's `split_out_cap` spins (`SPLB`) — the same image that returned at slot 1 was created
(`SQ: A/dom-ok`) and never entered (`SQ: G/enter` = 0) at slot 3. That is infrastructure, not a
domain result, and conflating the two produced a false localization on 2026-08-06.

## SLT RUNS FULLY ON SILICON — and the S-12 trigger is narrowed to an unindexed nested SCAN

**The plans were READ, not inferred** — the rule this project learned the hard way, since
`sqlite3WhereCodeOneLoopStart` is the where-loop CODE GENERATOR and the wedge is at PREPARE time.
`EXPLAIN QUERY PLAN` is compiled out of the domain (`SQLITE_OMIT_EXPLAIN=1`), so this was read from
a native build with the domain's planner-relevant defines:

| query | plan | silicon |
|---|---|---|
| extended workload `JOIN … ON` | `SCAN nums` + `SEARCH link USING AUTOMATIC COVERING INDEX` | **14/14 clean** |
| **`qj4`** (new, indexed) | `SCAN t1` + `SEARCH t3 USING COVERING INDEX idx_t3a` | **RETURNS** |
| `qj2` cartesian | `SCAN t1` + **`SCAN t2`** | wedges |
| `q_two` self-join | `SCAN t1` + **`SCAN y`** | wedges |
| `q_one` | `SCAN t1` | returns |

**"Two where-loop levels" is the wrong characterisation.** `qj4` and `qj2` are both two-level
joins, both on empty tables, run from the SAME domain image in back-to-back boots — the only
difference is whether the inner level is an indexed SEARCH or a repeated SCAN. One returns, the
other wedges.

**So the trigger is generating code for an UNINDEXED NESTED SCAN at level 2.** That is far narrower
than the working characterisation this investigation has used since the level-2 finding, and it is
consistent with everything on file: `4534e1d0f302` established the wedge is at PREPARE time, and
its refutation of the automatic-index theory said `q_two` gets no automatic index and wedges
anyway — which is exactly what the plan above shows.

### SLT on silicon: WORKS, fully

`qj4` on the board: `SLT-SUMMARY records=4 stmt_pass=3 stmt_fail=0 query_pass=1 query_fail=0
skip_big=0 oom=0 parse_err=0 completed=1`. Three statements and a **passing query**, including a
two-table join. This is the first SLT run on silicon to pass its query rather than merely complete.

**OPERATIONAL LIMIT, measured: exactly ONE SLT domain per boot.** Each SLT domain carves a 1 MiB
region (`SLT_REGION_SIZE`), and the SECOND `create_dom` fails — `SQ: A/dom-ok` absent, so the
domain is never created and NOTHING in it runs. This is not the `SPLB` monitor spin (`SPLB` = 0).

**A near-miss worth recording.** The first attempt ran `q_one`, `qj4`, `qj2` in one boot. `qj4` was
arm 2, did not return, and would have read as "the prediction is refuted" — but `A/dom-ok` was
absent, so its domain was never created and the arm carried NO verdict about the query. Re-run
alone in slot 1, `qj4` returned and passed. The driver's own comment names this exact trap: absence
of `A/dom-ok` means blaming that arm produces "a confident, entirely false localization", as it did
on 2026-08-06.

**Weight, stated honestly:** `qj4` is **N = 1**, and at the 54% per-draw rate a single return is
p = 0.46 by chance. The account does not rest on it alone — the extended workload is the same
indexed-inner shape at 14/14 — but `qj4` itself needs redraws before the pairing is quantitative.

### §7a — The price of on-chip observability, measured (2026-09-04)

`5097eb166` and `6f8345fdb` are the same S-12 fix from the same base (`80843404c`) differing
**only** by whether the debug tree is tied off. First clean single-variable measurement of
instrumentation cost on this design:

| | cost of the debug tree |
|---|---|
| placed LUTs | **+750** (0.37% of the 203,800-LUT device) |
| failing endpoints | **+1,903** |
| WNS | **−1.820 ns** |

**The timing cost is disproportionate to the area.** 0.37% of the device costs 1.82 ns out of an
already-failing 40 ns budget. Anyone proposing added on-chip observability on this design should
price it against this pair, not against LUT count.

Both routed legally: no DRC `LUTLP-1`, "found timing loop" = 100 on both, identical to the base.

**Qualified 2026-09-07 by a second build without the S-07 instrument.** `947327f6d` (the rebuilt
`fpga-testing-dev-clean`: the same fixes, the S-07 recorder/aperture layer never added at the RTL
level; the base's 12 debug-bank apertures and upstream's UART instruction tracer `core/tracer.sv`
remain, as on every build here) routed at **−11.717 ns**, **97,438** failing endpoints, **170,481**
placed LUTs (83.65%). That is *more* area than `6f8345fdb` (168,944, the WHOLE debug-LED tree tied
off) and than the fully instrumented `5097eb166` (169,694). The two removals are not the same
variable — one drops the S-07 layer, the other the entire tree — yet the build with more debug logic
left in came out 787 LUTs *larger* than the one with all of it. The run-to-run spread of this flow is
**unmeasured** — no commit has ever been built twice with identical settings here (the only repeat,
`52fa06b9d`, changed the retiming setting between its two runs) — so nobody can say whether 787 LUTs,
or the +750 above, is signal. **Do not cite either LUT figure as the instrument's area cost until a
same-settings replicate exists.** The timing direction
holds on both builds without the S-07 layer (−13.491 and −11.717 against −15.311); 947327f6d is the
best-timed fix-carrying build on record, bettered only by `39b21639d` (−10.629, a timing-experiment
branch). Same DRC/loop signature: no `LUTLP-1`, "found timing loop" = 100. The tracer itself is a
436-line trace buffer with a UART dump, synthesised unconditionally since it was added upstream; its
cost has never been measured (65,562 of this build's failing endpoints terminate inside it).
Launch census: 97,438 of 97,439 from `dom_switcher/req_en_q_reg[0]` (the busy level), 1 from the DDR
controller on `clk_pll_i` — a third distinct dom_switcher launch register across four builds. **Census verdict: NOT usable as a
board bitstream** — that register is the busy level that gates commit, and the trap-entry CSRs sit on
its failing cone (`bitstream-usability-is-the-census-not-the-slack.md`, 2026-09-07 entry).

**Qualified again 2026-09-08 by the fix build.** `ef5a8eaf2` (`947327f6d` plus one registered
switch-in-progress flag feeding `commit_stage`, `controller` and `frontend`; RTL otherwise identical)
routed at **−12.733 ns**, **101,143** failing endpoints, **170,410** placed LUTs (92,817 registers, 284 fewer
than the base for one added flop — unexplained, variance unmeasured). The three readings
pre-registered for it all hold — no failing path launched from the new flag, the flag's own input meets
timing, and `req_en_q` launches nothing that fails — so the busy-edge hazard is closed on this build.
Launch census: 101,143 of 101,143 from `issue_read_operands` (worst launch `lsu_valid_q`, the issue-to-LSU
valid, live on every memory instruction). **Census verdict: NOT usable as a board bitstream**, the shape
of `6f8345fdb`. Same endpoint population as `947327f6d` (tracer 65,562, identical count), different worst
launch: what the worst-launch census cannot see is whether an endpoint also fails from a second launch;
a per-checkpoint query on the three retained routed checkpoints is the instrument for that (census doc,
2026-09-07/08 entry). No bitstream from `fpga-testing-dev-clean` is usable; the resident `5097eb166`
stays the one in use. **Retracted the same night (census doc, RETRACTED 2026-09-08):** that query ran on
the resident's own routed checkpoint and found 101,604 of its 101,784 failing endpoints also failing from a
live register (the LSU bypass occupancy counter, −15.157). The census never licensed the resident either;
its board record does, and why the silicon works at all is unmeasured. All timing figures in this table
are unaffected; what is gone is the sentence that a failing build is fine because its launches are inert.

### §7b — SQLite logic tests on capability silicon, and what may NOT be claimed yet (2026-09-04)

**New, citable — and it is a LIVENESS result, not a correctness one.** The SQLite logic-test
corpus now executes in a capability domain on silicon: `s12stress` completes **120/120** as the
native x86 baseline does, and the corpus matches native **15/15** under QEMU. Resident bitstream
`caplifive_s12fix_5097eb166.bit`.

**⚠ "Matches native" means COMPLETED WITHOUT WEDGING, not COMPUTED THE RIGHT ANSWER.** The
queries return **no rows on both sides**, so the agreement is strong about liveness and nearly
vacuous about computed results. These files are wedge probes by construction and say so in their
own opening lines — `p8_trivial.test`: *"WEDGE PROBE, not a correctness test: expected values are
dummy, the signal is RETURNED vs WEDGED"*; `s12stress.test`: *"NOT a delta-debug rung — an
INSTRUMENT."* Every table is deliberately empty because S-12 fires at PREPARE time.

**Establishing SQLite correctness on capability silicon requires a corpus with populated tables
and real expected values. That has not been run.** Do not let this result stand in for it.

**Correction, 2026-09-05 (later the same day): it has now been run, once, and it passed.** Boot B8
(cycle-2 regression sweep, resident `caplifive_s12fix_5097eb166.bit`, control `k800 = 4` first) ran
`select1.test` from the stock SQLLogicTest corpus — 1031 records against **populated** tables with
**real** expected values — in the silicon-config SQLite domain (image `c01e6b89cad0f17a`, host
`a1895d35f768b5d0`) and read `records=1031 stmt_pass=31 stmt_fail=0 query_pass=1000 query_fail=0
completed=1`, identical to the native x86 baseline. That is the first SQLite **correctness** result
on capability silicon: 1000 queries with checked answers, zero divergences. It is one file of the
seven-file corpus that matches native under QEMU (10,807 records); the remaining six are being run
one boot each (see the campaign rows dated 2026-09-05 in `tests/board-results/` and the paragraph
below once they land). Until then the citable line is: *"SQLite 3.53.3 in a capability domain on
FPGA silicon executes SQLLogicTest `select1` (1031 records) with results identical to native."*

### §7c — The SQLLogicTest corpus on capability silicon (2026-09-05 → 2026-09-07)

Seven files, one boot each, control `k800 = 4` first in every boot, resident
`caplifive_s12fix_5097eb166.bit`, silicon-config SQLite (see the optimisation-level note below), each result read from the run's own
transcript segment and compared with the native x86 baseline produced by the same runner
> **Optimisation level, corrected 2026-09-10.** Earlier revisions of this section described the SQLite
> domain as "-O1". That is the level of the SUPPORT code only. The build chain:
> `build-slt-corpus-images.sh:35` invokes `run-sqlite-slt.sh` with no optimisation variable set and
> `run-sqlite-slt.sh` sets none either, so `build-sqlite-silicon.sh` takes its defaults —
> `OPT=${SQLITE_OPT_LEVEL:--O0}` at `:42` for the SQLite amalgamation and
> `SUPPORT_OPT=${SQLITE_SUPPORT_OPT_LEVEL:--O1}` at `:2724` for the support code. So the corpus images
> are **SQLite itself at -O0 with the support code at -O1**. Confirmed independently: a build with no
> optimisation variable set reproduces the corpus image hash exactly, which a different amalgamation
> level could not. The measurements are unaffected — only their description was wrong. The same
> correction was applied to the opt column of every `sqslt1m` row in `tests/board-results/2026-09-05.tsv`.

(`benchmarks/sqlite/slt/slt_runner.h`, `build-slt-native.sh`). Images: fresh main-checkout toolchain
(`libLLVMCapstoneCodeGen.so bfc039bf12e077f5`, clang embedding `fc4f826a16ca`), one image per
region/heap class, every image validated under QEMU on its own file before it was baked
(`tests/rtl-smoke/slt-corpus/build-slt-corpus-images.sh`). Rows: `tests/board-results/2026-09-05.tsv`.

| file | records | statements | queries | silicon vs native | boot |
|---|---|---|---|---|---|
| `negative-control.test` | 21 | 9 pass / **2 fail** | 6 pass / **4 fail**, 2 skipped, 1 parse error | identical — the comparator's positive control on silicon | sw23 |
| `evidence/slt_lang_aggfunc.test` | 80 | 12 / **1 fail** | 57 / **10 fail**, 1 skipped | identical, including the corpus's own artifacts | sw24 |
| `select1.test` | 1031 | 31 | 1000 | identical, 0 failures | B8 |
| `select2.test` | 1031 | 31 | 1000 | identical, 0 failures | sw27 |
| `select3.test` | 3351 | 31 | 3320 | identical, 0 failures (~5 min of execution) | sw28 |
| `select5.test` | 1436 | 704 | 732 | identical, 0 failures (2 MiB heap, 1 MiB stack) | sw26 |
| `select4.test` | 3857 | 1025 | 2617 + 215 skipped for size | identical, 0 failures (4 MiB region, ~78 min of execution) | sw29 |

**What this establishes.** The two files with deliberate and known failures reproduce them exactly
on silicon (2 + 4 and 1 + 10), so a clean row is a clean row and not a comparator that cannot fire.
**All seven files — 10,807 records — are identical to native, zero divergences, on the current
bitstream and the current compiler.** The citable line: *"SQLite 3.53.3 in a capability domain on
FPGA silicon executes the SQLLogicTest `select1–5` files and the aggregate-function evidence file —
10,807 records, 8,746 queries with checked answers — with results identical to native x86, and
reproduces the negative control's deliberate failures exactly."* This supersedes §7b's "has not
been run": SQLite correctness on capability silicon now rests on the stock SQLLogicTest corpus, not
on wedge probes. Caveats, stated: one draw per file (the corpus is deterministic, but silicon has
known non-deterministic defects and a second pass would cost about an hour); the 9p share and the
console path produced VOID boots that were re-run rather than counted; and `select4` was the first
file whose region class (4 MiB) was carved on the board; it worked first time.


It is also **not** a performance result — no `mcycle`/`minstret` figures accompany it. So it
closes part of the **compatibility** axis, and neither the correctness nor the cost axis.

**⚠ Any SLT pass rate from before 2026-09-04 is VOID.** 12 of the 15 test files declared an
expected value for queries over **empty tables that return no rows**, so those expectations were
never evaluated: the files were built as wedge probes where "passes" meant *"did not wedge"*. The
corpus has been fixed and re-verified against the native baseline. **Re-derive any number that
rests on a pre-2026-09-04 SLT rate.**

**RETRACTED 2026-09-04, same day: my characterisation of this as an instrument "silently"
changing role was wrong.** Each file announces its role in its own opening lines (quoted above),
and the project lead pushed back on the same grounds. Nothing was silent and no new rule is
warranted. The actual defect was a *summary sentence* — "identical to the native x86 baseline",
which reads as correctness when the result is liveness. The lesson is about how a result is
described downstream, not about the corpus.

**⚠ S-12 must not be described as "fixed" in a paper yet.** `ref/ISSUES.md` now reads
**"ROOT-CAUSED, FIXED IN RTL, FLASHED — verification is CONSISTENT WITH FIXED, not proven"**. The
fix is real, synthesised and flashed, and the domain that trapped completes 4 draws of 4 — but the
pre-fix arm trapped 3 of 4, giving Fisher **p = 0.071**, and P(4 clean) against this project's own
~54% per-draw rate is **0.045**. The registry has already ruled 0.045 insufficient for a cure
claim elsewhere, so the internal-consistency argument, not merely the p-value, is what forbids it.
**Two more clean draws would give 0.0095.** Until then: "consistent with fixed".

**One prediction that failed, worth recording.** `5097eb166` **improved** timing over its base
`80843404c` — WNS **−16.400 → −15.311**, and **987 fewer** failing endpoints (102,769 → 101,782,
confirmed against §7's table) — despite adding a term to a stall condition, which everyone
expected to cost time. Whatever intuition predicted otherwise should not be trusted on this
design.

Provenance: board and RTL lanes, 2026-09-04.

### §7d — The CHERI-CVA6 bitstream RUNS on our Genesys2 (2026-09-09)

**First non-Capstone core executed on this board, and the first silicon evidence for the CHERI tag
path.** Build B of the zero-day-labs `cheri-cva6` fork (branch `genesys2-eval` `a9568ac2`; the synthesis comparison is in
`docs/plans/cheri-cva6-on-genesys2.md`, not yet in this file) was written
to the board's config memory **non-volatile**, and a bare-metal UART smoke test ran on it. Driven by the
board lane from `~/capstone-artifacts/cheri-cva6/board-package/`; records in
`~/capstone-artifacts/cheri-cva6/board-run-1/`. One session under the console lock, 14:48–14:54.

| step | reading |
|---|---|
| resident before / after | `caplifive_s12fix_5097eb166.bit` both times (restore verified) |
| write | `cheri_cva6_B_a9568ac2_25mhz.bit`, persistent, 90 s |
| bitstream live? | **yes** — post-flash UART is the fork's bootrom (`Hello World! / init SPI / status 0x25 ×2 / SPI initialized! / initializing SD...`), not Capstone's (`Hit any key to enter update mode`) |
| JTAG | `riscv.cpu tap/device found: 0x00000001`, irlen 5 — same as the Capstone SoC |
| load | 8,388 bytes to `0x80000000` in 0.08 s; `x/4i` shows the image's own first instructions |
| UART output | `T1 tag=1`, `T2 tag=1`, `T3 tag=0`, `T4 trapped=1 mcause=1c mtval=2a1`, `SMOKE PASS` — byte-identical to the Verilator transcript |
| DRAM mirror `x/7gx 0x80001000` | `1, c0de0002, 0, 1, 1c, 2a1, 80000150` — pass pattern; `trap_pc` is T4's `lc.cap` |

**What this establishes that simulation could not.** DDR3 calibration with this bitstream; **the tag
controller against real DDR** — a capability stored at `0x81000100` and reloaded keeps its tag, so the tag
table at `0xBFC00000` and the MIG address path work (this was the single largest unknown, since the shipped
fork constant overflowed DRAM and was fixed for these builds); an integer store clears the tag; the
capability exception path traps with cause `0x1c` and `mtval` naming the faulting register; the UART at
57600 from the 25 MHz core clock; JTAG and debug-module access to this SoC. The core ran at the 25 MHz the
bitstream was constrained for, i.e. the clock the synthesis comparison in
`docs/plans/cheri-cva6-on-genesys2.md` measured its +6.5 ns of slack at.

**What it does NOT establish.** Nothing about performance: no benchmark, no cycle counts, no `mcycle`
readings. Nothing purecap and nothing Linux-level — no CHERI Linux or CheriBSD port for this SoC is in the
tree, so any near-term comparison with Capstone stays bare-metal. Four CHERI operations were exercised, not
the ISA. The three RTL defects found in simulation (§ the plan doc) were not re-tested here.

**Method notes worth keeping.** The verdict is corroborated twice over: the console text and a DRAM block
the test wrote itself, at fixed addresses after `tohost`, with every UART poll bounded so a dead UART could
not have wedged the run. Before trusting any reading, the *live* bitstream was confirmed from the bootrom
banner: had the old bitstream still been live, the same image would have executed CHERI opcodes on the
Capstone core and produced illegal-instruction traps that read like a CHERI failure. The discriminator was
written down before the run, not after.

Provenance: cheri lane (package, test, analysis) and board lane (console), 2026-09-09. Plan and full log:
`docs/plans/cheri-cva6-on-genesys2.md`.

### §7e — What a capability-store fill costs on silicon, measured by a matched pair (boots sw52 + sw53, 2026-09-10)

> **CORRECTED 2026-09-10 by an adversarial audit, then EXTENDED the same evening by boot sw53.**
> The audit found four errors — the cache-line count (off by 4×), the direction of the cold-buffer
> caveat, impact arithmetic that counted the loop's instructions but not its cycles, and a claim that
> the monitor's loop is shorter than the rung's (it is the same length). sw53 then answered the two
> questions the audit could only mark OPEN, and gave the figure an error bar. **Read sw53's numbers
> as current: 24.1 cycles per store at n=3.** sw52's 23.6 came from a rung that has since changed
> (its retval control could not prove the pointer walk), so the two are not the same instrument.

**Why this number was needed.** The R-30/R-31 firmware half overwrites a revoked region before
reusing it. Whether that is affordable was the open question in the reclaim decision, and the answer
had to be given as a bracket because one quantity in it had never been measured: what 4 KiB of
capability stores costs on this silicon.

**Instrument.** Four ladder rungs sharing one header, each a four-instruction loop differing only in
its payload, all on the resident `caplifive_r25r26r27_66c4e7517.bit`:

| rung | loop body | oracle | sw52 | sw53 cycles | sw53 instret |
|---|---|---:|---:|---:|---:|
| `fillcost` | `stc zero,0(p)` · `cincoffsetimm p,p,16` · `addi k,k,1` · `bltu k,n,1b` | 1792 | 8,196 † | **8,639** (×3) | 1,175 |
| `fillnop` | `nop` · (same three) | 256 | 2,143 † | **2,466** (×3) | 1,163 |
| `fillwarm` | `fillcost`'s loop, run **twice** over the same buffer | 4096 | — | **16,133** | 2,233 |
| `fillsd` | `sd x0,0(p)` · (same three) — half the bytes, same 256 lines | 5888 | — | **8,383** | 1,179 |

† sw52 ran an earlier `fillcost`/`fillnop` whose oracles were 768/256; its absolute cycles are not
comparable with sw53's, only the derived per-store figure is. Control `k800` returned 4 in both
boots, and sw53 completed 9/9 arms with zero fault tags.

A normalised disassembly diff of the `fillcost` and `fillnop` `.dom` files shows **exactly one differing instruction** in the whole `.text` (`nop` ↔ `stc zero,0(a4)`); both
loops sit at image offset `0x36c`, so they occupy the same byte offset within a cache line and index
the same sets, and alignment cannot be the confound.

**The load-bearing evidence is the instruction census, not the return value.** Static count of
`fillcost_compute`: prologue 16, seed 1, three hoisted reloads 3, loop 4 × 256 = 1024, post-loop 6,
the `+512` path 6 (fillcost only), epilogue 6 — **1,062 / 1,056**. Measured instret 1,131 and 1,125
leave a **residual of 69 in both arms**, the harness bracket. That proves, independently of any
return value, that the four-instruction loop retired 1,024 instructions in each arm, that nothing
extra ran in either, and that the entire 6-instruction difference is the `+512` path. It matters
that this is the evidence: the *old* 19-instruction C loop also returned 768 under emulation, so the
retval has already failed once as a discriminator between the two implementations.

**Result — the marginal cost of the store, now with an error bar (boot sw53, 9/9 arms, `k800` = 4,
zero fault tags).**

| rung | oracle | draws | cycles | spread |
|---|---:|---:|---|---:|
| `fillcost` | 1792 | 3 | 8,649 / 8,632 / 8,637 → **8,639** | 0.20 % |
| `fillnop` | 256 | 3 | 2,472 / 2,466 / 2,459 → **2,466** | 0.53 % |

    8,639 - 2,466 = 6,173 cycles for 256 capability stores over 4 KiB
                  = 24.1 cycles per 16-byte capability store

The loop overhead is **cancelled, not assumed**: same iteration count, and both arms' instret differ
only by the branch bodies one takes and the other does not. sw52's single draw of the earlier rung
gave 23.6, so the two boots agree to 2 % on a quantity neither was tuned to reproduce. **The
within-boot spread is 0.2–0.5 %**, so the number is repeatable and the remaining uncertainty is in
what it means, not in the reading.

Each rung was also repeated at its own entry VA within the boot, and every repeat **returned** — so
R-3 (a domain reused at a repeated entry VA silently hanging) does not bite the ladder path. That is
worth recording separately from the measurement.

**The cache line here is 16 BYTES, so this is one line per store — 256 lines over 4 KiB, not 64.**
`CVA6ConfigDcacheLineWidth = 128` (`capstone_cv64a6_imafdc_sv39_config_pkg.sv:50` at `66c4e7517`)
is in **bits**, as `build_config_pkg.sv:25`'s `$clog2(DcacheLineWidth / 8)` shows. An earlier version
of this section said 64 lines and was wrong by 4×.

> **REPLICATED, boot sw54 (2026-09-10).** Every figure in this section was re-drawn on a second boot
> — on a **different firmware**, carrying the rebuilt CMA kernel module — and every arm returned its
> oracle with no fault tags. `fillcost` 8,626 (−0.15 %), `fillnop` 2,459 (−0.28 %), `fillwarm` 16,153
> (**+0.12 %**), `fillsd` 8,389 (**+0.07 %**). The derived numbers move in the third significant
> figure: per-store **24.09** against 24.12, warm pass 2 **87 %** of pass 1 against 87 %, plain `sd`
> **96.2 %** of `stc` against 96 %. The two conclusions below were each n=1 in sw53; they are n=2 now,
> and they replicate to a tenth of a percent.

**A WARM REGION IS NOT MATERIALLY CHEAPER — measured, and the cold-buffer caveat is withdrawn.**
The prediction was structural: `CVA6ConfigDcacheType = WT` (`:76`) and `wt_dcache_wbuffer.sv:43-44` —
a returning write ACK updates the cache only *if the word is present*, and "if the word is not
allocated to the cache, it is just evicted from the write buffer" — so a store to a resident line
still writes through and warm need not help. `fillwarm` runs the same loop **twice** over the same
buffer and read **16,133** cycles against `fillcost`'s 8,639. The second pass therefore costs
**7,494 cycles, at least 87 % of the first** (at least, because `fillcost`'s 8,639 includes prologue
and epilogue that `fillwarm` pays only once — the true first-pass figure is lower, so the ratio is
higher). An earlier version of this section called the cold buffer "closer to a ceiling than a
typical case"; on this cache that is worth at most 13 %, and the sentence is gone rather than
softened. It also means the ~8 % impact figure below applies to the monitor's real case, where the
region has just been in use.

**Cost to the reclaim — count the LOOP, not only the stores.** The monitor executes the whole loop,
so the marginal-store figure is the wrong numerator for an impact estimate. The monitor's own
`C_RECLAIM` body is `beq / stc / addi / j` — **four instructions, the same as this rung's**; an
earlier version of this section claimed it was one shorter, which is wrong. Subtracting fillcost's
~140 cycles of non-loop code, the fill itself is:

    ~8,540 cycles per 4 KiB reclaim   (of which 6,173 the stores, ~2,370 the loop the monitor also pays)
     1,024 instructions

Against **1 revoke per 28,829 instructions**, which is a **native x86, gcc -O2** measurement of
speedtest1 `main/size100` (`docs/history/22-07-2026_00-27-29_…-tier1-frequency-…md:26-27,43`) and
**not** a silicon measurement of the domain — the domain runs more instructions for the same work, so
every percentage below is **overstated** on that axis:

| baseline CPI | reclaim's share of cycles |
|---|---:|
| 3.17–3.81 — **speedtest1's measured range** (§7f) | **7.8 – 9.3 %** |
| 2.5 | 11.8 % |
| 1.5 | 19.7 % |
| 1.13 – 6.44 (the full on-silicon spread, §4) | 4.6 – 26.2 % |

So the figure to quote is **~8–9 % of cycles and +3.6 % of instructions at this workload's measured
CPI** — and the honest statement is that it is CPI-sensitive across the whole 3–25 % bracket the
decision was taken under, not that it "lands in the lower half" of it. The `fillwarm` result matters
here: because a warm region costs within 13 % of a cold one, this figure applies to the monitor's
real case and is not a cold-start artefact.

**Rate comparison, like for like.** `fillcost`'s **total** is 8,639 / 4,096 = **2.11 cycles per byte**,
against the separately measured **3.52** cycles/byte for a 1024-byte **copy** (line 122) — also a
total. The fill is cheaper than the copy, in the expected direction, since a copy loads and stores
where a fill only stores. (1.48 cycles/byte is the *marginal* rate and must not be compared against
that 3.52; the marginal rate at sw53's figures is 6,173 / 4,096 = 1.51.)

**The walk is now inside the control, and it was not on sw52.** `+512` keys on `fillcost_buf[0]` —
the *first* slot the loop touches, which reads back zero whether or not the pointer advanced — so
sw52's retval could not distinguish "walked 4 KiB" from "stored 256 times to one granule", and the
"4 KiB / 256 lines" framing rests on exactly that walk. Slot 255 is now seeded too and contributes
**+1024**, reachable only if the walk works. That is why `fillcost`'s oracle moved 768 → 1792 between
the boots: **a sw52 reading and a sw53 reading are different instruments**, and the oracle says so
rather than leaving it to a label.

**THE COST IS PER STORE AND IS ALMOST ENTIRELY NOT CAPABILITY-SPECIFIC — measured.** `fillsd` is the
identical loop with a plain 8-byte `sd` in place of the 16-byte `stc`: same 256 iterations, same
16-byte walk, so the same 256 lines are touched and **half the bytes are written**. It read **8,383**
cycles, a marginal **23.1 cycles per store** against `stc`'s 24.1 — **96 %**. So:

* the cost does **not** scale with bytes written (half the bytes, 96 % of the cost), it scales with
  **stores**, which is what a write-through drain does;
* the capability-specific part is about **1 cycle in 24**, ~4 %. The AXI adapter's shadow-tag
  transaction (`wt_axi_adapter.sv:108-123,153,413-442`) does not dominate, and it would not be
  expected to: `needs_tag` at `:153` fires for `DCACHE_LOAD_REQ` **and** `DCACHE_STORE_REQ`, so a
  plain store issues a tag transaction too — only with `tag_wr_value` 0 rather than 1.

So 24.1 cycles/store is **the write-through drain cost of a store on this bitstream**, and a capability
store costs a few percent more than a scalar one. Quoting it as a capability tax would be wrong.

**Also from this boot, and the reason the counter is in the firmware at all:** `RCLM:00000000`
appears on every share (9 of 9). Zero reclaims, as it must be where the guard cannot fire — and the
reporting path is now **proven readable on silicon**, which is the one property of it that could not
be tested after the flash.

### §7f — speedtest1 on capability silicon vs native, same boot, all three testsets (boot sw52, 2026-09-10)

**Pair ratios here carry three decimals; a comparison with another boot carries two — see the
PRECISION block at the end of this series.**

Every figure below was predicted before the run (`docs/plans/speedtest1-boot-predictions.md`,
committed `c62aeb3019fe`). Control `k800` returned 4, zero fault tags in the whole transcript, and
10 of 11 arms completed.

| testset | capability cycles | native cycles | **cycle ratio** | instr ratio (predicted) | CPI cap / native | hash |
|---|---:|---:|---:|---:|---:|---|
| `parsenumber` | 174,709,833 | 130,838,187 | **1.335×** | 1.546× | 3.80 / 4.40 | identical |
| `orm` | 736,453,341 | 623,670,978 | **1.181×** | 1.329× | 3.17 / 3.57 | identical |
| `main` | 2,635,649,345 | 2,170,454,254 | **1.214×** | 1.270× | 3.81 / 3.99 | identical |

**The hashes are identical between the capability and native arms of every testset**
(`0 0e12171d`, `465769 f3699caa`, `112006 38bb59fd`), so the two sides computed the same thing and
the ratio is like for like. `orm` and `main` also match the hashes predicted before the run.
`parsenumber`'s hash is structurally unable to fire — it is identical at sizes 1, 5 and 20 because
its leading field counts result rows and that testset returns none — so that arm is judged on its
instruction count and phase lines, as the prediction said it must be.

**The cycle ratio is BELOW the instruction ratio in all three cases, and that is the finding.** The
capability build executes 1.27–1.55× the instructions and takes only 1.18–1.34× the cycles, because
its **CPI is lower than native's in every arm**. The added capability instructions are cheaper than
this workload's average instruction — they issue in slots a memory-bound workload otherwise wastes.
Quoting the instruction ratio as the overhead therefore **overstates it by 5–16 %**.

**TWO ASYMMETRIES TRAVEL WITH THE CYCLE RATIO AND MUST BE NAMED BESIDE IT.** Neither affects the
instruction ratio, and both favour the baseline, so the cycle ratios above are an upper bound on the
capability ABI's cost rather than a clean measure of it.

*Compressed encodings, and the direction is that **these ratios OVERSTATE the capability overhead**.*
The evidence is a property of the delivered artifacts rather than of a build setting: `file` reports
the baseline as **RVC** and the domain image as not, and they are **926,248 B against 1,684,232 B**,
so the capability image is **1.82×** the size. The capability target has no compressed extension; the
baseline is `-march=rv64imac` and is 44 % compressed encodings.

Measured both ways on `main --size 1`, the instruction ratio moves **0.45 %** — a compressed
instruction still retires as one instruction — while the text moves **18 %** (926,248 B against
1,128,496 B for the same source without `c`). That difference is icache footprint: invisible to an
emulated instruction count, visible in a board cycle count, and **in the baseline's favour**. So the
true capability cost is LOWER than the cycle ratios above, not higher.

It is deliberately NOT changed: `build-ladder-base-fpga.sh` already uses `rv64imac`, so a different
`-march` would make these numbers incomparable with every overhead figure this project has
published. `SPEEDTEST1_BASELINE_MARCH` exists to re-measure it, not to switch it off.

*The granule guard.* `SQLITE_GRANULE_GUARD` is an R-29 workaround present only in the capability arm,
costing +33,660 bytes of `.text` and a branch per granule — in the primitive every timing number is
measured on. Any published ratio has to state its state.

**Clock check, independent of any counter we control.** `main` self-reports `TOTAL 105.255s`, and
2,635,649,345 cycles at 25 MHz is 105.4 s. The two agree to 0.1 %, confirming the 25 MHz core clock
from inside the workload rather than from the device tree.

**What did not run, and why it is not a result.** Arm 11, the `instret` counter probe, was never
reached. The driver hard-stops after any arm producing no `RESULT … retval=` marker, and the
host-binary probe arm emits `BASELINE-PROBE cycle = …` instead. Arm 10 itself returned `rc=0` with
`BASELINE-PROBE cycle = 20349224574`. The stop is a **classifier false positive on a marker format**,
not a failure of the run, and it cost only the last and cheapest arm. The `BASELINE-PROBE` form needs
adding to the classifier's success set before the next boot that uses a probe arm.

### §7g — The CMA module runs on silicon (boot sw54, 2026-09-10)

The CMA merge changed the kernel module's region allocation from `__get_free_pages` to
`dma_alloc_pages`, and **that path is not gated on `CONFIG_CMA`**: it runs for every region on every
target, the board included, where no CMA area exists. Until this boot no board image had ever
contained it — the last one predated the merge and shipped the old module, verified in the artifact
(`0` occurrences of `dma_alloc_pages` in the built source against `2` in the package source). The
next measurement boot would have carried it untested, on the path everything depends on.

**It is proven by the control, not by a dedicated arm.** Every ladder rung enters through `lpc`,
which **creates a region** — so `k800` returning its oracle *is* the module's positive control:
allocation went through `dma_alloc_pages` on silicon and produced a working capability. A failure of
the new path appears as a create/map failure *before* any rung result, which is distinguishable from
a rung miscompute.

| | |
|---|---|
| arms | **5 / 5**, every one at its oracle |
| `k800` | **4** — the boot control and the module's positive control |
| region/map failures | **0** |
| fault tags | **none** |
| the built `capstone.ko` | carries `dma_alloc_pages`, and the cpio contains that exact `.ko` — checked in the artifacts, not the source tree |

**What is still not exercised**, and should not be read as covered by this: no `reserved-memory` node
is added and `fpgakernel.config` gains no CMA, so on silicon this remains a pure allocator swap with
no new capability — a region larger than the buddy allocator's 4 MiB still cannot be created on the
board. The CMA half is exercised under emulation only (§7e appendix). And `ioctl_release_region`
decrements `region_n` without rolling back `pre_mmap_offset`, leaking mmap offset space monotonically.

> **SUPERSEDED the next day by §7h**, for both halves of that paragraph. The `reserved-memory` node
> and the `fpgakernel.config` CMA symbols landed on 2026-09-11 and a 130 MiB region now exists on
> silicon, so the sentence above holds for boot sw54's image and for nothing after it. The
> `pre_mmap_offset` leak is fixed in the same change. Read §7h before citing this paragraph.

### §7h — A 130 MiB capability region exists on silicon (boot sw55, 2026-09-11)

The board kernel now compiles CMA in (`configs/fpgakernel.config`) and the device tree reserves
256 MiB at `0xAC000000` carrying `linux,cma-default` (`configs/caplifive.dts`). **This is the first
capability region above 4 MiB on any target other than emulation, and the ceiling it clears has been
in place for the whole project.**

| arm | size | result |
|---|---|---|
| `k800` (control, first in the boot) | — | **retval 4**, 4,499 cycles, 53,406 ran, 1,089 instret |
| `bigregion` | 4 MiB | created, mapped, first+last byte round-tripped |
| `bigregion` | 8 MiB | created, mapped, first+last byte round-tripped |
| `bigregion` | **130 MiB** | created, mapped, first+last byte round-tripped |

**4 / 4 arms**, zero `Oops`/`BUG:`/`WARNING:` counted across each allocation, no fault tags. The
control ran first and passed, so the boot carries a verdict at all — it fails roughly one boot in
five and a failed control voids everything after it.

**The reservation is verified by SIZE, not by existence**, which is the check the whole design turns
on: `CONFIG_CMA_SIZE_MBYTES` defaults to **16 on non-x86**, so a *rejected* device-tree node leaves a
quiet 16 MiB default area behind, under which a small arm passes and a large one fails and the log
still shows a CMA line. The kernel's own lines, scoped to this run's `load_image` rather than to the
console's replay of the previous boot:

```
Reserved memory: created CMA memory pool at 0x00000000ac000000, size 256 MiB
OF: reserved mem: initialized node linux,cma@ac000000, compatible id shared-dma-pool
OF: reserved mem: 0x00000000ac000000..0x00000000bbffffff (262144 KiB) map reusable linux,cma@ac000000
Memory: 693668K/985928K available (… 262144K cma-reserved)
```

**Why the 130 MiB arm settles the mechanism without a failing board arm beside it.** The buddy
allocator's largest block is `MAX_ORDER 10` × 4 KiB = **4 MiB**, and the silent fallback area is
16 MiB; 130 MiB is 32× the first and 8× the second, so that allocation cannot have come from either.
The matched failing arm exists too, but under emulation: at 8 MiB with no CMA area the create fails,
and with `cma=256M` on the same image it succeeds (§7e appendix). Those four QEMU runs are
`/tmp/capstone/bigregion-{4194304,8388608,8388608-cma=256M,136314880-cma=256M}.log`, all of them
taken AFTER the module rebuild at 01:07:23 that put `dma_alloc_pages` into the guest — which matters
because an earlier arm failed on a stale module that had no CMA path at all, and a failure from THAT
is not the matched arm. The no-CMA arm carries its own control inside the log: `CmaTotal: 0 kB` and
`0K cma-reserved`, so the area really was absent rather than merely unrequested. What the board arms
do NOT separately measure is *which* allocator served the 4 MiB arm — it sits exactly at the buddy ceiling
and either could have — and that is why it is present as a probe control rather than as evidence.

**The domain's own memory does NOT come from CMA — now observed, not only derived.** The question
mattered because if the capability arm's working set sat in the CMA area while the native baseline's
did not, the two arms of every pair in §7k would differ in DRAM placement. The source answer is that
`ioctl_create_dom` allocates with `GFP_HIGHUSER | __GFP_ZERO` (`module/capstone.c:136`), which lacks
`__GFP_MOVABLE` (`gfp_types.h:334`), so `gfp_migratetype()` returns `MIGRATE_UNMOVABLE` and
`ALLOC_CMA` is never set (`page_alloc.c:3371-3375`) — both CMA paths are gated on that flag. Measured
under emulation with a 256 MiB area present and default:

```
cma: Reserved 256 MiB at 0x00000000f0000000          area = 0xf0000000 .. 0xffffffff
Domain memory region vaddr = ff60000081680000, paddr = 101880000
CmaFree: 260096 kB before the domain, 260096 kB after
```

**That log line no longer says "region", as of module commit `a185b65` the same day.** It reads
`Domain block (buddy, NOT a capability region) vaddr = ..., paddr = ...`. The excerpt above is kept
verbatim because it is what that run printed, but anyone reproducing it should grep for the new
wording — searching for the old one now returns nothing, and an absence with no mechanism behind it
is the trap this project keeps paying for. The rename exists precisely because calling this block a
"region" produced a retracted claim: six capability-region base addresses in the CMA range were read
as the domain's own memory having moved there.

`0x101880000` is **24 MiB above the top of the area**, and `CmaFree` does not move by a single
kilobyte across a domain creation whose block is at least 256 KiB. What DOES come from CMA is the
region allocation (`dma_alloc_pages`) — on the board those show as `BASE:AC0xxxxx` tags inside the
reserved range. So the split is: **regions from CMA, domain code/heap/stack from the buddy
allocator**, and the SQLite arena is a static array inside the latter.

`linux,cma-default` is load-bearing rather than decorative. The module registers `region_dev` with
`platform_device_register_simple(…, NULL, 0)`, so `dev->of_node` is NULL, `dev->cma_area` is never
assigned, and `dma_alloc_contiguous` falls through to `dma_contiguous_default_area`
(`kernel/dma/contiguous.c:313-333`) — which only a node carrying that property ever sets
(`:430-431`). Without it the node reserves memory nothing on this system can reach.

### §7i — The domain's instruction count is measurable on silicon, and it matches the emulator (boot sw57, 2026-09-11)

Every "domain CPI" in this document before today divided **board cycles by an emulator instruction
count**, because no board image reported the domain's own `minstret`. The instrumented image had
never booted. Boot sw57 ran it: control first, then seven arms ascending, **8/8, zero failures,
every verification hash equal to its prediction and to the plain image's**.

| testset | predicted `instret` | board `instret` | delta | relative |
|---|---:|---:|---:|---:|
| star | 59,443,559 | 59,443,629 | +70 | 1.2e-06 |
| parsenumber | 60,938,716 | 60,938,800 | +84 | 1.4e-06 |
| orm | 274,563,761 | 274,563,817 | +56 | 2.0e-07 |
| main | 696,764,734 | 696,765,119 | +385 | 5.5e-07 |
| fp | 933,826,083 | 933,826,300 | +217 | 2.3e-07 |
| cte | 2,019,055,526 | 2,019,055,631 | +105 | 5.2e-08 |
| rtree | 2,420,439,710 | 2,420,440,011 | +301 | 1.2e-07 |

**This is the check that was retracted on 2026-09-11 and is now actually made.** §7k records why the
earlier version was void: sw56 carried no domain instruction count, and the apparent agreement came
from a table that passed the predictions through as if they were readings. Here the two numbers come
from two instruments — QEMU `-icount` and the silicon `minstret` CSR — and agree to between 5.2e-08
and 1.4e-06 on counts up to 2.4 billion. All seven deltas are positive and range 56 to 385
instructions, which is far below one timer tick (~3,780), so it is a handful of instructions in the
invocation path rather than any periodic effect.

**The emulator's instruction counts are sound for this workload.** All seven deltas are positive and
between 56 and 385 instructions on counts up to 2.4 billion — far too small to be a timer tick
(~3,780 instructions), so it is a handful in the invocation path.

**Domain CPI, measured rather than mixed:** star 3.847, parsenumber 3.783, orm 3.416, main 3.833,
fp 4.458, cte 2.985, rtree 3.736.

**`cte` is confirmed as a real effect, not a bad prediction.** It is the only pair in §7k whose
measured cycle ratio exceeds its instruction ratio (1.166 against 1.141), and the two readings were
indistinguishable without this boot: the tick correction makes the gap worse, and the CPI route is
closed algebraically because domain-CPI over baseline-CPI *is* cycle-ratio over instruction-ratio.
Its prediction was right to 5.2e-08, so §7k's row stands.

**The denominator question is now arithmetic instead of framing.** With both arms measured on the
board, the instruction ratio can be taken against the baseline's own (ticked) count or against a
tick-free one, and the two differ by the tick:

| testset | vs BOARD baseline `instret` | vs tick-free count |
|---|---:|---:|
| star | 1.2495 | 1.3263 |
| parsenumber | 1.3536 | 1.4412 |
| orm | 1.2740 | 1.3487 |
| main | 1.1973 | 1.2698 |
| fp | 1.2769 | 1.3720 |
| cte | 1.0927 | 1.1407 |
| rtree | 1.3170 | 1.4050 |

Using the board's ticked baseline lowers every ratio by ~6%. That is the §7k tick asymmetry and
nothing else; the right-hand column reproduces the predicted ratios to three decimals, which follows
from the table above rather than being independent of it.

**THE FIRST REPEATABILITY BOUND THIS PROJECT HAS, and it is free.** sw57 measured cycles for the
same seven testsets that sw56 measured, on an image 144 bytes away and with instruction counts
differing by at most **0.00009%** — so essentially all of the cycle difference is machine rather than
program. Nothing in §7f, §7k or §7i bounded run-to-run variation before this; every ratio published
here was a single measurement.

| testset | sw56 cycles | sw57 cycles | delta |
|---|---:|---:|---:|
| star | 228,836,250 | 228,700,875 | −0.059% |
| parsenumber | 230,682,632 | 230,540,552 | −0.062% |
| orm | 937,346,376 | 937,905,097 | +0.060% |
| main | 2,675,472,428 | 2,670,465,463 | −0.187% |
| fp | 4,165,418,923 | 4,163,234,642 | −0.052% |
| cte | 6,048,111,965 | 6,027,440,964 | −0.342% |
| rtree | 9,084,912,260 | 9,042,939,578 | −0.462% |

Range **−0.462% to +0.060%**, spread 0.52 pp, population sd 0.17 pp.

**What it bounds, stated precisely: run-to-run variation AND the 144-byte layout difference
together.** With one pair per testset the two cannot be separated, so this is an upper bound on
repeatability rather than an isolate of it. **The operational consequence is that a ratio quoted to
three decimals is quoting precision nothing here supports — the second decimal is what this bound
carries.**

**A length trend is NOT claimed, and the check is the one that refuted `cte`'s explanation.** Signed
delta against run length gives Pearson r = **−0.885**, which looks like a drift; removing the two
longest arms collapses it to **−0.324**. Two leverage points, not a trend, exactly as with §7k's
CPI correlation. The same test, applied to our own result rather than someone else's.

**One pre-registered control did not transfer and is recorded as void rather than as a pass.**
`instret − mcycle = 39`, which held on all seven arms under emulation, is unobservable on silicon by
construction: `-icount` makes mcycle and minstret the same quantity, so a 39-instruction bracket
asymmetry shows; on the board cycles are real and run ~3.8x instructions, so it cannot be seen. It
neither passed nor failed here.

### §7j — Position does not move this workload, and repeatability decomposes (boot sw58, 2026-09-11)

Run on the **instrumented** image, so for the first time all seven pairs carry **cycles AND
instructions on BOTH arms in one boot** — sw56 had pairs without domain instructions, sw57 had
domain instructions without pairs. **16/16 arms, control passed, zero failures.** This section
supersedes §7k's table as the measurement of record; §7k stands as the first pairing and for the
tick analysis.

**POSITION DOES NOT MATTER, and this is the first comparison with exactly one variable in it.**
`main`'s domain arm ran twice in one boot, at position 8 and again at position 16, same image, same
memory, same allocator provenance:

| | cycles | instructions |
|---|---:|---:|
| position 8 | 2,670,343,266 | 696,765,119 |
| position 16 | 2,670,767,218 | 696,765,119 |
| difference | **+0.0159%** | **identical** |

The instruction counts being bit-identical is what makes the cycle number readable: the same program
demonstrably ran both times. Every earlier attempt at this question carried three confounds at once
(image, position, region provenance) and could not settle it. **It is settled: position is worth
0.016%, which is smaller than run-to-run variation.** Interleaving pairs costs nothing, and the
positional caveats attached to earlier cross-boot comparisons can be dropped.

**THE DOMAIN'S INSTRUCTION COUNT IS FULLY DETERMINISTIC.** All seven arms returned instruction
counts **bit-identical** to sw57's — not close, equal, across 59 M to 2.4 G instructions and two
separate boots. That is a stronger instrument check than any pre-registered falsifier asked for.

**REPEATABILITY DECOMPOSES, and the old bound was mostly not repeatability.** §7i bounded
run-to-run variation and a 144-byte layout difference together at ±0.462%, because one pair per
testset could not separate them. sw58 separates them: it re-measures the **same** images sw57 and
sw56 used, so its deltas are run-to-run alone.

| comparison | observations | range | \|max\| | sd |
|---|---:|---|---:|---:|
| **same image**, different boot (sw58 vs sw57 domain, sw58 vs sw56 baseline) | 14 | −0.065% to +0.040% | **0.065%** | **0.027 pp** |
| different image (+144 B), different boot — §7i's bound | 7 | −0.462% to +0.060% | 0.462% | 0.171 pp |

**6.4x apart. The layout difference dominated the old figure; genuine run-to-run variation is a
sixth of it.** So §7i's "upper bound, not an isolate" caveat was exactly right, and this is the
measurement that isolates it.

**Consequence for how many digits may be quoted, which cuts the other way from §7i.** A ratio pairs
two single measurements, so within one image it carries about 0.027 pp — **±0.0003 on a ratio near
1.2, not ±0.0021.** Three decimals ARE supported for a ratio measured within one boot on one image,
which is what §7j's table is. §7i's two-decimal rule was correct for the bound then available and is
too conservative for this measurement. **It still applies unchanged to any comparison ACROSS
images**, where 0.171 pp is the right figure.

**The seven pairs:**

| testset | domain cycles | baseline cycles | cycle ratio | instr ratio | domain CPI | baseline CPI |
|---|---:|---:|---:|---:|---:|---:|
| star | 228,711,030 | 182,914,247 | 1.2504 | 1.2494 | 3.848 | 3.845 |
| parsenumber | 230,632,786 | 181,904,586 | 1.2679 | 1.3535 | 3.785 | 4.040 |
| orm | 938,018,036 | 790,674,212 | 1.1864 | 1.2740 | 3.416 | 3.669 |
| main | 2,670,343,266 | 2,193,564,038 | 1.2174 | 1.1974 | 3.832 | 3.770 |
| fp | 4,162,524,273 | 3,349,560,200 | 1.2427 | 1.2767 | 4.457 | 4.580 |
| cte | 6,027,572,213 | 5,186,457,627 | 1.1622 | 1.0927 | 2.985 | 2.807 |
| rtree | 9,042,959,076 | 7,646,394,787 | 1.1826 | 1.3169 | 3.736 | 4.160 |
| **TOTAL** | 23,370,760,681 | 19,595,469,697 | **1.1930** | | | |

Instruction ratios here use the baseline's own **ticked** board count; against a tick-free count they
are 1.3263 / 1.4412 / 1.3487 / 1.2698 / 1.3720 / 1.1407 / 1.4050. The denominator question is §7i's
and is unchanged by this boot.

**The tick model holds on a third boot**, six of seven arms inside the pre-registered
0.01500–0.01520 band. `cte` is outside again at 0.014992, the same arm and the same direction as
sw56 — which is consistent with it being the emulated-tick subtraction artefact already identified
rather than anything new.

### §7k — speedtest1 on capability silicon vs native, SEVEN testsets, same boot (boot sw56, 2026-09-11)

**SUPERSEDED AS THE MEASUREMENT OF RECORD BY §7j.** sw58 re-ran these seven pairs with BOTH arms
instrumented, which sw56 could not do — its domain arms report cycles only. Cite §7j for the pair
ratios. This section keeps its value as the first pairing on silicon, as the source of the four
named asymmetries, and as where the periodic-tick analysis is derived.

**Pair ratios here carry three decimals; a comparison with another boot carries two — see the
PRECISION block at the end of this series.**

**IMAGES, added 2026-09-12 — this section cited its arms by label only, like §4g.2 did.** Read out
of sw56's own log: domain `speedtest1_seven.dom` = **`49994ed31852`**, native
`speedtest1_baseline` = **`072595ff0866`**, host **`db0c9f388980`**. **These are the images a bridge
arm must re-run unchanged**, and without them in the record no future bridge could be checked
against it — which is exactly how sw59 came to be reported as a bridge it was not (see §4g.2).
**Trip hazard, and it has already caught one boot:** the `speedtest1_baseline` currently staged in
the overlay is `24cb59fa7dbfb8fb`, which is cell ② — the **lookaside** build, not this one. Read
both hashes back from the overlay *and* the target dir immediately before a bridge boot and require
`49994ed31852` / `072595ff0866`, or the boot is void before it starts.

Same bitstream as §7f, `caplifive_r25r26r27_66c4e7517`, deliberately: seven testsets on the *same*
silicon as the three already published is worth more than newer silicon on a result that could not be
compared with sw52. 15 arms, 7 pairs, zero failures, `k800` control passed at both ends of the boot.
Floating point and rtree are restored (`SQLITE_FULL=on`), which is what takes the runnable set from
three testsets to seven. Every pair's verification hash was identical between the two arms under
emulation on the delivered artifacts.

| testset | capability cycles | native cycles | **cycle ratio** | instr ratio (predicted) | native CPI |
|---|---:|---:|---:|---:|---:|
| `star` | 228,836,250 | 182,846,119 | **1.252×** | 1.326× | 3.844 |
| `parsenumber` | 230,682,632 | 182,023,238 | **1.267×** | 1.441× | 4.043 |
| `orm` | 937,346,376 | 791,058,884 | **1.185×** | 1.349× | 3.670 |
| `main` | 2,675,472,428 | 2,193,595,600 | **1.220×** | 1.270× | 3.769 |
| `fp` | 4,165,418,923 | 3,350,136,929 | **1.243×** | 1.372× | 4.581 |
| `cte` | 6,048,111,965 | 5,186,592,912 | **1.166×** | 1.141× | 2.807 |
| `rtree` | 9,084,912,260 | 7,646,977,018 | **1.188×** | 1.405× | 4.161 |
| **total** | **23,370,780,834** | **19,533,230,700** | **1.196×** | 1.290× | 3.681 |

**The spread is the new result.** §7f had three testsets and a 1.181–1.335 range; seven give
1.166–1.267 on cycles and 1.141–1.441 on instructions. `cte` is the only pair whose measured cycle
ratio EXCEEDS its predicted instruction ratio (1.166 against 1.141), and it is also much the
lowest-CPI arm at 2.807 — and it stays unexplained after
the obvious hypothesis was tested and failed.

*The hypothesis and why it does not hold.* §7f's own mechanism is that the capability arm's extra
instructions issue in slots a memory-bound workload wastes; `cte` is the least memory-bound arm, so
it should have the fewest spare slots and the smallest advantage. That predicts
`cycle ratio ÷ instruction ratio` falling as native CPI rises. Across the seven arms Pearson r is
−0.727 — but **remove `cte` and it collapses to −0.278**. The trend is the point it was meant to
explain, which is a single leverage point and not evidence. The remaining six arms show no such
relation.

*What could still settle it, and what cannot.* Correcting for the tick makes `cte` worse, not better:
it raises every measured cycle ratio, taking `cte` from 1.166 to roughly 1.23 against its 1.141
predicted. The CPI route is closed algebraically — domain CPI over native CPI IS cycle ratio over
instruction ratio, so it cannot supply independent evidence about the same discrepancy. That leaves
two possibilities, a real effect or an error in the *predicted* domain instruction count for this one
testset, and **with no board-side domain instret they are indistinguishable**. The instrumented
domain image is the only thing that separates them.

**THE `instr ratio` COLUMN IS A PREDICTION, NOT A MEASUREMENT — sw56 carries no domain instruction
count.** The domain marker is `SPEEDTEST1-CYCLES <n> HIGHWATER n/a HEAP 2097152 DROPPED 0 RC 0`,
cycles only; `instret` appears once in the boot, in the `k800` control rung. Those figures are QEMU
`-icount` counts measured on the delivered artifacts. A `speedtest1_instret.dom` image exists and is
waiting on a boot of its own; until it runs, no statement of the form "board instructions agree with
emulated instructions" is supported for the domain arm. (An earlier version of this claim was
retracted on 2026-09-11: the apparent agreement came from a table that carried the predictions
through as if they were readings.)

**A THIRD ASYMMETRY, AND IT POINTS THE OPPOSITE WAY FROM THE TWO IN §7f.** The native arm is an
ordinary Linux process and is ticked at 100 Hz throughout; the capability arm is not ticked at all.
`handle_interrupt` (`sbi_capstone.c:1856-1861`) answers `IRQ_M_TIMER` by clearing `MTIP` in `mie` and
setting `STIP` in `mip`, and `mie.MTIP` is re-armed only by an `SBI_EXT_TIME_SET_TIMER` call from
S-mode (`:1772-1775`) — which Linux cannot issue while it is not running. So a domain arm takes at
most one machine timer interrupt and then executes tick-free for the rest of its run, up to 363 s for
`rtree`.

Measured: the native arm's board instruction count runs ~6 % above its emulated count, and the excess
tracks CYCLES rather than instructions — **3,745 to 3,787 instructions per tick across seven arms
spanning a 44-fold range of durations**, where the same excess expressed per instruction ranges over
4.39 % to 7.45 %. At 25 MHz and `CONFIG_HZ=100` that is one tick per 250,000 cycles.

So §7f's two asymmetries make the native arm look FASTER and overstate capability cost; this one
makes it look SLOWER and understates it. They are named separately rather than netted, because
netting estimates of different precision produces a number with no error bar and a lot of authority.

*What the tick would be worth if it were removed*, and this column is an ESTIMATE resting on an
assumption that is not measured — that the tick handler's CPI equals the workload's. An interrupt
handler runs cold-cache with a different instruction mix, so treat the direction as established and
the magnitude as indicative:

| testset | tick as % of native cycles | cycle ratio, tick-adjusted |
|---|---:|---:|
| `star` | 5.78 % | 1.328× |
| `parsenumber` | 6.08 % | 1.349× |
| `orm` | 5.54 % | 1.254× |
| `main` | 5.71 % | 1.294× |
| `fp` | 6.93 % | 1.336× |
| `cte` | 4.20 % | 1.217× |
| `rtree` | 6.26 % | 1.267× |
| **total** | **5.54 %** | **1.267×** |

**Two denominators are defensible and mixing them is not.** User work against user work (both arms
tick-free) and whole machine against whole machine (both as they actually run) are each honest
questions. Putting the board's native instruction count in a denominator while the capability
numerator stays an emulated count compares a ticked machine against an unticked one and calls the
difference capabilities. Which of the two the paper wants is a framing decision, not a measurement.

**A fourth caveat, new with floating point.** Both arms are soft-float — the baseline is
`-march=rv64imac_zicsr`, and its disassembly holds zero hardware FP instructions against a positive
control that finds 6 of 6 in an `-march=rv64imafdc` object — so there is no hardware-FP confound. But
the implementations differ: the baseline resolves `__divdf3` and its siblings from buildroot's libgcc,
the domain from our compiler-rt builtins at `SQLITE_SUPPORT_OPT_LEVEL`. On `fp`, `cte` and `rtree`
those routines may be a large share of the instructions. Same class as the RVC and optimisation
asymmetries in §7f; not yet quantified.

**Unchanged from §7f and still applying:** the RVC asymmetry, the optimisation asymmetry, and the
`SQLITE_GRANULE_GUARD` R-29 workaround present only in the capability arm.

**THE DENOMINATOR IS TEN, AND IT WAS NINE HERE UNTIL 2026-09-11.** This document never stated one;
it was derived from the paragraph below, which named two exclusions beside seven results. **The
omission was the claim.** `speedtest1.c` defines eleven `testset_*` functions, of which `debug1` is
self-described as "a testset used for debugging speedtest1 itself", leaving **ten benchmarks**. The
count was never taken from the source — it came from an early blocked-testset table that omitted
`trigger`, and it propagated for a month.

**What does not run, and why — three, not two.**

`json` needs a 6 MiB arena against a 4 MiB ceiling — the arena is a static array in the domain's
globals storage and `__get_free_pages` tops out at order 10 on this kernel (`MAX_ORDER` is 10 and
INCLUSIVE from Linux 6.4; `CONFIG_ARCH_FORCE_MAX_ORDER` is unset) — and it independently triggers the
S-14 pre-entry fault.

`app` enters and then takes a capability access 8 bytes off 16-byte alignment. Root-caused: SQLite's
`exprDup` sub-allocates `Expr` nodes out of one byte buffer in 8-byte steps (`ROUND8`), so a node
lands 8 mod 16 and the first capability-typed field stored into it faults. That is a **porting-cost
result** in its own right — a structure packed for an 8-byte world, addressed by a capability that
needs 16 — and it is the same family as the R-29 granule guard.

**`trigger` is broken in SQLite 3.53.3 itself and cannot run anywhere.** It was never tried until
2026-09-11, and it fails identically on the capability domain and on a stock native build of the same
source: `SQL error: no such table: t1`. The cause is in the benchmark, not in either platform —
`testset_trigger` creates `z1`, `z2` and `t3`, then its insert loop writes to `t%d` for `jj` 1..3,
i.e. `t1`, `t2`, `t3`. `CREATE TABLE t1` appears **nowhere in the file** (grep count zero); the first
two tables were evidently renamed to `z1`/`z2` without the loop being updated. So the reachable
maximum is **nine of ten**, and the tenth is an upstream defect rather than a capability limitation.
Stated for the version we pin; no claim is made about later releases.

#### TIMING — every silicon figure in this document was measured on a bitstream that does NOT meet timing, and a pre-registered criterion for that was never applied

Raised by the synth lane 2026-09-11 while handing over the R-30/R-31 bitstream. It belongs here
rather than in that handover, because it is not about that bitstream.

**The resident bitstream fails timing, and always has.** `caplifive_r25r26r27_66c4e7517.bit`:
**WNS −12.425 ns**, **102,508 of 174,960 failing endpoints**, `Timing constraints are not met.`
Every §7 row — §7a through §7k — was measured on it.

**A criterion for exactly this was pre-registered and then not applied.** `RATE-RULE.md`, quoted in
`ISSUES.md:234`:

> *"WNS non-negative makes the S-07 validation unconditional; negative means **everything measured
> on this bitstream needs re-reading**."*

It came back negative. The registry already records, for S-07, that it "was never applied". The
wording is general — *everything measured on this bitstream* — and it has not been applied to the
rest of the corpus either.

**What this does and does not mean, because the distinction decides how to read every row above.**
A timing-failing design is not wrong everywhere; it is wrong *intermittently and data-dependently*.
So it does not invalidate a number, it removes the guarantee that the number is reproducible for
reasons intrinsic to the design. That matters most for exactly the things this project measures:
the run-to-run spread quoted in PRECISION, and any single-draw result. It matters least for
instruction counts, which are architectural and were shown bit-identical across boots in §7i.

**And it is the same hazard the silicon investigation is trying to characterise** — intermittent,
data-dependent misbehaviour is indistinguishable, from a board log, between "a marginal path" and
"the defect under study". That is the strongest form of the concern and is why it is recorded rather
than noted.

**It is NOT a reason to prefer the resident bitstream over the R-30/R-31 one.** They are
timing-identical: `1bfff7776` reports the same −12.425 and the same 102,508 failing endpoints, and
the synthesis row records the pair as *"identical"*. Refusing the flash on timing grounds would keep
us on a bitstream with the same timing and without the fix. The pre-registered range −15.3 … −11.7
answered *"did the two operators move timing"* — cleanly, no — and was never capable of answering
*"is this flashable"*; the synth lane says plainly that they reported it as PASS without separating
those, and that is the right correction to have on the record.

**THERE IS A SECOND, BLUNTER CRITERION, AND IT IS IN THE SYNTHESIS SCRIPT, NOT IN `docs/`.**
`corev_apu/fpga/scripts/run.tcl` says, in as many words:

> `WNS <  0  -> DO NOT FLASH. Restore retiming or lower the clock, and say so.`

Recorded because this lane asserted to the lead that *"the repo doesn't say that"*. It does. The
search behind that claim covered `capstone/docs/` and never reached the synthesis scripts — a
conclusion drawn from a filtered view, which is the failure mode this project has a standing rule
about.

**Its scope, which is what decides whether it binds the R-30/R-31 flash.** The text appears TWICE in
`run.tcl` at `s12-ldc-rolling-filter` — `:93-99`, added by `1fc34e158` (2026-08-18) when retiming was
set **false**, and `:109-114`, added by `a3dbae618` (2026-08-19) when it was set back **true**. Both
are clauses of the retiming-OFF decision: `:93-95` reads *"If this design depends on it to meet
50 MHz, **disabling** it yields negative slack"*, and `:111-112` states the condition outright —
*"When off, this is an ACCEPTANCE CRITERION and not optional"*. Retiming is **on**
(`run.tcl:115`, `RETIMING true`), on the resident revision and on the one awaiting flash alike.

**And at the revision actually in the bitstream, the rule is not present at all.**

    git show 1bfff7776:corev_apu/fpga/scripts/run.tcl | grep -c 'DO NOT FLASH'   ->  0
    git show 1bfff7776:corev_apu/fpga/scripts/run.tcl | grep    RETIMING         ->  true   (:87)
    1fc34e158 / a3dbae618 ancestors of 1bfff7776?                                ->  NO / NO
    merge-base(s12-ldc-rolling-filter, 1bfff7776)                                ->  7e4dc440ff72

The rule commits are 2026-08-18/19; `1bfff7776` is 2026-09-10 on a line that never carried them.
The same holds for the resident `66c4e7517`.

**None of which makes the flash safe — it makes it NEUTRAL.** −12.425 ns at 50 MHz is a large, real
violation, and the RTL lane is right that no build in the eleven-build series (−10.629 … −16.400)
has ever met `WNS >= 0`, the one running the board included. Their argument was never *"run.tcl
forbids this flash"*; it was that after the 2026-09-08 census retraction **no criterion licenses any
flash on this design, the resident bitstream included** — and they themselves concede the flash
*"holds the timing risk constant rather than raising it."* That is an objection to operating the
board at −12.4 ns at all. It is the lead's call rather than a discriminator against this bitstream,
and the lead has ruled it by authorising this reflash directly.

**WHAT SURVIVES THE SCOPE CORRECTION — raised by the RTL lane, and it is the sharper form of the
concern.** `run.tcl` states a READINESS bar twice, and unlike the do-not-flash branch it is not
confined to the retiming-off case:

> `:100`  *"'Synthesis completed' is not the bar. Ready = synthesis has RUN and CLOSED TIMING."*
> `:114`  *"Either way: ready = synthesis has RUN and CLOSED TIMING. 'Completed' is not the bar."*

`:114`'s *"Either way"* is genuinely ambiguous — it can read as spanning the two WNS branches
directly above it, or as spanning the retiming on/off choice the paragraph is about. Stated as
ambiguous rather than resolved in the direction that suits the flash. On **either** reading the bar
reaches this build, because this build has not closed timing.

**AND IT CONTRADICTS `CLAUDE.md`, which carries the weaker version:** *"a hash is ready when
synthesis has RUN, not when the checks pass"* (`CLAUDE.md:310-311`) — no closed-timing clause. The
lead's own file states the rule without the clause, and that weaker version is the one consistent
with every build this project has shipped. **Which is meant is the lead's to settle; neither file has
been edited by a lane over it.**

Note what this does *not* change: the readiness bar has never been met by any build in the series,
the resident bitstream included. So it is the same objection as the RTL lane's, restated one level
up — an argument about operating this design at all, not a discriminator that separates the pending
bitstream from the one already flashed.

**What would settle it** is a timing-clean build, which is a synthesis question and not a
measurement one. Until then: quote §7 rows with this caveat attached, prefer instruction counts to
cycle counts where a claim can be carried by either, and treat any single-draw cycle figure as
carrying an unquantified intermittency term on top of the spreads in PRECISION.

#### CONFIGURATION — every speedtest1 figure in the §7 series was measured with SQLite's LOOKASIDE POOL OFF, and nobody chose that

**This is not a defect in the numbers. It is a statement about which SQLite they describe**, and it
has to be read before any of them is placed beside a result from another system.

**How it happened.** `build-sqlite-silicon.sh:1003-1006` does not own the define list — it harvests
it, as **text**, out of `build-sqlite-capstone.sh`:

```
_blocks='/^SQLITE_DEFINES=(/,/^)/p'
SQLITE_DEFINES=$(sed -n "$_blocks" "$SCRIPT_DIR/build-sqlite-capstone.sh" \
                 | grep -oE '\-[DU][A-Za-z0-9_]+(=[^ ]*)?' | tr '\n' ' ')
```

The array it reads carries `-DSQLITE_DEFAULT_LOOKASIDE=0,0`. The override that turns the pool on
sits **outside** that array (`build-sqlite-capstone.sh:158-160`), deliberately — an in-array shell
expansion is copied *literally* by the six text harvesters and reached `dev` once, breaking every
silicon domain with `use of undeclared identifier '$'`. So the placement is correct and the
consequence was unintended: the harvest can only ever yield `0,0`.

**Both arms were affected, which is what makes the ratios survive.** The native baseline had no way
to carry lookaside at all until 2026-09-11 — `build-speedtest1-baseline.sh` harvests the same block
and its compile line had no extra-defs hook. So the domain arm and its baseline were *both*
lookaside-OFF. **The pair ratios in §7f and §7k are therefore internally consistent and stand as
measured; what they are ratios OF is a configuration SQLite does not ship.**

**Why that distinction matters outside this document.** Lookaside is a small-allocation cache above
memsys5; with it off, every small allocation goes to memsys5 instead. Any claim about *where SQLite's
allocations go* is a different claim under the two settings — and the project's CheriBSD comparison
measures SQLite **as it ships, lookaside ON**, with a headline that counts slots returning to a pool
free list. That figure requires the pool to exist. **A lookaside-OFF silicon row and a lookaside-ON
CheriBSD row must not be blended into one story**, and neither is wrong for saying so.

**Fixed on both sides as of `df3c1944b47d`**, so the configuration is now a choice rather than an
accident: `SQLITE_LOOKASIDE` reaches the baseline through the same idiom the feature defines use,
and the domain takes `DOMAIN_EXTRA_DEFS='-DSQLITE_DEFAULT_LOOKASIDE=1200,40'` (verified by running
it — image `ccb73bc08db39990` reports `Successful lookasides: 25010`, not by reading the script).
**Any new row must state which setting it used.** The rows above this block were taken with the pool
OFF. **The first lookaside-ON silicon rows exist since 2026-09-14** (§7q: boot sw74's baseline and
sw75's cell 5 at size 1; boot sw77's `main --size 20` pair at 1.1937 against the OFF pair's 1.1940) —
the pool moves both arms by about 2 % and the ratio by 0.0003 at that size.

### §7l — The overhead ratio is size-robust, and the workload scales super-linearly (emulated, 2026-09-11)

**Every silicon number in §7f–§7k is at `--size 1`, and speedtest1's own default is 100.** That
makes our absolute figures incomparable to published work, which is the external collaborator's
point and a fair one. Before spending an eight-hour board pair on the default, the question worth
answering off-board was the cheaper one: *does the ratio we already publish depend on size?*

It does, slightly, and in the direction that makes the capability arm look better.

| | domain | baseline | instruction ratio |
|---|---:|---:|---:|
| `main --size 1` | 696,846,378 | 548,768,448 | **1.2698** |
| `main --size 20` | 17,755,282,312 | 14,150,699,286 | **1.2547** |

Emulated under `-icount shift=0`, so these are instruction counts. Both sizes hash equal to a
native oracle built from the same define set (`111130 1e792c9d` and `3807866 2738af78`), `DROPPED 0`.

**Emulator provenance.** Every count in this section ran on a `qemu-system-riscv64` built 2026-09-11
15:18 with `capstone-qemu` HEAD at `cabc953e58` (per the submodule reflog) — possibly already carrying
the fault-path diagnostics that landed as the merge `656cc034899f` at 15:33. The pin has since moved to
`deb7d757565d`. Across the whole range the changes are fault-path diagnostics and `cap_rev_tree.c`;
these arms fault zero times and execute zero `mrev` (verified with a positive control on the shipping image), so the counts should be
unaffected by the bump — recorded as *should*, not as measured.

**All four counts come from the SAME two binaries**, and that is load-bearing rather than tidy. The
first version of this comparison put today's 128 MiB-arena arms against the size-1 arms from the old
1.75 MiB static-heap build — which differ in memsys5 buddy-tree depth as well as in row count, so
the pair had two variables and the ratio movement could have been either. Re-measured on the new
binaries, size 1 gives **1.2698**, reproducing §7k's tick-free column to four decimals. The arena
change itself costs +0.012 % on the domain arm and +0.008 % on the baseline. **So the −1.19 %
movement is size, and only size.**

**Repeatability of an emulated count, measured rather than assumed.** The size-20 baseline arm was
run twice: 14,150,697,439 then 14,150,699,286, agreeing to 1.3e-7 relative. icount here is very
nearly but *not* bit-exactly deterministic. At ~1e-5 pp on the ratio this is four orders of
magnitude below the movement being reported, so the finding stands — but "the repeat is exact" was
written once in a draft of this section and would have been wrong.

#### The scaling law has curvature, and a two-point fit does not see it

The board-time estimate rests entirely on how the workload scales, so it was measured at four sizes
rather than extrapolated from two. The **baseline** arm carries this: it emulates ~6× faster than
the domain arm, which makes the measurement affordable.

| size | baseline instructions | × size 1 | predicted from the 1→20 exponent | error |
|---:|---:|---:|---:|---:|
| 1 | 548,768,448 | 1.00 | — | — |
| 20 | 14,150,699,286 | 25.79 | — | — |
| 50 | 40,505,714,995 | 73.81 | 38,236,033,773 | **+5.9 %** |
| 100 | 90,025,541,852 | 164.05 | 81,103,098,762 | **+11.0 %** |

The exponent is **1.0848** fitted 1→20 and **1.1075** fitted 1→100. Index maintenance is the
expected mechanism. The size-100 baseline arm verifies against the native oracle
(`23674002 573a4409`) in 157 s of emulation.

#### What this means for a size-100 board pair

At 25 MHz, from the measured silicon size-1 cycles of §7k (domain 2,675,472,428; baseline
2,193,595,600), scaled on the measured curve with size-1 CPI held constant:

| | domain | baseline | pair |
|---|---:|---:|---:|
| size 20 | 0.76 h | 0.63 h | **1.39 h** |
| size 100 | 4.79 h | 4.00 h | **8.79 h** |

**CPI is the one unknown left and it scales those hours linearly.** A 128 MiB working set on this
core will miss where a 1.5 MiB one hit, and nothing measured so far constrains by how much; QEMU
models it not at all. At 1.5× the size-1 CPI the pair is 13.2 h, at 2× it is 17.6 h. Any stage
timeout for a size-100 arm has to be set against the upper end, not the estimate.

#### What may and may not be claimed from this

**May:** the overhead ratio this project publishes is approximately size-invariant over a 20× range,
and `--size 1` is its *pessimistic* end. A size-100 measurement would confirm the ratio and apply a
correction of roughly two percent in the capability arm's favour — projected 1.2467, which the
domain arm at size 100 would replace with a measured number.

**May NOT:** that any of this is a silicon result. These are emulated instruction counts. Native CPI
for `main` is 3.769 and the capability arm's 3.840, so cycles and instructions part company on
hardware — an emulated instruction ratio cannot stand in for a measured cycle ratio, and the case
for the size-100 boot is comparability with published work rather than a correction to what we
assert.

**Precision.** These are deterministic counts to 1.3e-7, not silicon measurements, so the §7 PRECISION
block does not govern them; the ratios above are quoted to four decimals because the counts support
it. Any comparison of one of these against a *silicon* figure is a cross-instrument comparison and
gets two decimals at most.
#### PRECISION — how many digits any figure in the §7 series may be quoted to

**There are two regimes and they differ by a factor of six. Which one applies depends on what is
being compared, not on which section the number sits in.**

| comparison | what varies | sd | a ratio carries | digits |
|---|---|---:|---:|---|
| two arms within one boot (any pair ratio) | run-to-run only | 0.027 pp | ±0.0003 on a ratio near 1.2 | **three** |
| the same quantity across boots or images | run-to-run **+** layout | 0.171 pp | ±0.0021 | **two** |

Both figures are measured, not assumed. The decomposition is in §7i and §7j: sw57 against sw56
compared the *same workload* on images 144 bytes apart and gave 0.171 pp over seven observations;
sw58 re-measured the *same images* and gave 0.027 pp over fourteen. **The layout difference dominated
the original bound — genuine run-to-run variation is a sixth of it.**

**Every pair ratio in the record therefore supports three decimals**, including §7f's and §7k's. A
pair ratio is domain-arm over baseline-arm — two entirely different programs — so there is no
"two layouts of one program" term in it at all and run-to-run is the whole of the variation. An
earlier version of this block applied the 0.171 pp figure to those ratios and called them two-decimal
figures. That was too conservative, and a rule that was too conservative is relaxed out loud here for
the same reason the ones that were too permissive were tightened out loud.

**Two decimals remain right for comparing the same quantity across boots or images** — §7j's total
of 1.1930 against §7k's 1.1965 is the type case, 0.29 % apart, which is inside the cross-image figure
and outside the same-image one, exactly as it should be for two different images.

**THE PROPAGATION RULE, because three independent instances of getting it wrong happened in one
session, across two lanes.** The sd of a DIFFERENCE and the sd of a RATIO are the same number here,
because both combine two independent single measurements: if one measurement carries σ, a difference
carries σ√2 and so does a ratio. **The tabulated 0.027 and 0.171 are already difference sds. Anyone
reaching for √2 has already been given it.** Applying it a second time inflates the bound by 1.41×
and produces a figure that looks defensibly conservative and is simply wrong.

**These are bounds, not measured error bars.** One pair per testset does not give an error bar, and
the 0.171 pp figure in particular mixes a systematic — the layout difference is per-testset, not
noise — with genuine variation. If anything 0.027 pp is itself conservative for a within-boot pair,
whose two arms run minutes apart sharing thermal and DRAM state, where the fourteen observations
behind it are across boots.

*No conclusion in the record rests on a digit these bounds kill, checked rather than assumed:*
`cte` exceeding its instruction ratio is 1.166 against 1.141, a gap of 0.025; §7f's
cycle-below-instruction finding has gaps of 0.056 to 0.211; §7k's spread runs 1.166 to 1.267,
endpoints 0.101 apart. All clear even the coarser bound by an order of magnitude.

### §7m — THE BRIDGE HOLDS: §7f–§7k carry forward to the flashed bitstream (boot sw63, 2026-09-13)

**This is the arm that was owed, and it is the first time it has actually been run.** sw59 was
reported as the bridge and was not — it ran different images on both arms, which is retracted in
§4g.2. The protocol was §7k's images **unchanged**, one post-flash boot, the `main` pair's ratio
against **1.220** inside the **0.171 pp** cross-boot band.

**The images were verified byte-identical to sw56's after the bake, from both `overlay/` and
`build/target/`** — which mattered, because the overlay was holding the Sep-12 lookaside matrix
builds under the very names the bridge wanted:

| | image | sha256 (16) |
|---|---|---|
| domain | `speedtest1_seven.dom` | `49994ed3185257c6` |
| native | `speedtest1_baseline` | `072595ff0866ac4b` |
| host | `sqlite_host.user` | `db0c9f388980d86c` |

Bitstream `caplifive_r30r31_1bfff7776` — the only variable. Monitor `4274268`, firmware
`f21972371293`. 16 arms: control, seven pairs on §7k's own selector, trailing control.

Both sides are recomputed **from their own absolutes** at full precision. An earlier version of
this table compared sw63's 4-decimal ratio against §7k's *rounded* 3-decimal one, which manufactured
a uniform `+0.05 .. +0.07 pp` offset that was a rounding artefact and not a measurement; and it
marked `fp`, `cte` and `rtree` as having no §7k figure when §7k gives all seven. Both corrected
below — the section understated its own result. (Bench-lane audit.)

| testset | domain cycles | native cycles | **sw63 ratio** | §7k ratio | Δ pp |
|---|---:|---:|---:|---:|---:|
| `star` | 228,660,057 | 182,724,408 | **1.2514** | 1.2515 | −0.013 |
| `parsenumber` | 230,603,162 | 181,901,068 | **1.2677** | 1.2673 | +0.041 |
| `orm` | 937,250,292 | 791,407,931 | **1.1843** | 1.1849 | −0.064 |
| **`main`** | 2,674,517,545 | 2,193,042,052 | **1.2195** | **1.2197** | **−0.013** |
| `fp` | 4,163,178,943 | 3,347,924,554 | **1.2435** | 1.2434 | +0.015 |
| `cte` | 6,046,747,088 | 5,188,753,059 | **1.1654** | 1.1661 | −0.075 |
| `rtree` | 9,083,060,968 | 7,646,127,258 | **1.1879** | 1.1880 | −0.011 |

**All SEVEN pairs agree, worst |Δ| 0.075 pp against a 0.171 pp band**, with **mixed sign** — two up,
five down — which is what noise should look like and what the artefactual uniform `+` concealed.
`main`, the pair the protocol names, is −0.013 pp. **§7f–§7k carry forward to
`caplifive_r30r31_1bfff7776`.**

**The absolutes moved, together and one way.** Every one of the seven domain arms is *faster* on the
new bitstream, by 0.010–0.077 %, and four of the fourteen arms sit just outside §7j's 0.065 %
repeatability bound. That is the protocol's "absolutes move together, ratio holds" branch: a PASS
with a note that the flashed bitstream is uniformly ~1e-3 faster. Recorded so nobody later reads the
0.08 % as noise.

**Gates, all green.** Controls at *both* ends (`retval=4`, cycles 4476 and 4573, `instret=1089`
matching every earlier boot on this bitstream) — §7k had only a leading control; the trailing one
was added here and perturbs no measured arm. **7/7 pairs agree on their verification hash**, so
each pair computed the same answer, not merely the same number of cycles; `main` reads
`111130 1e792c9d…`, the native oracle. `DROPPED 0` on all fourteen arms. `HEAP 2,097,152` on every
arm — the compile-time 2 MiB geometry, which is *why* the next line holds.

**R-33's representability fix was inert here — by CONSTRUCTION, not by demonstration, and the
distinction is one this project keeps paying for.** Every constant on this path is a power of two
(heap 2 MiB, stack 1 MiB, region 64 KiB, hostcall 4 KiB), so the arithmetic says nothing can round,
and the capture duly contains **zero** `not representable` lines. But **that zero carries no
information on its own**: the `pr_info` at `modcapstone/module/capstone.c:279` landed the same day,
sw63 was the first boot to carry it, and it **has never been observed to fire anywhere** — so a
check that has never fired cannot distinguish "nothing rounded" from "the message never reaches the
capture". An earlier version of this paragraph said the boot *proves* the fix inert; it does not,
and saying so was this document's own "a clean result is not evidence until the check is known to
fire" rule being broken in the act of invoking it. (Bench-lane audit.) **Closed 2026-09-13 by boot
sw72:** the line fired once on a `bigregion.user 1419584` arm and zero times on the `4194304` arm of
the same boot, positive and negative control together (§7o), so the zeros above now say what they
appear to say.

**The positive control is cheap and is owed:** one `--pool`-derived arm, where 1,419,584 rounds by
1,728, shows the line once. After that a zero means something permanently. Until then the correct
reading is that the geometry is identical to §7k's *because the sizes are powers of two*, which is
independently checkable and is what actually licenses the comparison.

**What this does not license.** It re-ties the *ratios*. Cross-boot *absolutes* still belong to the
0.171 pp regime, and three of these seven testsets (`fp`, `cte`, `rtree`) have no §7k ratio recorded
here to compare against — their values are new, not confirmations.

### §7n — BOOT sw64: the size-20 rehearsal FAILED, and it failed at the 128 MiB arena's SHARE, not at entry (2026-09-13)

> **Superseded the same day by §7o.** sw64's stall was reproduced on an exact redraw (sw66), placed in
> the domain's share-entry code by the monitor's own SHA5/SHA6 definitions, and removed by deleting the
> domain's second `delin` (sw68). The "size vs per-image" framing below and the "hung inside the
> share without a cause" reading are the pre-audit picture; read §7o before citing anything here.

**No CPI was measured and no ratio exists.** The rehearsal did not complete. Recorded because the
failure is specific and useful, and because the way it cost 8 hours is a process defect of this
lane's making.

**What ran.** `caplifive_r30r31_1bfff7776`, monitor `4274268`, firmware `cadc8f2c4470`, the manifested
size-100 artifact set (`23da3b126a304585` / `ed54878519b96816` / `7c27697818b0abe0`, verified by hash
from both `overlay/` and `build/target/` after the bake). Control first: **`k800 retval=4`,
cycles 4486, instret 1089** — the boot carries a verdict. Then the `--size 20` domain arm.

**The one genuinely new positive result: a 128 MiB arena is CREATED on silicon.**
`SQ: C2/mkarena` then **`SQ: arena_bytes=134217728`**. Creation of a region that large, as SQLite's
heap, works. §7h had shown a 130 MiB region created and mapped; this shows one created *as the
speedtest1 arena*.

**Where it stopped, read from the monitor's own share trace rather than inferred.** The host prints
each marker *before* the corresponding call, so a printed marker is an attempt, not a success. The
`SHA0..SHA6` sequence (`SHA5` = about to leave M-mode for the domain, `SHA6` = the domain returned)
appears four times:

| block | `SHA1` = `region_cpmp[id]` | ends at |
|---|---|---|
| 1 | 7 | `SHA6` — returned |
| 2 → `share1`, metadata | 0x0B | `SHA6` — returned |
| 3 → `share2`, payload | 0x0C | `SHA6` — returned |
| **4 → `share3`, the 128 MiB arena** | **−1** | **`SHA5`, and no `SHA6`** |

So the monitor handed control to the domain on the arena's share and **never got it back**. That is
why there is no `SQ: G/enter`: the host never reached its own `enter` call at `sqlite_host.c:617`,
because `shared_region_annotated` at `:584` did not return. `SHA1 = −1` is **not** a fault — it means
the region is read from `regions[]` rather than a CPMP slot (`sbi_capstone.c:1749`), and an earlier
reading of it as a failure signature was wrong.

**The driver classified this as an R-16 entry stall, and that label is too coarse.** R-16 is "the
domain never ran". Here the domain *did* receive control, inside the share, and hung there. Those
need different fixes.

> **⚠ CORRECTED 2026-09-13, SAME DAY: THIS IS NOT A SIZE QUESTION, AND THE "2 MiB WORKS"
> COMPARISON WAS A MISCOUNT.** The paragraph here originally offered "size-dependent or per-image"
> and cited §7k's "three shares with a 2 MiB arena" as a working counter-example. **§7k did not have
> three shares.** `share3` is `#ifdef SPEEDTEST1_REGION_ARENA` (`sqlite_host.c:580-587`), and sw56's
> own capture reads `share1=7, share2=7, **share3=0**` — two shares per arm, no region arena at all.
> The same is true of §7m's bridge, sw63. It was a fourth, non-SQLite `SHA` block being read as a
> share. **So there is no known-good point for this path at ANY size, and sw64 was the first board
> exercise of the `REV_SHARED` region-arena share, ever.** The right third option — which the
> original paragraph did not list — is **branch-dependent**.

**The account that fits, and it is not about size.** The host shares the arena `REV_SHARED`
(`sqlite_host.c:584-586`). The monitor **de-linearises it before the domcall**
(`sbi_capstone.c:1396-1399`: `if (cap_type(r) == CAP_TYPE_LINEAR) r = __delin(r);`), so the domain
receives a **NONLIN** capability. The domain then de-linearises it **again**
(`speedtest1_measure.c:722`). On this RTL `DELIN` raises `UNEXPECTED_CAP_TYPE` for any non-LINEAR
operand, and a domain runs with **`mtvec = 0`** — so the fault vectors to pc 0, lands outside PCC and
re-faults forever, silently. `SHA5` then nothing is exactly that shape.

**QEMU cannot see this, by construction.** Its `helper_csdelin` was patched to return early when the
capability is already NONLIN, so a double de-linearise is silent under emulation and fatal on
silicon — which is why every emulated run of this configuration is green.

**The source carries the contradiction that let it ship.** `speedtest1_measure.c:705` says the
REGION_ARENA grant "arrives already-NONLIN from a REV_SHARED share, where the delin is redundant";
`:713`, *inside that branch*, says "The grant arrives LINEAR". The monitor settles it for `:705`.

**Nothing on the SHA5→SHA6 path scales with size**, which is the other half of why the size framing
was wrong. The only length-bounded loop in the monitor is the reclaim fill, gated on
`cap_type == UNINIT` and sitting *before* SHA3/SHA4/SHA5 — all of which printed. The domain side is
three O(1) instructions. No constant was found that 134,217,728 exceeds and 2,097,152 does not.

**Still N=1 and still an account rather than a reading.** The discriminator is **not** a size sweep:
an `--arena N` ladder shares `REV_BORROWED` (`sqlite_host.c:601`), which the monitor does *not*
de-linearise, so it exercises a path that cannot reproduce this fault and would return a clean sweep
proving nothing. The cheap discriminator is one arm with a trap vector installed
(`INTERP_DOMAIN_MTVEC=1`), which converts the silent re-fault into a labelled `mcause`/`mepc`.

**HOW THIS COST 8 HOURS, which is the part worth carrying forward.** `SQLITE_IDLE_S` was set to
28,800 s deliberately, to protect an arm that is *legitimately* silent for hours — speedtest1's
output goes into the shared region and prints only at the end. That same setting turned a failure
that was decidable in minutes into a full 8-hour timeout. **And the protection that should have
caught it was inert:** `ENTRY_STALL_S=420` was exported, but `ENTRY_STALL_S` is read **only** by
`board-watchdog.sh`, which this driver never started — so it did nothing. The value was copied from
earlier drivers that also never started the watchdog, where it was equally decorative.

**The fix is a separate short gate, not a smaller budget**, because the two requirements genuinely
conflict: reaching `SQ: G/enter` (or a share's `SHA6`) must be bounded in **minutes**, and only after
that should the long idle budget apply. A single idle number cannot serve both, and setting it for
the silent arm is what made an 8-hour loss out of a 3-minute answer.

### §7o — sw64's stall REPRODUCES on an exact redraw and sits in the domain's share entry; the matched pair is built and PROVEN matched (boots sw66 + sw67, 2026-09-13)

**Off-board first, and it moved the question.** Before any board time, three things were settled
under QEMU:

1. **sw65's image runs clean at 128 MiB.** The exact sw65 artifacts (`33399c3629281d88` +
   host `ec5abac09a602404`) with `--kernel-arg cma=256M`, `main --size 1 --verify`: `A/dom-ok …
   F2/share3 → G/enter → H/return`, `RC 0`, `SPEEDTEST1-CYCLES 9867027194`, verification hash
   `112006 38bb59fd…`. So sw65's `sqlite3_initialize refused` (`ENT2:5117BAD3`, a constant that
   carries no SQLite rc) does **not** reproduce off-board. It never could have been checked before
   the boot: the runner's QEMU phase (`run-speedtest1-measure.sh`) passes `cma=` only when
   `SPEEDTEST1_QEMU_EXTRA=cma` is set by hand, so sw65's build ran it without one, the host died at
   `C2/mkarena` ("above 4 MiB this needs a CMA area", `__EXIT_CODE__1`) after the image was already
   written, and the driver staged the image anyway; and `preflight-board-run.sh`'s QEMU-pass-by-hash
   gate (C13) loops over `BAKED_RUNGS` only — an `SQLITE_STAGE_DOMS` image is never checked. **sw65 went
   to the board unexercised.** (A hash trap for the record: the `main --size 1` verification hash
   is `111130 1e792c9d…` for a `SQLITE_FULL=on SQLITE_FLOAT=on` build and `112006 38bb59fd…` for the
   default `off/off` build. Both are correct; they are different programs.)
2. **sw65 was never a matched pair for sw64.** It changed three things at once: the mtvec flag,
   the feature set (sw64's image has rtree and `__extendsfdf2`, i.e. `SQLITE_FULL=on
   SQLITE_FLOAT=on`; sw65's has neither) and the source state (`dev` of 2026-09-13 06:00 UTC,
   thirteen port commits past sw64's build; the module beneath both was `8da1559`). Its result therefore said nothing about sw64's stall — exactly the "measures
   whichever difference you did not intend" failure the debugging rules name.
3. **The real pair, proven rather than asserted.** sw64's build commit was recovered from the
   worktree reflog (HEAD at the build timestamp: `cc55013c2106`), its flags read off the artifact,
   and the same recipe **without** the flag rebuilt on the unchanged toolchain binary reproduced
   sw64's artifacts hash for hash: `sqlite_silicon.dom 23da3b126a304585`, `sqlite_host.user
   7c27697818b0abe0`. Only then was `INTERP_EXTRA_CFLAGS=-DINTERP_DOMAIN_MTVEC=1` added: the pair
   image is **`214b300efd169f03`** (1,861,120 B; `.text` +0x60 with `domain_main` displaced by exactly
   that and every data/bss/gct address identical; one `csrw mtvec` site vs none; host byte-identical). Its licence to be staged is a QEMU size-1 pass at `cma=256M` with the
   oracle hash `111130 1e792c9d…`, `RC 0`. Banked with a manifest at `~/capstone-artifacts/sw64-pair/`.

**Boot sw66 — the exact redraw of sw64 (same artifacts by manifest, monitor `4274268`, module
source `8da1559`; sw64's own bake already carried both, so this is a redraw with no known
difference; `fw_payload` `ce4abf775707` vs sw64's `cadc8f2c4470` — the bake is not
byte-deterministic, the inputs were).** Control first, then the native baseline at `--size 20`,
then sw64's domain image at `--size 20` last.

| arm | result |
|---|---|
| `k800` control | `retval=4` — boot valid |
| native baseline, `main --size 20` | **RETURNED**: `TOTAL 2168.979 s`, verification hash **`3807866 2738af78`** (the size-20 oracle), **`SPEEDTEST1-CYCLES 54,230,566,323`**, `HEAP 134217728`, `DROPPED 0`. This is the size-20 denominator sw64 was meant to buy: 36 minutes on silicon, `TOTAL` = cycles / 25 MHz to four digits, i.e. the port's clock is plumbed and nothing more |
| sw64's domain image, `main --size 20` | **STALLED AT THE SAME POINT AS sw64**: `A/dom-ok, B, C, C2/mkarena, D/mapped, E/share1` (SHA0…SHA6 all present), `F/share2` (SHA0…SHA6), `F2/share3`: `SHA0:1 SHA2:0 SHA3:2 SHA4:1 SHA5:1` — and nothing after. No `SHA6`, no `G/enter`. `RCLM:00000000`; the share trace shows the arena's `:08000000`. Zero `not representable` lines |

**Reading.** Second draw of the same image, same host, same point: **the stall is deterministic for
this image, not a draw.** Where it is, the monitor says itself (`sbi_capstone.c:126-127`): `SHA5` =
"about to leave M-mode for the domain", `SHA6` = "the domain returned from the share entry". So
the core hangs **inside the domain's share-entry execution for share3** — the interp glue's `test`
entry (`stc ra`, then `RUN_CAP_INIT`, then the dispatch to the share handler's arena branch) —
with `mtvec = 0`, which is M-1's silent re-fault shape. Two corrections this forces:

* **The S-15 double delin is NOT the discriminator.** The share handler's arena branch is
  code-identical between sw64's commit and `dev` (the only non-comment drift is the Sublet blocks);
  sw65's image carries the same `__builtin_capstone_cap_delin` on the same NONLIN grant and
  **passed share3 on silicon** (`SHA6`, `G/enter`). So whatever faults in sw64's share entry, it
  is not that call.
* **What does differ is the feature set and hence the image**: `FULL=on FLOAT=on` brings rtree and
  the float builtins, 1,861,088 vs 1,684,352 B, more globals under `RUN_CAP_INIT` and a different
  layout. The pair image carries exactly that feature set plus a trap vector, installed by the glue
  **before** cap-init and on share entries too (`start-gp-captable-interp.S:779`, common to every
  entry kind) — so a trappable fault in that path returns with `mcause`/`mepc` instead of hanging.

**The watchdog, again.** The entry watchdog was wired (§7n's fix) and its detector fired in replay —
and it still could not fire live: it measured idle by driver-log growth, and the console's
`[fpga] [event] user_count` chatter (233 lines after share3, against ONE uart line) reset it every
few seconds. The stalled arm sat 23 minutes and was ended by hand. Fixed the same day
(`f7f2c9030623`: liveness is `[uart]` lines), with the positive control replayed on sw66's own log
and a negative control for an empty log. This is the third instrument on the same 3-minute
question that read as working and was not.

**Boot sw67 — the pair image (`214b300efd169f03`), control first, `--size 20` last.** Control
`retval=4`. The domain arm: `E/share1` and `F/share2` as before, then **`F2/share3 → SHA5 → SHA6`**
— the pair image **passed share3**, where its twin without a trap vector hangs — then `G/enter`,
`H/return`, **`obs = 0x5117BAD3`** (`SQLITE_HC_ERR_INITIALIZE`), "speedtest1 did not run". No trap
marker in `obs` (top nibble `0x5`). That is sw65's result, now on an image that differs from sw64's
in exactly one variable.

**The reading — an account whose mechanism is traced through the RTL, with its confirming
measurement still owed.** The trap handler exists for one discriminator
(`start-gp-captable-interp.S`, at `.Ldomain_trap`): *if a build with this handler RETURNS where
the build without it HANGS, the domain faulted.* sw66 hangs at share3; sw67 returns from share3.
That is an elimination argument, and an audit of it (2026-09-13, evening) found it broader than its
evidence: the handler has never been seen to fire on this bitstream (`board-b65.sh` said so before
its boot), and share3 differs from shares 1 and 2 in **three** respects, not one — the domain's
delin, a table-resident region (`SHA1:FFFFFFFF`) where shares 1/2 are CPMP-resident, and 128 MiB
against 64 KiB. What the audit *did* establish, at the bitstream's own RTL commit `1bfff7776`:
`func DELIN` raises `UNEXPECTED_CAP_TYPE` for any operand that is not LINEAR (mcause 27 — DYN base
24 + code 3, so it is trappable, unlike R-24's base), the two `lcc` selectors in that branch are
legal on a NONLIN operand, so **the delin is the only type-sensitive instruction in the share3
handler**; and the monitor at `4274268` really does hand the domain a NONLIN grant
(`sbi_capstone.c:1396-1408`; the trace shows `AREV:00000002`, `SHA2:00000000`). Why `obs` shows no trap is in the handler's last
three instructions — `ldc(a0, sp, 80); sw t0, 0(a0)`: it writes its packed `mcause`/`mepc` word
*through the region capability saved at entry*, which for a share entry is the shared region
itself — for share3, the first word of the 128 MiB arena, which nothing reads. The share entry then
takes the shared unwind and `domreturn`s; the monitor prints `SHA6` (its value is `dom_id`, not a
return code) and the host proceeds to `enter`. So *if* the discriminator fired, its word landed
where no instrument looks — and sw65's "did not confirm" was this same gap, not a negative. The
reading that names the site is the host printing `arena[0]` after share3 returns, with the pair
image unchanged: **predicted `0xF6C09D13`** (marker 0xF, mcause 27, offset of the displaced delin
`(0x373ec + 0x60 − 0x10000) >> 2`); any other 0xF word names the real site; no 0xF/0xE word means
the handler never ran and the mtvec correlation needs another explanation. Owed to the next boot.

Downstream is then forced by SQLite itself: the arena branch's assignment `sqlite_arena = …` never
executed, so the domain calls `sqlite3_config(SQLITE_CONFIG_HEAP, NULL, 0, 64)`; SQLite's
documented behaviour for a NULL heap pointer is to *revert to its default allocator*
(`sqlite3.c:187958`, `memset(&sqlite3GlobalConfig.m, 0, …)`), and this domain's default `malloc`
is the freestanding stub that returns `NULL` (`adapted/capstone_sqlite_libc.c:116`) — so
`sqlite3_initialize()` fails and the port returns `SQLITE_HC_ERR_INITIALIZE`. One mechanism accounts for
four boots: **S-15's double delin faults on silicon** — as an account with its mechanism sourced, not
yet as a measured root cause. With `mtvec = 0` (M-1) the fault re-faults
forever and reads as a hang at share3's `SHA5` (sw64, sw66); with the trap vector installed the
share entry returns and the run dies at `sqlite3_initialize` (sw65, sw67). QEMU cannot see any of
it because its `helper_csdelin` returns early on a NONLIN operand.

(All four boots ran with `FPGA_BITSTREAM_UNVERIFIED=1`; the bitstream is asserted, not re-read,
and the same assertion holds across the pair, so the comparison stands.) Two things follow. The fix
is the one the S-15 entry named before sw65 clouded it: the domain
must not delin a grant the monitor has already de-linearised (`speedtest1_measure.c`, the
REGION_ARENA branch; the comment that said "the grant arrives LINEAR" was the error). And the
instrument has a hole to close: a trap taken in a *share* entry must be reported somewhere the
host reads, not written into the shared region. **The proof of the fix is a fifth boot**: the
fixed image, without a trap vector, must pass share3 and run — pre-registered as sw68 below.

**Boot sw68 — the fix, one variable against sw66.** The fix image `f795151f3ed4883b` is built
from `dev` with sw64's flags and **no trap vector** (`csrw mtvec` sites: 0; the boot's driver
refuses an image that installs one), same size as sw64's (1,861,088 B). Its instruction stream is
sw64's **minus exactly one instruction** — the `delin a2` at #40,190 of 373,094; after normalising
the 4-byte shift of every later PC-relative immediate, zero other instruction positions differ
(delin sites 2 → 1). It is staged with sw64's
**own** host and baseline by content (`7c27697818b0abe0`, `ed54878519b96816`) — so the set differs
from sw66's in the removed delin and nothing else. Its QEMU licence at `cma=256M`: hash
`111130 1e792c9d…`, `RC 0` (QEMU proves the image runs, not that the fix works — it cannot see the
fault). Pre-registered before the boot: **`share3 → SHA6, G/enter, H/return`, hash
`3807866 2738af78`** = the delin was the fault (the region's residency and size are unchanged in this image, so a
pass exonerates them as the sole cause) *and* the fix works on silicon, with the domain arm's cycle
count as the size-20 measurement against the baseline arm in the same boot — pre-registered
numerically: the silicon ratio at size 1 is 1.2195–1.2201 (§7k, §7m) and the emulated §7l says the
ratio is size-robust, so against sw66's baseline of 54,230,566,323 cycles the domain arm is
predicted at **≈66.1 G cycles (ratio ≈1.22, ~44 min at 25 MHz)**; a ratio outside 1.17–1.27 is a
finding about scaling, not a pass; **`SHA5` last again** = the
fault is not the delin but something else in the share3 handler (the `get_base`/`get_end` reads on
the NONLIN grant are the only other type-sensitive operations), and the next instrument is a host
that reads the trap word back out of the arena's first word after an mtvec build; any other `obs`
after `G/enter` = a different, real result. Control `retval=4` or the boot is void.

**Boot sw69 — the reading the audit asked for: the trap word read back. An instrument arm, not a
pair member:** its host is `dev`'s with the readback compiled in, not sw64's, so it is judged on the
`arena0` word alone. The pair image **unchanged** (`214b300efd169f03`) with a host that maps the arena *after* share3 returns and
*before* the domain is entered, and prints its first word (`sqlite_host.c`, `SPEEDTEST1_ARENA_READBACK`,
off by default; host `ae9e0ba31fa4e634`; its QEMU plumbing control prints `SQ: arena0=0`, the value
no trap can leave). `--size 1`, so the arm returns in seconds either way. Pre-registered from the
glue's packing and the RTL at `1bfff7776`: **`arena0 = 4139621651` (`0xF6C09D13`)** = marker 0xF,
mcause 27 (`UNEXPECTED_CAP_TYPE`), offset `0x9D13` = the pair image's delin at
`(0x373ec + 0x60 − 0x10000) >> 2` → S-15 as claimed, site named; any other word with top nibble
0xF/0xE names a different site at `_start + 4·(word & 0x3FFFFF)`; `0` means the handler never ran and
"returned ⇒ trapped" is void; `UNMAPPED` is an instrument fault. Together with sw68 this closes the
audit's two open alternatives from both sides: sw68 removes the delin and keeps the region's
residency and size; sw69 reads the fault's own address. 

**Boot sw69 — RESULT.** Control `retval=4`. The readback arm returned **`SQ: arena0=4139818259`**,
which is **`0xF6C09D13`** — bit-for-bit the pre-registered word (the decimal `4139621651` written
above and in the driver header was a mis-decimalisation of that same hex; `0xF6C09D13` is
`4139818259`). Its **mcause field (bits 27:22) is 27** = DYN base 24 + code 3 = `UNEXPECTED_CAP_TYPE`,
exactly the trap the RTL raises when `func DELIN` gets a non-LINEAR operand. Read from the run-scoped
`boot.txt` (line 128), immediately after `F2/share3` and **before** `G/enter` — so the trap word was
written during the share3 domcall, which is where sw64 and sw66 hang, not in the main entry. After
it, the same host printed **`SQ: obs=1360509651` = `0x5117BAD3`** = `SQLITE_HC_ERR_INITIALIZE`, the
initialize failure sw65 and sw67 died on. So the readback names S-15's fault on silicon end to end:
the domain's redundant delin at share3 raises `UNEXPECTED_CAP_TYPE`, the glue's handler packs the
cause into the shared arena's first word, and the run then fails exactly where the trap-vector boots
did. The mcause is the audit's owed reading; the exact `mepc` offset is `word & 0x3FFFFF` and lands
in the pair image's arena-branch delin as predicted.

**Boot sw68 — RESULT (the fix side).** Control `retval=4`. Baseline arm: hash `3807866 2738af78`,
**54,214,856,567 cycles** (sw66: 54,230,566,323 — 0.03 % apart; the size-20 denominator is
repeatable). The fixed domain arm: `E/share1` SHA5→SHA6, `F/share2` SHA5→SHA6, **`F2/share3`
SHA5→SHA6**, then **`SQ: G/enter`** — the image that is sw64's minus one instruction, with no trap
vector, **passes the share on which sw64 and sw66 hang** and enters. Region residency and size are
unchanged in this image, so of the audit's three differences only the delin was removed and only
the outcome changed. The domain arm then ran size-20 to completion: hash **`3807866 2738af78`**
(the same size-20 oracle the baseline verified to, so the domain computed the identical program),
**64,732,455,367 cycles**, **ratio 1.1940** against the baseline denominator above. That is inside
the pre-registered band (1.17–1.27, predicted ≈1.22) written down before the boot, so the capability
overhead on the full workload is 1.19x native and the S-15 fix runs the real benchmark end to end on
silicon, not merely past the share. `H/return` and the verification hash both present; the arm was
judged on the hash, not on completion alone.

**2026-09-13 late, before the size-100 pair (boot sw73): the pre-run and the controls.**
*Size-100 pre-run on QEMU `-icount`* (fix image `f795151f`, the readback host's predecessor
`6e3cd65c`, `cma=256M`): `--size 20` → hash `3807866 2738af78`, **17,755,282,312** instructions
(§7l's count, reproduced); `--size 100` → hash **`23674002 573a4409`** (the native oracle),
**111,482,174,837** instructions — the domain arm at size 100 had never run anywhere before this;
its instruction ratio against §7l's baseline count 90,025,541,852 is **1.2383** (the projection was
1.2467); HEAP 134217728, DROPPED 0, RC 0. On sw68's measured CPIs (native 3.831, domain 3.646) the
board pair projects to 3.83 h + 4.52 h; the pre-registered cycle ratio for sw73 is ≈1.18, band
1.14–1.24, two decimals.
*Boot sw72, the controls boot* (control `retval=4`): **R-33's rounding log fired for the first time
anywhere** — `bigregion.user 1419584` (rebuilt to count the module's "not representable" line in
dmesg) reports `not-representable-lines=1`, and the representable `4194304` arm reports `0`, positive
and negative control in one boot; every earlier "zero rounding lines" reading is now informative.
**The S-15 instrument is fixed and controlled:** the measurement host now reads every REV_SHARED
region's first word back after the share (default-on; a planted sentinel in the arena makes
"unchanged" a positive reading). Negative control on QEMU (fix image): no trap line, sentinel
`0x5EED0000` = 1592590336 unchanged, oracle hash. Positive control on silicon (pair image
`214b300efd169f03`): `share-trap=4139818259` = `0xF6C09D13`, mcause 27, offset 160844, at share3, then
`X/fail obs=3` and no `G/enter` — sw69's reading, now produced by the measurement host and terminal.
Scope: this names a trap only for a domain that installs a trap vector; a measurement image (mtvec
0) never returns from the faulting share, and the entry watchdog is what bounds that case. (Two
pre-registration decimals in the drivers were mis-converted by hand — `0xF6C09D13` and `0x5EED0000` —
and both hardware readings matched the hex exactly; decimals are now produced by machine.)

### §7p — `main --size 100` ON SILICON: the pair completes, ratio **1.18**, exactly the pre-registered value (boot sw73, 2026-09-14)

**Setup.** One boot, driver `board-b73.sh` (bare launcher, pre-registrations in its header, `BUDGET=43200`
per arm, entry watchdog on), image f795151f3ed4883b (the S-15 fix image, `mtvec = 0`, staged by hash —
the same bytes as sw68's), measurement host 2af56927aaf907e9 (the share-readback build validated on sw72
and on QEMU), native baseline ed54878519b96816 (sw64's, 128 MiB heap), control `k800`, the #3 module in
the initramfs. Arms in order: control → `speedtest1_baseline warm --testset main --size 100 --verify` →
control → `speedtest1.dom --speedtest1 --testset main --size 100 --verify` LAST. Launched 00:11
(bake waited for the machine lock), booted 00:19, driver rc=0 at 08:53. Readings from the run-scoped
driver log after this run's own `load_image`, and the driver's own end summary agrees line for line.

| arm | hash | `SPEEDTEST1-CYCLES` | wall (speedtest1 `TOTAL`) | `HEAP` | `DROPPED` | `RC` |
|---|---|---:|---:|---:|---:|---:|
| control (before / after baseline) | `RESULT k800 retval=4` / `retval=4` | — | 1 s / 1 s | | | |
| native `warm`, size 100 | **`23674002 573a4409`** | **337,235,381,252** | 13,488.6 s = 3.75 h | 134217728 | 0 | 0 |
| domain, size 100 | **`23674002 573a4409`** | **398,572,346,349** | 15,941.9 s = 4.43 h | 134217728 | 0 | 0 |

**Ratio 398,572,346,349 / 337,235,381,252 = 1.1819 → 1.18 at two decimals.** Pre-registered before the
boot: ≈1.18, band 1.14–1.24 (§7l's projected instruction ratio 1.2467 × sw68's CPI ratio 3.646/3.831).
Both arms hash to the native oracle, so the ratio is of one program producing one answer.

**Decomposition** (instruction counts from the QEMU `-icount` pre-run of the same two binaries, §7o:
native 90,025,541,852, domain 111,482,174,837): instruction ratio **1.2383**; CPI native
337.24 G / 90.03 G = **3.746**, domain 398.57 G / 111.48 G = **3.575**, CPI ratio **0.954**; and
1.2383 × 0.9545 = 1.182, the measured cycle ratio. The domain retires 24 % more instructions and
runs them at a 4.6 % lower CPI, as at size 20 (sw68: 1.2547 × 0.952 = 1.194). The board's own native
instret, `BASELINE-WARM INSTRS 95,113,570,125`, is 1.0565× the icount (the same +6 % board-vs-icount
offset §4g records: kernel work is inside a native process's count and outside a domain's); the
domain arm is the cycles-only image and prints no instret by design (§7l).

**Size robustness on silicon, three points now:** size 1 **1.2195** (sw63, §7m), size 20 **1.1940**
(sw68, §7o), size 100 **1.1819** (this boot) — the ratio falls slowly with size, and the emulated
instruction ratios fall the same way (1.2547 at 20, 1.2383 at 100, §7l/§7o). Cost of the pair: 8.18 h
of core time + one boot, against the 8.4 h derived in the plan.

**The five invalidators, checked before the ratio was written:** (1) both hashes equal the size-100
oracle — yes; (2) `DROPPED 0` on both arms — yes; (3) the count is ~164× size 1, not 100× — the
native board instret is 164.4× §4g's size-1 board instret (578,533,909) and 1.0565× the size-100
icount, consistent; (4) NOMEM, the named hours-in failure (128 MiB built against 120 MiB measured
need) — absent: `RC 0` and the oracle hash on the domain arm; (5) `capinit-reload-scan` — satisfied by
staging the scanned image by hash (the driver's manifest check). Also read: the module's rounding line
fired 0 times (informative since sw72's positive control); the share-readback instrument printed no
`share-trap` line and read the arena sentinel back unchanged (`SQ: arena0=1592590336` = `0x5EED0000`)
before `G/enter` — the negative reading on silicon for a measurement image; all three shares returned
(`SHA5`→`SHA6`); watchdog never fired.

**What it may be quoted as, and its caveats.** The capability domain runs SQLite's `speedtest1 main`
at its default size on this silicon at **1.18× the native cycle count**, whole program, one boot,
N = 1 per arm (§7j's run-to-run spread on the same image is 0.027 pp, so two decimals are honest).
Every caveat bound to the §7 rows applies: the bitstream does not meet timing (§7 TIMING block);
lookaside OFF on both arms (§7 CONFIGURATION block — this image is built by the same path, no
override); memsys5 in a 128 MiB region arena on both arms (arena-matched); cycles only, at 25 MHz.
Not comparable to the §4g emulated matrix (different heap geometry and vehicle).

### §7q — Boot A of the 2026-09-14 plan (boot sw74): the Sublet cell re-based on the #3 module, the lookaside-ON baseline, and the R-30 reclaim filling at the rounded arena

**Setup.** Driver `board-b74.sh` (bare launcher, pre-registrations in its header), the #3 module
(d04bd83) and monitor 4274268, control `k800`; host `sqlite_host_rr.user` 2c9e82d101b48160 — the
default-mode SQLite host rebuilt against the merged libcapstone with `-DSQLITE_HOST_REVOKE_RESHARE=1`
(it runs the revoke-reshare probe on every teardown) and `-DSQLITE_HC_REGION_SIZE=65536` (the first
rebuild without that define passed the string gate and failed both images at the region check on QEMU;
the pair is what caught it) and proven as a QEMU pair; cell 6 image `ceeded2533a74bce` (sw61's bytes,
reproduced byte-identical by the measure script on the #14 toolchain, pass record written by that run);
the lookaside-ON native baseline `d95dd98c0de73c68` (cell 2's recipe rebuilt). Boot 12:44, driver rc=0
13:12; eight arms staged, four ran.

| arm | what | reading | pre-registered |
|---|---|---|---|
| 1 | control | `RESULT k800 retval=4` | 4 |
| 2 | cell 6 (Sublet, lookaside ON) `--size 1`, `--arena 1419584 --tables 1750285` | `112006 38bb59fd`, **`HEAP 911104`**, **2,794,183,730** cycles, `DROPPED 0 RC 0`, `sublet: split=5481 mrev=37874 delin=32565 …` | oracle; 911104; ~2,795.7 M ±1 % (sw61: 2,797,516,229 at HEAP 910008); the QEMU counters |
| 2, teardown | the revoke-reshare probe on the 1,419,584 request (the arm sw60/sw62 wedged on) | the module creates the region at `ALEN:0015B000` = 1,421,312 (the R-33 rounding, executed); `SQ: released pool rc=1`, `RR/share-A`, `RR/share-B`, `RR/probe-region=22`, `RR/done`; **RCLM 00000000 → 00000001 around the pool-release ecall**; **no RCSH, no RCPR, no RCRE** | exactly this; sw60 read `RCSH:000006C0` here |
| 3 | native baseline, lookaside ON, `warm --size 1` | `112006 38bb59fd`, **2,108,202,651** cycles (`BASELINE-WARM 2,108,267,102`), `HEAP 2097152` | oracle; ~2,107.5 M ±1 % (sw60's cell 2: 2,107,533,496) |
| 4 | cell 6 `--size 1 --stats` — the SAME image's second run | `A/dom-ok`, four shares returned, `G/enter`, then **no return in 900 s**; trap latch 0x89 (a stale ecall, no capability trap latched); no monitor tag | return + `Successful lookasides: 25010` |
| 5–8 | baseline `--stats`, control, probe @4194304, probe @1419584 | not run (a wedged domain takes the core) | — |

**Readings.** Cell 6 on the #3 module runs 0.12 % faster than sw61's reading and reports the rounded
arena's `HEAP 911104`; its counters equal today's QEMU run's, as sw61's equalled theirs. The
lookaside-ON baseline reproduces sw60's cell-2 reading to 0.03 %. **The reclaim of the rounded arena completes through csinit on this module + monitor, twice** (sw74
arm 2's teardown and sw74b arm 3's): RCLM climbs 0 → 1 and none of the three reclaim traps (RCPR, RCSH,
RCRE) fires. **What that is and is not (claim-auditor, 2026-09-14):** the "shortfall" sw60 and sw62
reported was never a short fill — it was R-33's bounds re-encoding, and ISSUES already says so
(sw62's `RCEN:00056C00 / RCCU:00056A40 / RCSH:000001C0`: the cursor reached the full 354,880 while the
re-encoded `end` read 448 high, exactly `round_up(354880, 512) − 354880`; sw60's 1,728 is the same gap
for 1,419,584). Two things changed between those boots and this one: the module now rounds the request
before the monitor sees it (`ALEN:0015B000`), and the monitor moved 2c49c41 → 4274268, which changed
what RCSH measures and added RCRE for the re-encode gap. **The pass is therefore inferred to be the
rounding's doing, not controlled:** RCRE has never been observed to fire, and the one arm that would
settle the attribution — monitor 4274268 with a pre-rounding `.ko` and the same request, predicted
`RCRE:000006C0` + wedge — has not been run. The `--tail` probe arms of this boot (the representable
4,194,304 control and the 1,419,584 repeat) were lost to arm 4's wedge; sw74b carries the control.
Also a pre-registered miss: the module's "not representable" line is a `pr_info` (dmesg), so it is not
in the UART capture; `ALEN` is the evidence the rounding ran. Recorded under R-33 in ISSUES, with a
pointer from R-30.

**⑥/⑤ on silicon is not re-based by this boot:**
cell 5's #3-module run is Boot B.

**The arm-4 wedge is R-12's budget, not a new defect — settled by boot sw74b (13:17–13:45, same
firmware set, fw `17112c3863fb`).** sw74b ran the same cell-6 image with `--stats` as the FIRST domain
run: it returned — `112006 38bb59fd`, `HEAP 911104`, **`Successful lookasides: 25010`** on silicon (the
lookaside-ON positive check for the Sublet cell), the probe to `RR/done` with RCLM 0 → 1 and no trap,
plus the native baseline with `--stats`: `Successful lookasides: 25122`. Then the plain run as the
SECOND domain run entered (`A/dom-ok`, four shares with `ALEN:0015B000`, `G/enter`) and wedged
exactly as sw74's arm 4 had. **Both wedge reads show `rev_node_head[15:0] = 0xFFFF`** — the 16-bit
bump head at its sentinel, R-12's deliberate stall on rev-node exhaustion. The arithmetic: one Sublet
run mints `split + mrev` = 5,481 + 37,874 = 43,355 nodes and nothing reclaims them (R-12: no
reclamation), so a second run crosses 65,535 mid-workload. The `--stats` path was never the
variable. **Rule for every driver from here: at most ONE Sublet-cell workload per boot; a memsys5
cell's mints are its boundary borrows and shares, far below the pool (sw73 ran size 100 in one boot),
so its second run is safe.** The two `--tail` probe arms and the trailing control were lost in both
boots; the representable-arena control moves to Boot B's last arm.

**Boot B (sw75, 13:48–13:59, fw `10d84925dfe2`): cell 5 on the #3 module, its `--stats` re-run, and the
representable-arena probe control — all five arms returned.**

| arm | what | reading | pre-registered |
|---|---|---|---|
| 1, 3 | controls | `retval=4`, `retval=4` | 4 |
| 2 | cell 5 (memsys5, lookaside ON, 2 MiB static heap; `e6ee5255c896aa21` = the archived recipe + the stack declaration 385,024) `--size 1` | `112006 38bb59fd`, `HEAP 2097152`, **2,551,483,818** cycles, `DROPPED 0 RC 0` | oracle; ~2,551.5 M ±1 % (sw60: 2,551,506,640) — **−0.0009 %** |
| 4 | the same image, second run, `--stats` | `112006 38bb59fd`, **`Successful lookasides: 25010`**, 2,552,134,789 cycles (+0.026 %, the statistics print) | returns (a memsys5 cell mints only shares and boundary borrows) |
| 5 | probe, `--tail --arena 4194304 --tables 1750285` (the control Boot A lost) | arena `ALEN:00400000` (no rounding, a power of two), tables `ALEN:001AB800` (rounded), `RR/share-A … RR/done`, **RCLM 0 → 1**, no RCPR/RCSH/RCRE, then the designed end (`released tables rc=1`, `obs=0x5117BAD4`) | exactly this |

**Readings.** The static-heap cell is unmoved by the module change and the declaration knob (the loaded
bytes are the same; the block is order 10 either way): −0.0009 % against sw60. The re-based silicon
Sublet-over-memsys5 ratio on the #3 module is **⑥/⑤ = 2,794,183,730 / 2,551,483,818 = 1.0951** (sw61/sw60
read 1.0964) — still the two-geometry "configuration" comparison (§4g.5's caveat stands: ⑥'s arena is
the rounded 1,421,312 in a shared region, ⑤'s is a 2 MiB static heap). The second run of the memsys5
cell returning is the negative control for the R-12 account of the Sublet wedges: it is the Sublet
cell's 43k mints per run, not "a second run", that exhausts the pool. The representable-arena probe
reclaims exactly as the rounded one did, so nothing in the reclaim path depends on the arena having
been rounded — the attribution question in ISSUES R-33 (module rounding vs the monitor's new check)
stays as the auditor left it, and its settling arm stays owed.

**Boot C (sw76, 14:08–14:21, fw `f80c301f8cd2`): the entry watchdog's live positive control — it fires.**
Control `retval=4`, then sw64's image `23da3b126a304585` (the pre-fix image that stalls at share3, the
readback host 2af56927, `--size 1`) last. Reading, all as pre-registered: `Domain requirement =
1281952 (stack 1048576)`, `A/dom-ok`, share1 `SHA5 → SHA6`, share2 `SHA5 → SHA6`, share3 (`ALEN:08000000`,
the 128 MiB arena) `SHA5:00000001` and nothing after — no `SHA6`, no `G/enter`, no share-trap line
(the readback runs after a share returns; this one never does — the instrument's scope, as stated).
The watchdog: **`ENTRY-STALL 781s  last share marker=SHA5:00000001, no SHA6 for 421s -> domain never
ran; the rest of this boot is worthless. Aborting runner.`** — 421 s after the last UART line, exactly
the configured `ENTRY_STALL_S=420`; it sent `TERM`, the runner ended (rc 130, its own handling of the
signal), ran its release (`gdb_stop`, power off), and the run-scoped log carries exactly one boot banner
after this run's `load_image` — no reset in the window. This is the first time the abort has fired on a
live board; every earlier "it would have caught sw64" was a replay. A first attempt of this boot was
blocked by the preflight's control-record check (`first rung 'k800' is not listed …`) although the
same check had passed on sw74/74b/75 minutes before and passes on re-run. Ruled out afterwards: the
controls file is unchanged since 2026-09-12 (content and mtime), the HEAD and stash reflogs of the shared
checkout record no git operation between 13:46 and 14:23, the driver's stderr carries no fork or exec
error, and the check re-run from the wrapper's cwd two minutes later matched `k800`. Unexplained; the one
hypothesis the evidence cannot exclude is a momentary rewrite of the file by the other lane that works in
this checkout. The boot cost nothing but a bake.

**Boot D (sw77, 14:24–17:53, fw `a6458cb568fe`): the lookaside-ON pair at `main --size 20` — SQLite as it
ships, on silicon.** Domain `90ef29431abdbdee` (sw64's recipe: FULL/FLOAT, 128 MiB region arena,
`mtvec = 0`, the S-15 fix, plus `SQLITE_DEFAULT_LOOKASIDE=1200,40`), host 2af56927, native
`78e523cb05cbb91f` (sw64's baseline recipe plus lookaside). Both images QEMU-passed at size 20 before the
boot (§4g's lookaside-ON rows).

| arm | what | reading | pre-registered |
|---|---|---|---|
| 1, 3 | controls | `retval=4`, `retval=4` | 4 |
| 2 | native, lookaside ON, `warm --size 20` | `3807866 2738af78`, **53,142,976,993** cycles (2,171 s) | oracle; ~53.65 G ±3 % |
| 4 | domain, lookaside ON, `--size 20` | `3807866 2738af78`, **63,437,512,052** cycles, `HEAP 134217728 DROPPED 0 RC 0`, no share-trap line | oracle; ~63.97 G |
| 5 | native, `--size 1 --stats` | `111130 1e792c9d`, **`Successful lookasides: 25122`** | lookasides > 0 |
| 6 | domain, `--size 1 --stats` — the image's SECOND run in the boot | `A/dom-ok`, regions 1 and 2, `C2/mkarena`, then nothing for 7,200 s: no `D/mapped`, no shares; wedge read: trap log 0x83, `rev_node_head` 0x0121 (289 nodes, not R-12) | lookasides > 0 (lost) |

**Ratio 63,437,512,052 / 53,142,976,993 = 1.1937 → 1.19 at two decimals**, the pre-registered value
(band 1.15–1.24). sw68's lookaside-OFF pair of the same size and image family read 1.1940: the pool
changes the ratio by 0.0003 and takes 1.98 % off the native arm's cycles and 2.00 % off the domain's.
Decomposition on the QEMU icount counts of the same binaries (native 14,003,848,190, domain
17,544,109,561): instruction ratio 1.2528, CPI native 3.795 / domain 3.616, CPI ratio 0.953.
**This closes the CONFIGURATION block's gap: a silicon speedtest1 row with the lookaside pool ON now
exists**, and it says the same thing the OFF rows said. The domain image's lookaside is proven on
QEMU (`Successful lookasides: 25010` at size 1) because its silicon `--stats` arm was lost to the wedge
below; the native binary's is proven on silicon (arm 5).

**Arm 6 is a third second-run shape, and it is not R-12.** The image's second run stopped after
`SQ: C2/mkarena` — the host creating its second 128 MiB arena region of the boot — with no domain
marker after it, no capability trap latched and the rev-node head at 289. The REGION_ARENA host does
not release its arena at teardown (only the static-heap pool/tables paths call `release_region`), so
the second `create_region` of 128 MiB is asked of a 256 MiB CMA area that still holds the first;
whether that allocation blocks or the ioctl wedges is not established (N = 1, host-side, no UART). Not
M-7 (no `release_region` preceded it) and not the Sublet cell's node budget. The runner's own summary
line for this arm reads "R-16 entry stall; markers stop at SHA5 with no SHA6": that was the fixed text of
its never-entered branch, not a reading — the transcript carries no share tag at all — and the raw
`driver.log` keeps it. The branch now keys on the transcript's last share tag (`stall_shape`, self-tested
on this arm's and sw76's real transcripts) and prints a host-side stop as one. **Driver rule, added to
the state doc: a REGION_ARENA image runs ONCE per boot**; its `--stats` positive check belongs on
QEMU or in a boot of its own. Placed last by design, so the pair cost nothing.

**Where the size-20 numbers now stand on silicon** (same image family, same bitstream, cycles):
native OFF 54,214,856,567 (sw68) / ON 53,142,976,993 (sw77); domain OFF 64,732,455,367 / ON
63,437,512,052; ratios 1.1940 / 1.1937. Every caveat of §7p applies (timing not met, cycles only, N = 1
per arm).

### §7r — E1 of the Sublet-paper plan: the S1/S2 safety matrix on hardware, repetition 1 (boots sw78 r1b1–r1b5, 2026-09-14 evening to 2026-09-15 00:1x)

**What ran.** The three probe families of the collaborator's `s2/3-manager-hierarchy` branch (`d5954cc0e7e7`),
built in a detached worktree of that branch with one addition, a `DOMAIN_BASE_VA` knob for
`build-nginx-domain.sh` and the PostgreSQL `pgdom_link` (`3a7be75437e9`), so that several probe images
share one boot at distinct entry VAs (R-3, preflight C15). Every image ran on capstone-qemu first at its
board VA (the run-scoped log is its pass record; the s9 cell aborts the emulator, capstone-qemu PR #4, and
is board-only). The 16 nginx cells are the gate `run-nginx-uaf.sh` defines: one image per compile-time
stop, 13 returning a mark (`NGX_DOM_MARK`, low 24 bits of `ngx retval`) and 3 expected to fault. On this
RTL a capability fault inside a domain is a wedge (M-1), so each FAULT cell is the last arm of its boot and
is read from the debug mux's trap log. Control `k800 = 4` opened every boot; one boot banner after each
run's `load_image`; bitstream `caplifive_r30r31_1bfff7776`, monitor `4274268`, module `d04bd83`.

**Two facts about the harness found on the way.** (1) The entry VA has a ceiling on QEMU: an nginx image
linked at `0x610000` never reaches the guest shell, at `0x410000` and below it runs; the slots are now
512 KiB apart from `0x10000` (`0x10000 … 0x310000`). (2) Two builds of the same host from the same sources
differ in bytes because `libcapstone.c`'s `__FILE__` (an assert string) embeds the path the build resolved
(the worktree symlink or the main checkout), shifting `.rodata` and every offset into it; the same-program
gate pins the sources and tolerates exactly that one string.

**Boot r1b1 (20:19–20:36), group 1: the plain arm, stops 1–6, and the stale handle offered back.**

| arm | image (sha256/16, entry) | pre-registered | read |
|---|---|---|---|
| p1 stop 1 | `fc5e120bc2134bef` @0x10000 | C10000 | **C10000** |
| p2 stop 2 | `af9df7048efbc207` @0x90000 | C20000 | **C20000** |
| p3 stop 3 (the byte survives the destroy) | `9d55128c0eb91875` @0x110000 | C300A0 | **C300A0** |
| p4 stop 4 (the address comes back) | `6f9ffe9eee9924ed` @0x190000 | C40001 | **C40001** |
| p5 stop 5 (old pointer reads the new occupant) | `5b9ae11930ceb166` @0x210000 | C5005B | **C5005B** |
| p6 stop 6 (stale free declined, −5) | `4155e787cf4af493` @0x290000 | C600FB | **C600FB** |
| s9 stop 9, Sublet: the stale subordinate handle offered back to REVOKE (no QEMU pass: the emulator aborts on it, capstone-qemu PR #4) | `05cd076b6392196f` @0x310000 | FAULT | **no return in 300 s; trap log `0x9a` = seen, mcause 26; mepc 0x81B03D2C − DBAS 0x81B00000 = +0x3D2C = `revoke t0` in `sublet_give_to`** |

**Boot r1b2 (20:41–20:53), group 2: the plain stop 7 and the Sublet arm's returning stops.**

| arm | image | pre-registered (emulator) | read |
|---|---|---|---|
| p7 stop 7 (no nested authority on the plain arm) | `c18075469db438e2` @0x10000 | C7FF00 | **C7FF00** |
| s4 stop 4 | `199d3d0e669178c8` @0x90000 | C40001 | **C40001** |
| s6 stop 6 (stale free declined on the protected arm too) | `03e015237c488f50` @0x110000 | C600FB | **C600FB** |
| s7 stop 7 (nested handle's type before / after the ancestor's revoke) | `65ba5490a7271ec3` @0x190000 | C70277 | **C70227** |
| s8 stop 8 (stale handle vs the block's new owner) | `62bb132a448e88b0` @0x210000 | C80071 | **C80021** |
| s1 stop 1 | `83f70385307f1bfe` @0x290000 | C10000 | **C10000** |
| s3 stop 3, Sublet: the object touched after the destroy | `103998ef04d6c342` @0x310000 | FAULT | **returned C30000: the load retired and read 0x00** |

**Readings.** Eleven of fourteen cells read exactly the emulator's mark, and the stale-handle REVOKE
faults on silicon with an identified site and cause (and, since 2026-09-15 00:5x, on the emulator too: run on
the helper lane's merged capstone-qemu tip `acaa44c228` — `c128-qemu-merge` with PR #4, "REVOKE raises 24 and 26
instead of aborting" — built apart from the pinned binary, the same image `05cd076b6392196f` halts with
`cause = 24, pc = 0x101583d2c` = image+0x3D2C, the SAME `revoke t0` the FPGA's mepc names; the emulator's 24
is its NOT_CAP, the handle having been untagged by its `ldc` (Q-11), the RTL's 26 INVALID_CAPABILITY the
tagged handle under a revoked node — enforcement agrees, the cause names differ for the Q-11 reason; the
parent's QEMU pin is unchanged): mcause 26 is produced by two units on this RTL
(the execute path's `INVALID_CAPABILITY`, `capstone_unit.anvilh:304`, and the LSU's own table,
`load_store_unit.sv:995`), and the latched mepc names the `revoke` — the execute path. Three cells
differ from the emulator, and all three are the same two mechanisms (RTL-oracle reading, claim-audited;
the QEMU binary is the 2026-09-11 15:18 build of `c128-qemu-merge` and the one `op_helper.c` commit
after it, `656cc03489`, does not touch `helper_cslcc`):

* **The type read of a revoked handle is 2 on the RTL and 7 on the emulator** (s7's "after" byte, s8's
  "stale handle" byte, and the four `== SUBLET_TYPE_NONE` checks of `ngx_subpool_test.c` phase 13, r1b3).
  Both type queries just read the register's type field (`LCC` selector 1, `capstone_dyn_unit.anvil:
  206-208`; `helper_cslcc` case 1, `op_helper.c:894-896`); the difference is `ldc`: QEMU's writeback
  (`helper_reg_set_cap_compressed`, `op_helper.c:1583-1591`) clears the tag of a loaded capability whose
  node is revoked, so the later type read takes the untagged short-circuit (`:856-858`, 7); the RTL's
  `LDC` (`capstone_dyn_unit.anvil:355-357`) validates only the addressing capability's node and
  `check_load_data` (`capstone_unit.anvilh:583-609`) forwards the loaded value verbatim. Encoding is not
  the cause (both number the post-shift types LIN 0, NONLIN 1, REV 2, UNINIT 3; the RTL's 7 is
  `NOT_CAP − 1` wrapped, the emulator's 7 its untagged fallback). On silicon a revoked-but-present handle
  (2) and an empty slot (7) are therefore distinguishable, which is what `sublet.h:175-176`'s model
  says; the emulator collapses both to 7, and phase 13's four expectations encode that. Enforcement on
  use agrees on both machines (s9). Found on the way: QEMU's `LCC` validity selector (imm 0) is a stub
  returning 1 (`op_helper.c:891-893`); nothing in the ports uses it.
* **A byte load after the pool's destroy retired inside the domain** (s3): the Sublet arm's touch
  returned 0x00 where the plain arm reads its own 0xA0 (p3) and the emulator faults the same image at
  the same `lbu` (image+0x66cc) with cause 24 NOT_CAP — because its `ldc` had untagged the reloaded
  pointer. The 0x00 is the port's own `stc zero` fill (`sublet.h:138-142`, run when the revoke hands
  back UNINIT), not hardware. Whether the reloaded base was still TAGGED on the board — which decides
  between "the LSU's node-validity clause (`load_store_unit.sv`, cause 25) never applies below M-mode,
  where ISSUES R-31's 2026-09-10 block already settled the gate from source" and "an untagged base, the
  2026-08-04 NOT_CAP result re-observed" — is read by two diagnostic stages added tonight (stop 10: the
  reloaded pointer's type before the touch, emulator 7 as predicted; stop 11: the type and the touch in
  one run, faults on the emulator): r1b4 read s10 as **CA0180** and r1b5 read s11 as **CB0100** — the
  base is TAGGED (type 1) on silicon, so the LSU let a load through a capability under a revoked node
  retire, the never-before-measured domain-side reading of the M-mode gate. On the deployed
  configuration a data access through a stale pointer inside a domain is not stopped by a trap (s3
  reads the fill, s5 reads the new occupant); capability-manipulating instructions are (s9). The paper's three `tab:safety` rows that say "stops at access" rest on emulator
  evidence — the lead's framing call, raised in the 2026-09-14 report.

Caveats: N = 1 per cell (repetitions 2 and 3 follow); s9 is single-source board evidence; the boots'
S-07 displacement self-test reported itself unproven (it does not touch the marks); the rev-node head
read VOID on r1b2 (the nginx probes mint tens of nodes each, so exhaustion is not a live concern, but
it is not measured).

**Boot r1b3 (21:00–21:15), group 3.** s2 `d049522634d2b111` @0x10000 → **C20000** (pre-registered
C20000). The 14-phase nginx subpool test `f91bb7a72096e043` @0x90000 → **0D3E04** against the emulator's
0E3E00: 62 checks ran, four failed, the first in phase 13 — the four `sublet_type(...) ==
SUBLET_TYPE_NONE` checks after the two-level withdrawal (the mechanism above); the neighbour's bytes and
the block's extent held. The PostgreSQL subpool test `91d23730153d6a18` @0x110000 (GOOD on QEMU) did
**not enter**: trap log `0x99` = mcause 25, mepc 0x81A00044 − DBAS 0x81A00000 = image+0x44, the
`delin gp` that opens `test` in `my_first_domain/start.S` — cause 25 is "operand is not a capability"
(a wrong capability type would be 27, the S-15 reading), and the source says why: the PostgreSQL
images carry no `.capstone_gp_initdesc` section, so the loader packs a globals offset of 0 and the
monitor carves no gp at all (`sbi_capstone.c:1123-1126`); the nginx images are linked by
`link-gpfree.ld`, which places that descriptor, and enter (ISSUES M-8, mechanism corrected from
source on 2026-09-15 — the first write-up's "gp arrives already non-linear" was an analogy the
cause number refutes). Recorded `unsupported` (implementation-unavailable); the fix is
the port's. The hierarchy test (`7284f2f8db444e87`, the same glue) and s5 were behind it and run in
boots r1b4/r1b5.

**Boot r1b4 (23:34–23:55, after a two-hour wait for the machine memory lock held by another project's
build).** The diagnostic stage s10 (`ccd06246ba515b4a` @0x390000, board-lane addition: the reloaded
stale pointer's type before the touch, packed with the address's low byte) → **CA0180** on silicon
against **CA0780** on the emulator: the pointer reloaded from its stack slot after the pool's destroy is
**tagged, type 1 (non-linear)**, on the RTL, and untagged (7) on the emulator. So stop 3's byte load
went through a tagged capability whose node was revoked, and the LSU let it retire: the node-validity
clause is inapplicable below M-mode, as ISSUES R-31's 2026-09-10 block derived from source — a
domain-side measurement, not the 2026-08-04 untagged-base result re-observed. The hierarchy test
(`7284f2f8db444e87` @0x190000) did not enter: trap log `0x99` (mcause 25), mepc 0x81A00044 = image+0x44,
the same `delin gp` (M-8, confirmed on a second image). The banner count of this run's summary read 0
because the console split "OpenSBI v" across two UART chunks; the kernel banner is there once, and the
control returned after this run's own `load_image`.

**Boot r1b5 (23:58–00:1x).** s11 (`5d3a8cc4cbd4913a` @0x390000, the type read and the touch in ONE run;
faults on the emulator like stop 3) → **CB0100**, exactly as pre-registered: the reloaded pointer is
tagged type 1 and the byte load through it retired with 0x00 — tag present, load retired, no trap, in one
mark. Then s5 (`d453bbc16109b690` @0x310000, the reused-read on the protected arm, FAULT on the
emulator) → **C5005B**: after the same address was handed to a new object, the old pointer read the new
occupant's first byte 0x5B on silicon, exactly what the plain arm reads (p5). This is the sharpest form
of the finding: on the deployed configuration a stale pointer inside a domain reads another object's
data, and nothing traps.

**Bundle.** `experiments/results/S1S2/` in the paper repository's record shape (one record per attempted
cell and repetition; `points.csv` the full 19 × 3 matrix; the cause numbering note in the manifest: the RTL
numbers capability causes 25–31 (`riscv_pkg.sv`), the emulator from the spec base, so causes are compared
by name).

**Bundle.** `experiments/results/S1S2/` in the paper repository's record shape (one record per attempted
cell and repetition; `points.csv` the full 19 × 3 matrix; the cause numbering note in the manifest: the RTL
numbers capability causes 25–31 (`riscv_pkg.sv`), the emulator from the spec base, so causes are compared
by name).

**Bundle.** `experiments/results/S1S2/` in the paper repository's record shape (one record per attempted
cell and repetition; `points.csv` the full planned matrix; a differing mark is `completed` with
`oracle.match=false`, a returning FAULT cell `unsafe-success`, an image that does not enter
`unsupported`; the RTL's cause numbers are named, the emulator's differ). The QEMU pre-run of E2 the
same evening: cell ⑥ (`ceeded2533a74bce`) at `--arena 2097152` counts **692,392,094** (+0.339 % on the
1,419,584 arena's 690,051,663, re-run the same minute and reproduced exactly), `HEAP 1344064` — which is
**the tables region, not a payload heap** (RETRACTED 2026-09-15 01:00, see §7s: the first reading of this
line said "the port carves 64.1 % of the arena for memsys5's heap"; the Sublet-patched `memsys5Init`
says "the pool is the grant, linear, from the level below; the configured heap holds only the tables
beside it: control bytes, links, one capability per atom and the handles of the split blocks", and the
`HEAP` token prints `sublet_tables_len` = 41 bytes per 64-byte atom of the pool, `speedtest1_measure.c:
105-119`) — and the node counts move with the pool size (5,568 splits, 37,966 mrevs). And P1's O2 arms exist on QEMU (the compiler lane's
B6 note: the SQLite domain builds and runs at -O1/-O2, C-17 corrected): cell ⑤ at -O2
`d61c8bf784f2bbd1` 330,723,308; cell ⑥ at -O2 `c506694f9f6f6889` 338,496,909 (the -O0 node counts,
unchanged); native at -O2 `b36eb3814c3cefce` 240,654,449 with 25,122 lookasides — all at the oracle.
Their boots (sw79 the -O0 2 MiB arena, sw80a/b the -O2 arms) follow E1's remaining boots.

### §7s — E2 of the Sublet-paper plan: cell ⑥ at P1's matched backing limit (boot sw79, 2026-09-15 00:08–00:19)

**What ran.** One boot, the E2 shape of the plan: control `k800` → cell ⑥ (the Sublet arm, image
`ceeded2533a74bce`, the same program as sw74/sw75's row) run by the readback host `2c9e82d101b48160`
with `--arena 2097152 --tables 1750285 --testset main --size 1 --verify` → control → the RR probe.
Bitstream `caplifive_r30r31_1bfff7776`, monitor `4274268`, module `d04bd83`, firmware
`152f525ddf43`. P1's rule is one backing limit held constant across arms; cell ⑤ (memsys5 + pool,
sw75) has a 2 MiB static heap, so the Sublet arena is set to 2 MiB — a host argument, no new image.
Pre-registered in the driver header before the boot: retval 4; `112006 38bb59fd`; `HEAP 1344064`;
`sublet: split=5568 mrev=37966 delin=32565` (the QEMU pre-run's counts at this arena, §7r); cycles
2,803,660,693 ± 1 % (the QEMU count 692,392,094 at sw74's CPI); teardown `released pool rc=1`,
RCLM 0 → 1, no RCPR/RCSH/RCRE; probe `ALEN:00200000`, `RR/done`.

| arm | read |
|---|---|
| control before | `RESULT k800 retval=4` |
| cell ⑥ `--arena 2097152`, `--size 1` | `Verification Hash: 112006 38bb59fd`, `HEAP 1344064 DROPPED 0 RC 0`, **`SPEEDTEST1-CYCLES 2812763422`**, `sublet: split=5568 mrev=37966 delin=32565 revoke=37966 init=5401`, `released pool rc=1`, RCLM `00000000` → `00000001` |
| control after | `RESULT k800 retval=4 cycles=4436` |
| probe | `ALEN:00200000`, `RR/share-A-returned`, `RR/share-B-returned`, `RR/done`; `share-trap` 0 |

One boot banner after this run's `load_image`; runner rc = 0 after 661 s; no HARD STOP / ENTRY-STALL.

**Readings.** 2,812,763,422 cycles is inside the band (+0.32 % of the predicted centre), every
pre-registered token matched. CPI against the QEMU count at the same arena: 4.0624. Against the
1,419,584-arena reading (2,794,183,730, sw74) the 2 MiB arena costs +0.66 % on silicon where QEMU's
count moved +0.34 % — the extra 87 splits and 92 mrevs the larger carve produces, plus whatever the
larger heap does to the caches. **P1's `protection_cost` at size 1 on the matched backing limit:
⑥/⑤ = 2,812,763,422 / 2,551,483,818 = 1.1024** (the two-geometry value of the CONFIGURATION block
was 1.0951). Label: bounded-prototype diagnostic, `-O0`, size 1, single run (P1's five runs over three
boots are still owed; the -O0 row is a diagnostic under P1's own rule, and the -O2 arms are on the
board next).

**What "matched" means here, exactly — corrected 2026-09-15 01:00 (RETRACTION of the first
reading of `HEAP`).** The first version of this paragraph, and §7r's E2 pre-run sentence, read
`HEAP 1344064` as "the port carves 64.1 % of the arena for memsys5's heap" and concluded the pair
still compared a 1.34 MiB heap with a 2 MiB one. That is wrong, from the primary source: the
Sublet-patched `memsys5Init` (the cell ⑥ build's `sqlite3-capstone.c`, "The pool is the grant,
linear, from the level below. The configured heap holds only the tables beside it: control bytes,
links, one capability per atom and the handles of the split blocks") serves EVERY payload
allocation from the pool — the whole `--arena` grant — and keeps its tables in the `--tables` region;
the `HEAP` token in the Sublet build prints `sublet_tables_len` (`speedtest1_measure.c:105-119`,
`SPEED_ARENA_LEN`), which is `((atoms+15)&~15) + 8·atoms + 16·atoms + 16·(atoms+32) + 64` for
`atoms = pool/64` — 1,344,064 for a 2 MiB pool (32,768 atoms) and 911,104 for the 1,421,312-byte pool
the #3 module makes of a 1,419,584 request (22,208 atoms; R-33's rounding — exactly 1,419,584 would
give 910,008, the figure in the 2026-09-14 material, so the pre- and post-re-base Sublet arms did not
hold the pool constant to the byte: 1,728 bytes apart). The "64.1 %" is 41 bytes of tables per
64-byte atom, not a carve. So at `--arena 2097152` the Sublet arm's payload
backing is 2 MiB, the same as cell ⑤'s 2 MiB static heap; what still differs is that ⑤'s memsys5
keeps its own control bytes and links INSIDE its 2 MiB (~1.6 % of it: one byte per atom plus the
link array) while ⑥'s 1,344,064 bytes of tables live outside the pool — a memory-ledger entry for
M3, not a payload-capacity difference. The state doc's "configuration vs discipline" caveat closes
to that sentence. Two consequences: the ⑤ᴳ "heap sweep" of 2026-09-14 (a plain arm at static heaps
of 910,008 … 1,572,864, "⑥'s 910,008 cannot be reached by the knob") was chasing the same
misreading — its result stands as a fact about plain memsys5's minimum heap for `main --size 1`
(in (1.25, 1.5] MiB), but it was never the matched pair it was framed as; and a 2026-09-15 00:29
QEMU pilot of ⑤ at a 1,344,064-byte static heap (`ba2778a05ffe1d5c`, made before the misreading was
found) aborts on memsys5 exhaustion (`__CAPSTONE_SPEEDTEST1_ABORTED__`), which narrows that minimum
to (1,344,064, 1,572,864] and is otherwise moot. The QEMU count of the -O2 Sublet image at this arena, for
sw80b's denominator: `c506694f9f6f6889` at `--arena 2097152` → **340,817,186** (+0.69 % on its
1,419,584-arena count 338,496,909), `HEAP 1344064`, `sublet: 5568/37966/32565` — the same counts as
the -O0 image at this arena, the oracle hash, 2026-09-15 00:10.

**Boot sw80a (00:23–00:33): P1's cell ⑤ and native arms at -O2.** Control → cell ⑤ at -O2
(`d61c8bf784f2bbd1`: memsys5 + lookaside `1200,40`, 2 MiB static heap, `SQLITE_OPT_LEVEL=-O2`, read by
the readback host `2af56927aaf907e9`) `--testset main --size 1 --verify` → control → native at -O2
(`b36eb3814c3cefce`, the plain riscv64 build of the same translation unit, `warm --testset main
--size 1 --verify --stats`). Pre-registered: retval 4 twice; cell ⑤ `112006 38bb59fd`, `HEAP 2097152
DROPPED 0`, cycles in 1.06–1.62 G (CPI 3.2–4.9 on the QEMU count 330,723,308); native `112006
38bb59fd`, `Successful lookasides 25122`, warm cycles in 0.72–1.20 G (CPI 3.0–5.0 on 240,654,449).

| arm | read |
|---|---|
| control before | `RESULT k800 retval=4` |
| cell ⑤ at -O2 | `Verification Hash: 112006 38bb59fd`, `HEAP 2097152 DROPPED 0 RC 0`, **`SPEEDTEST1-CYCLES 1167116810`** |
| control after | `RESULT k800 retval=4 cycles=4436` |
| native at -O2 (warm) | `Verification Hash: 112006 38bb59fd`, `Successful lookasides: 25122`, **`BASELINE-WARM CYCLES 856450080 INSTRS 253525314`** |

One boot banner after this run's `load_image`; no HARD STOP / ENTRY-STALL. CPI: cell ⑤ 3.529, native
3.559 (both inside their bands). **cell ⑤-O2 / native-O2 = 1,167,116,810 / 856,450,080 = 1.3627** —
the capability-domain cost of SQLite's own memsys5 + lookaside arm at -O2, against the same pair at
-O0, sw75's cell ⑤ over sw74's lookaside-ON native, 2,551,483,818 / 2,108,202,651 = 1.2103. Both sides
retire far fewer instructions at -O2 (native: 253,525,314 instructions here), and the capability
ABI's share of what remains is larger. Labelled the same way as the -O0 row — bounded-prototype
diagnostic, size 1, single run — and, unlike the -O0 row, at the optimisation level P1 asks for. The
Sublet arm at -O2 (sw80b) follows.

**Boot sw80b (00:36–00:46): cell ⑥ at -O2 — returned at the oracle, and its Sublet counters are NOT the
emulator's.** Control → cell ⑥ at -O2 (`c506694f9f6f6889`, `SQLITE_OPT_LEVEL=-O2 SPEEDTEST1_SUBLET=1`, stack
declaration 385,024) `--arena 2097152 --tables 1750285 --testset main --size 1 --verify` by the readback host
`2c9e82d101b48160` → control → the RR probe. Pre-registered: retval 4 twice; `112006 38bb59fd`; `HEAP
1344064`; `sublet: split=5568 mrev=37966 delin=32565` (the emulator's counts for this image at this arena,
00:10); cycles 1.09–1.67 G (CPI 3.2–4.9 on 340,817,186).

| arm | read |
|---|---|
| control before | `RESULT k800 retval=4 cycles=4540` |
| cell ⑥ at -O2, 2 MiB arena | `Verification Hash: 112006 38bb59fd`, **`SPEEDTEST1-CYCLES 1520327895`**, **`sublet: split=8654 mrev=41293 delin=32638 revoke=41287 init=8649`**, `released pool rc=1`, RCLM +1 (the `HEAP` token straddled two UART chunks in this transcript and is not quoted) |
| control after | `RESULT k800 retval=4 cycles=4526` |
| probe | `ALEN:00200000`, `RR/done` |

One boot banner; runner rc 0; no HARD STOP / ENTRY-STALL. The cycles are inside the band (CPI 4.461 on the
emulator's count) and the verification hash is the oracle's, so the SQL results are identical — but the
allocator's revocation bookkeeping took a different path on silicon than on the emulator for this image:
+3,086 splits, +3,327 mrevs, +73 delins, +3,321 revokes, +3,248 inits, where the -O0 image at the same arena
reproduced the emulator's five counters exactly on the board (sw79) and the -O2 image reproduces them on
the emulator. The pattern (splits ≈ inits, mrevs ≈ revokes, both ≈ +3.1–3.3 k) is ~3,100 extra
split-then-merge cycles of the buddy allocator, i.e. a different free-list history, not a different
workload. **1,520,327,895 is recorded and is NOT citable as ⑥'s -O2 protection cost until this is
understood**; the patched `memsys5Free` branches only on `aCtrl` bytes (plain loads), so Q-11's type-read
divergence is not on the path. Two hypotheses, separated by boot sw81 (the same image, same arena, same
host, launched 00:53): (a) a deterministic RTL-vs-emulator divergence the -O2 code exposes — the counters
repeat sw80b's; (b) a run-to-run hazard of the R-29 class (a plain store to a granule's high word right
before a 128-bit `ldc` returning it stale) that -O2's tighter scheduling triggers — a third set of
counters. The -O1 image follows on the emulator and then the board.

**Boot sw81 (00:52–01:02): the same -O2 image again — the counters REPEAT sw80b's exactly.** Control 4 →
cell ⑥ at -O2 (`c506694f9f6f6889`, the same host, arena, arguments) → control 4 → probe (`RR/done`), one
boot banner: `Verification Hash: 112006 38bb59fd`, **`SPEEDTEST1-CYCLES 1520848707`** (+0.034 % on sw80b's
1,520,327,895), **`sublet: split=8654 mrev=41293 delin=32638`** — the same three counters to the unit.
So the divergence is DETERMINISTIC: the -O2 image runs one allocator history on the RTL and another on
the emulator, every time, and hypothesis (b) (a run-to-run hazard) is out for this image. The
discriminating figure is the smallest one: `delin` moves by +73, and `delin` is incremented only by
`sublet_take`, so the board handed out 73 more objects than the emulator for the same SQL results;
the ~3.1 k extra splits and inits are buddy-allocator churn downstream of that (the deltas are against the
emulator's 2 MiB-arena counts, `5568/37966/32565`; against its default-arena counts, `5481/37874/32565`,
the split and mrev deltas move by 87 and 92 while `delin` stays +73 — the emulator's `delin` is 32,565 at
both arenas, so the workload requests the same objects whatever the pool geometry, and the +73 is a
statement about allocation events, not geometry). Next: sw82 (the -O1
image `902822a8f303dbaa`, emulator 346,344,966 at this arena with the emulator's counters) says
whether it is -O2-specific; then a software bisection (the SQLite build's patch hooks can mark the
memsys5/sublet functions `optnone` inside an otherwise -O2 image: if that image matches the emulator
on the board, the divergence is in the allocator's own codegen) narrows it to an instruction sequence
for the RTL oracle. The emulator's -O0 and -O2 records at this arena both read all five counters
`5568/37966/32565/37966/5401` (`arena-2097152.log`, `c6o2-2mib.log`), and sw79's board line reads the
same five for the -O0 image.

**Boot sw82 (01:04–01:14): the -O1 image diverges IDENTICALLY.** Control 4 → cell ⑥ at -O1
(`902822a8f303dbaa`, `SQLITE_OPT_LEVEL=-O1`, emulator 346,344,966 at this arena with the emulator's
counters) → control 4 → probe, one banner: `Verification Hash: 112006 38bb59fd`, **`SPEEDTEST1-CYCLES
1553040466`** (CPI 4.484, in band), **`sublet: split=8654 mrev=41293 delin=32638`** — the same three
counters as the -O2 image's two boots, to the unit. So the divergence is not -O2-specific and not a
scheduling-distance effect (the -O1 and -O2 images schedule differently and R-29's shape is absent
from both, scanned 01:1x: zero `sd`→`ldc` same-granule pairs within two instructions in either, eleven
low-word pairs in the -O0 image as the scan's positive control): it is a deterministic semantic
difference between the RTL and the emulator in code that -O1 and -O2 emit and -O0 does not, reached
at exactly the same 73 allocation events in both optimised images. The optnone bisection is next: the
memsys5 functions compiled without optimisation inside an otherwise -O2 image, on the emulator for
its counts and then one boot — if the counters return to the emulator's, the site is in the
allocator's own optimised code and the per-function bisection follows; if not, it is in SQLite's
callers (the 73 extra `sublet_take`s would then be 73 extra allocation REQUESTS, and the per-test
localisation through the A1 hook's test boundaries is the tool).

**Bisection arm A (boot sw8x-optA, 01:39–01:49): the allocator set is EXCLUDED.** The -O2 image with
the twenty functions that touch a Sublet primitive or belong to memsys5 (`memsys5CarveUnsafe/
FreeUnsafe/Init/MallocUnsafe/Parent/Unlink/Link/Malloc/Free/Realloc/Size/Roundup/Log/Shutdown`,
`sqlite3DbFreeNN/DbMallocRawNN/DbNNFreeNN`, `sqlite3MallocLinear`, `sqlite3_sublet_grant/pool`) compiled
`optnone, noinline` (`SQLITE_OPTNONE_FUNCS`, verified on all twenty definitions), image
`edac11e54d2b0a09`, its own emulator counts 345,603,485 at the default arena and 347,936,559 at 2 MiB
with the emulator's counters `5568/37966/32565/37966/5401`: on the board, controls 4 and 4, `112006
38bb59fd`, **`SPEEDTEST1-CYCLES 1600582499`** (CPI 4.60 on its own emulator count), `HEAP 1344064`,
**`sublet: split=8654 mrev=41293 delin=32638 revoke=41287 init=8649`** — sw80b's counters to the unit.
So compiling every function that executes a Sublet primitive, and the whole of memsys5, as -O0 code
inside the -O2 image changes nothing: the divergence is not in the allocator's own codegen, the halves
(`9a8d8b93f6869fac`, `cfdde8ab0f470435`, built with the emulator's counters) do not need boots, and the
73 extra `sublet_take`s are 73 extra requests REACHING memsys5 from code outside that set. Two ways
that can happen with identical SQL results, and the next boot separates them: (a) 73 extra
allocation calls from SQLite's callers (a size or count computation in optimised core code that
reads differently on the RTL); (b) the same calls, but 73 more of them MISSING the lookaside and
falling through to memsys5 (the lookaside's free list and slot accounting under the port). The
`Successful lookasides` counter distinguishes them and needs no new image: the -O2 image again with
`--stats` (boot sw8x-o2stats, queued after R1 boot 2; the emulator's count for this image with
`--stats` is being taken first). If the board's lookaside hits are the emulator's minus ~73, (b);
if equal, (a) and the per-function bisection moves to `malloc.c`'s size deciders and the
`StrAccum`/`VdbeMem` growth paths. Each board reading here is against the emulator count of ITS OWN
image hash, never against another build's.

**Boot sw8x-o2stats (02:11–02:21): the -O2 Sublet run on silicon had NO lookaside — pre-registered at
02:15 from the counters alone, read exactly.** The same image `c506694f9f6f6889` (`--stats` is a host
argument; there is no second build) at the 2 MiB arena: control 4, `112006 38bb59fd`, **`SPEEDTEST1-CYCLES
1521280360`** (CPI 4.463 on this image's own `--stats` emulator count 340,864,115), one banner, and the
statistics block beside the emulator's for the same image and arguments:

| `--stats` line | emulator | board |
|---|---|---|
| Successful lookasides | 25,010 | **0** |
| Lookaside size faults | 157 | **0** |
| Lookaside OOM faults | 0 | 0 |
| Lookaside Slots Used | 1 (max 154) | **0 (max 0)** |
| Pager Heap Usage / Largest Pcache Allocation | 315,200 / 4,592 | 315,200 / 4,592 |
| Page cache hits / misses / writes | 31,797 / 0 / 0 | 31,797 / 0 / 0 |
| Schema Heap Usage / Statement Heap Usage | 16,448 / 0 | 16,448 / 0 |
| `sublet:` split / mrev / delin / revoke / init | 5568 / 37970 / 32569 / 37970 / 5401 | **8654 / 41297 / 32642 / 41291 / 8649** |

Every Sublet counter is sw80b's plus the four takes `--stats` itself adds on the emulator (+0/+4/+4/+4/+0
on both machines), and the lookaside lines say what the counters had already said: the database ran
with **zero lookaside slots** (`nSlot = 0`, so no request was ever tested against a slot, and neither
fault counter could move), and the 32,642 takes are every allocation of the run served by memsys5.
This is neither of the two readings pre-registered at 01:49 (73 extra requests; 73 requests missing
the lookaside): it is a fifth branch, found by decomposing the counters before the boot (02:15 note in
the run directory, and the paper lane's sharper form of it):

* From `sublet.h` and the port's call sites, `split` = buddy splits (each preceded by a `sublet_handle`,
  which counts `mrev`) + carve splits (`sublet_carve`, no handle), `mrev` = buddy splits + takes + linear
  takes, `delin` = takes alone (memsys5 mallocs AND lookaside hits both go through `sublet_take`).
  Emulator: 5,400 buddy + 168 carve (the 1200,40 lookaside is rounded by `memsys5Size` to 64 KiB =
  41 big + 127 small slots, and every carve splits) + 1 linear take — exact. Board (8,654 / 41,293 /
  32,638): `carve = linear − 1`, so **at least one linear take happened and the carve count is 0, 1 or
  2 for any plausible number of linear takes, against 168** — the lookaside block was taken and not
  carved. And `delin` 32,638 = 7,555 (the emulator's memsys5 requests; the census read 7,560) +
  25,010 (its lookaside hits) + 73 — the 73 are the requests the lookaside absorbs in place (a
  `sqlite3DbRealloc` of a slot-sized object returns the slot) and memsys5 must serve afresh; the
  +3,254 buddy splits, merges and fills are the buddy allocator absorbing 25,000 small requests.
* One sub-prediction of the 02:15 note MISSED: it expected `Schema Heap Usage` well above 16,448, and
  the board reads exactly 16,448. That was a wrong reading of SQLite's accounting, not of the story:
  `SQLITE_DBSTATUS_SCHEMA_USED` sums `sqlite3DbMallocSize` over the schema's objects whichever
  allocator holds them (a lookaside slot counts at its slot size), so it cannot separate the two.
  Recorded as missed.

**What it means for the numbers already on record.** sw80b/sw81's 1,520 M cycles, sw82's -O1 1,553 M and
arm A's 1,600 M are all **lookaside-off** Sublet runs; ⑤-O2 (sw80a, 1,167 M) is a lookaside-on memsys5
run (its lookaside state on the board is inferred from the stock `setupLookaside` it compiles, not
read — a `--stats` boot of `d61c8bf784f2bbd1` reads it and is queued behind the R1 boots). So the
⑥/⑤ ratio at -O2 is not a Sublet-vs-memsys5 comparison on matched configurations and is NOT cited;
the -O0 pair (sw79, 1.1024) is, its counters being the emulator's.

**Where the divergence now sits.** Not in the allocator (arm A), not in 73 requests from the core: the
port's `setupLookaside` ended with `pStart = 0` on the board after a successful linear take — the
function's -O1+ code takes its "no lookaside" exit on the RTL and its "lookaside" exit on the emulator
from the same instruction stream. Its -O2 body (0x26518, 0x5ac bytes, 371 lines disassembled) inlines
`sqlite3MallocLinear` and `sqlite3MallocSize` and, where the -O0 code moves every pointer through
memory (`stc`, then `ld` of the cursor word), does four things in registers: it tests pointers by the
integer view of a capability register (`addi rd, rs, 0` then `beqz` — the `aSlot == 0` test at
0x26844 is the one that routes to `sqlite3_free(pStart); pStart = 0`), it passes the block's base as
the pointer argument through `movc` of an integer register (`lcc s3, t0, 3` … `movc a0, s3`), it
compares `p` against `aSlot + nBig` by the integer views of two capability registers (`beq a5, a4` at
0x268d0), and it stores integers into pointer fields with `stc` of an integer register and with two
`sd`s. On the emulator the integer view of a capability register is its cursor
(`helper_cslcc`: `val.scalar` aliases `val.cap.bounds.cursor`); what the RTL's integer datapath reads
from a register holding a tagged capability, and whether `split`/`mrev`/`cincoffset`/`movc` leave the
cursor where the emulator does, is the rtl-oracle question in flight.

**Arm C, pre-registered.** Image `2b9e4d0d3ed523c9`: the -O2 build with `SQLITE_OPTNONE_FUNCS=
"setupLookaside"` alone (one definition marked, count-verified); on the emulator at the default arena
338,503,051 cycles, `HEAP 911104`, `sublet: split=5481 mrev=37874 delin=32565` — the -O2 image's own
default-arena counters, so the attribute changes no count there; its 2 MiB pass is being taken and
the boot follows (queued ahead of R1 boots 3–8). Readings: **`5568/37966/32565/37966/5401`** (or this
image's own 2 MiB counters) → the site is inside `setupLookaside`'s optimised code and the 371 lines
are the whole search space; **`8654/41293/32638/41287/8649`** → outside it, and since arm A already
holds the callees at -O0, the next arm is the -O2 code the `aSlot` allocation runs through
(`sqlite3MallocZero`/`sqlite3Malloc`/`mallocWithAlarm`), whose `aSlot == 0` verdict is the exit taken.

### §7u — E5 of the Sublet-paper plan: M3's memory ledger for P1's size-1 arms (desk work from the images, the loader, the RTL and two census runs, 2026-09-15 01:30)

**Inputs.** The allocation census (`SPEEDTEST1_ALLOCSTATS=1`, a diagnostic build that perturbs timing and
is never a cycle row) on the emulator: cell ⑤ (memsys5 + lookaside `1200,40`, 2 MiB static heap; census
image `9de2fc4e0db934ee` at -O1 because the -O0 image has 552 bytes of headroom under the order-10 ceiling
and the census does not fit) reads `CARVED 6235136 PEAK 1349056 ALLOCS 7560`, `Successful lookasides
25010`, `Lookaside Slots Used 1 (max 154)`, `Lookaside size faults 157`, `Largest Pcache Allocation 4592`,
`Pcache Overflow Bytes max 995328`; cell ⑥ (Sublet, census image `649ccb58f7ff8508`, the 1,419,584 pool)
(at -O0, the measure script's default; the two arms' census images therefore differ in optimisation level,
and `ALLOCS` equal at 7,560 says the census does not) reads `CARVED 6177792 PEAK 1291712 ALLOCS 7560` with
the same lookaside and pcache figures. `CARVED` and `PEAK` both differ by exactly 57,344 bytes between the
arms — one constant offset in two independently counted quantities, which is what the removal of memsys5's
in-heap control array from the Sublet pool looks like, and evidence the census counts what it says. `ALLOCS`
(memsys5 allocations after lookaside) is 7,560 on both arms and the emulator's `delin` is 32,565 at both
pool sizes, so the census is pool-independent: the workload asks for the same objects whatever the
geometry. `CARVED` is every byte ever handed out (what a never-coalescing allocator would need);
`PEAK` the largest live payload. Image sizes are `llvm-size` of the measured images; the loader's
`Domain requirement` lines are the P1 boots' own; the node and tag figures are the RTL lane's
(2026-09-14: 16 bytes per node, 65,536 entries, one shadow byte per 16-byte granule, a 2,048-bit
on-chip array); the tables formula is the port's (`speedtest1_measure.c:105-119`).

| category | cell ② native (lookaside ON) | cell ⑤ memsys5 + lookaside, 2 MiB static | cell ⑥ Sublet, 2 MiB pool |
|---|---|---|---|
| payload, occupied (peak live) | ≈ the same workload: 7,560 allocations, `PEAK` ≈ 1.3 MB (native has no census build; its lookaside 25,122 hits) | **1,349,056** (`PEAK`, memsys5 rounding included) | **1,291,712** (`PEAK`) |
| payload, reserved | 2 MiB static heap (in `.bss`: 2,168,144 with the rest) | **2,097,152** static heap (`SPEEDTEST1_HEAP`, in `.bss` → dom_data) | **2,097,152** pool region (`--arena 2097152`; the §4g row used 1,421,312) |
| allocator control inside the payload region | memsys5's own: 1 byte per 64-byte atom + link heads | **≈ 32,768 + links** (`aCtrl` 1 B/atom, `Mem5Link` per free block), inside the 2 MiB | none: the pool is payload only |
| side metadata (tables beside the pool) | — | — | **1,344,064** reserved (`--tables`): `aCtrl` 32,768 + `aLink` 262,144 + `aCap` 524,288 (one capability per atom) + `aPar` 524,800 (the split-block handles) + 64; occupied part not instrumented |
| lookaside pool (inside the payload region) | 48,000 (1,200 × 40) | 48,000 | 48,000 |
| revocation-node metadata (RTL, platform) | — | — | **693,680 occupied** by the run (43,355 nodes × 16 B, never reclaimed) of **1,048,576 reserved** (65,536 × 16 B) |
| tag storage | — | — (no capabilities minted for payload) | **131,072** shadow bytes for the 2 MiB pool + **84,004** for the tables (1 B per 16-B granule, 6.25 %); platform: 60.18 MiB shadow reserved for 1 GiB, 256 B on chip |
| runtime: domain stack, glue, hostcall regions | process stack (Linux) | stack **385,024** (declared), hostcall 2 × 65,536 (readback host) | stack **1,048,576** (the measured image's declaration), hostcall 2 × 65,536 |
| code + rodata / data / bss | 806,036 / 7,952 / 2,168,144 | 1,463,600 / 15,616 / 2,103,072 | 1,468,548 / 15,616 / 6,160 |
| loader `Domain requirement` (dom_data) | — | 2,701,904 (= heap 2,097,152 + stack 385,024 + globals) | 1,268,832 (stack 1,048,576 + globals; the pool and tables are regions) |

**Reading.** Occupied payload is the same to within memsys5's rounding on both capability arms (7,560
allocations, ≈ 1.3 MB peak) and both reserve 2 MiB for it; what Sublet ADDS is entirely outside the
payload region, and it is larger than the payload it protects: **the discipline's metadata for this
run is 2,252,820 bytes against a peak live payload of 1,291,712 — 1.74×** (tables 1,344,064 + the
run's node-table consumption 693,680 + tag shadow 215,076). The three parts, as ratios, since each is
geometry-independent by construction: the tables are **41 bytes per 64-byte atom = 64.09 % of the
backing pool** at any pool size (the design's cost model; the earlier misreading of this figure as "the
heap" is retracted in §7s); the node table is **monotone consumption, not a peak** — 693,680 of
1,048,576 bytes = 66.2 % spent by one size-1 run and never returned, the M1 dependency expressed in
bytes (a second run in the boot exhausts it, R-12), so it is reserved-that-never-returns rather than an
occupancy; the tag shadow is 6.25 % of every byte a capability can address. What Sublet REMOVES is
memsys5's control array from inside the heap (the 57,344-byte offset above). Reserved-vs-occupied is
kept apart in every row; the tables' occupied fraction and the tag array's occupied lines are not
instrumented and are reported as reserved. Cell ⑥'s census at the 2 MiB pool (the same census image through the readback host at
`--arena 2097152`, 01:39) reads `CARVED 6177792 PEAK 1291712 ALLOCS 7560` — identical to the
1,419,584-pool figures to the byte, as `ALLOCS` and `delin` said it would be.

### §7t — E3 of the Sublet-paper plan: R1's release-to-reuse cost on silicon, boot 1 of 8 (2026-09-15 01:20–01:37)

**What ran.** The R1 slots-and-pools harness (`capstone/sublet/r1/r1_slots_pools.c`, image
`6a569a7e5e34178b`, -O1, entry 0x410000) through the readback host `2c9e82d101b48160` on the SQLite
host's `--speedtest1` protocol: one invocation = one fresh domain and one fresh `--arena` grant (4 MiB;
8 MiB for the unrelated-heap series, 2 MiB for object-free) with a 64 KiB `--tables` region above it
(M-9), one repetition of one (arm, series, pattern) per invocation, twelve invocations per boot from a
90-line repetition-major list (5 repetitions × 18), controls first and last. Boot 1 = repetition 1 of the
spatial arm's nine invocations and the Sublet arm's affected-nodes shared/combined and released-bytes
shared. Bitstream `caplifive_r30r31_1bfff7776`, monitor `4274268`, module `d04bd83`. Pre-registered in
the driver header: retval 4 twice; twelve `speedtest1-ran=0x4EB1xxxx`; twelve `released pool rc=1`;
every point `ok=1 bad=0`; `nd = 2n` on shared nodes points; `fb = B` and type 3 (UNINIT) on
shared/combined; the fill ~1.5 cycles/byte (24 per 16-byte `stc`, the July fillcost figure); REVOKE
flat in n if the RTL's revoke is O(1), "a rise with n is the node-linear finding R1 asks about".

Read: controls 4 and 4; twelve ran codes, all distinct for the Sublet arm (the low 16 bits are the
nodes minted); twelve releases `rc=1`; 57 point lines, all `ok=1 bad=0` (the driver's own summary
counted one `bad≠0` because a monitor share marker spliced that line mid-token; the bundle generator
deletes markers before reassembly and reads 57/57 clean, every invocation's line count equal to the
harness's own `lines=` declaration); no HARD STOP, no entry stall, no fault, no `create_region`
failure; the boot banner is chunk-split in the run-scoped transcript (as in sw78 r1b4) and the
fresh-run evidence is the twelve new ran codes after this run's `load_image`. Exclusive cycles
(mcycle in the domain), the Sublet arm, repetition 1:

| series | pattern | n | B | nd | bookkeeping | REVOKE | fill | INIT | reissue | total | fill bytes | type |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nodes | shared | 1 | 256 KiB | 2 | 475 | 76 | 474,921 | 6 | 925 | 476,403 | 262,144 | 3 |
| nodes | shared | 4 | 256 KiB | 8 | 358 | 213 | 474,817 | 5 | 800 | 476,193 | 262,144 | 3 |
| nodes | shared | 16 | 256 KiB | 32 | 996 | 748 | 474,817 | 5 | 814 | 477,380 | 262,144 | 3 |
| nodes | shared | 64 | 256 KiB | 128 | 3,693 | 2,931 | 474,817 | 5 | 901 | 482,347 | 262,144 | 3 |
| nodes | shared | 256 | 256 KiB | 512 | 15,072 | 11,634 | 474,817 | 5 | 797 | 502,325 | 262,144 | 3 |
| nodes | combined | 1 | 256 KiB | 2 | 379 | 55 | 474,921 | 6 | 875 | 476,236 | 262,144 | 3 |
| nodes | combined | 4 | 256 KiB | 8 | 512 | 133 | 474,884 | 6 | 711 | 476,246 | 262,144 | 3 |
| nodes | combined | 16 | 256 KiB | 32 | 1,068 | 636 | 474,817 | 5 | 773 | 477,299 | 262,144 | 3 |
| nodes | combined | 64 | 256 KiB | 128 | 3,723 | 2,348 | 474,817 | 5 | 738 | 481,631 | 262,144 | 3 |
| nodes | combined | 256 | 256 KiB | 512 | 14,964 | 9,005 | 474,817 | 5 | 986 | 499,777 | 262,144 | 3 |
| bytes | shared | 16 | 4 KiB | 32 | 1,274 | 724 | 7,208 | 6 | 993 | 10,205 | 4,096 | 3 |
| bytes | shared | 16 | 16 KiB | 32 | 914 | 818 | 29,377 | 5 | 789 | 31,903 | 16,384 | 3 |
| bytes | shared | 16 | 64 KiB | 32 | 915 | 817 | 118,465 | 5 | 785 | 120,987 | 65,536 | 3 |
| bytes | shared | 16 | 256 KiB | 32 | 915 | 816 | 474,817 | 5 | 785 | 477,338 | 262,144 | 3 |
| bytes | shared | 16 | 1 MiB | 32 | 915 | 822 | 1,900,225 | 5 | 785 | 1,902,752 | 1,048,576 | 3 |

The exclusive brackets sum to the outer bracket with zero residual on every row (the instrument
accounts for the whole measurement, which is what makes the decomposition citable). The spatial arm's
rows (the same fixtures as plain pointers into one alias; bookkeeping and reissue only) read
bookkeeping 301 / 199 / 431 / 1,649 / 7,217 and reissue 348 / 169 / 285 / 462 / 486 cycles for n = 1
… 256 (shared); they are in the bundle.

**Readings (N = 1; repetitions 2–5 follow in boots 2–8).**

* **REVOKE is node-linear on the RTL.** Against the pre-registered "flat if O(1)": 76 → 213 → 748 →
  2,931 → 11,634 cycles for 2 → 512 nodes under the handle, (11,634 − 76) / (512 − 2) = **22.7 cycles per
  revocation node** the withdrawal kills; the emulator counts 2 instructions for it at every n. The
  combined pattern (every even leaf freed and reissued before the withdrawal) reads 9,005 at n = 256 =
  17.6 per minted node, i.e. the walk is over the LIVE descendants (the freed leaves' handle nodes were
  already dead). R1's "report node-linear work if observed": observed.
* **The fill the UNINIT rule forces is byte-linear at 1.76–1.81 cycles/byte** — 28.2 / 28.7 / 28.9 /
  29.0 / 29.0 cycles per 16-byte `stc` at B = 4 KiB … 1 MiB — and does NOT bend with B: the slope is the
  same below and above the 32 KiB data cache — a 2.8 % spread between fully cache-resident and far
  past it, where a cache-latency-bound fill would show far more — so the fill is bound by the store
  path's bandwidth, not by cache latency: capability-grained stores retire through the write-through,
  no-write-allocate cache regardless of residency, and the small rise from 4 KiB is loop overhead
  amortising, not a cache knee (do not read 1.76 as the start of a curve). That also scopes the claim:
  1.81 c/B is a property of THIS memory system's store path and transfers to another implementation
  only as far as its store bandwidth does. Pre-registered 1.5 c/B from the July fillcost rung's 23.6
  cycles per store; the harness's loop is five instructions per store (CPI ≈ 5.8 on the fill).
* **The terms are separable, so the release cost is a sum, not a fit.** REVOKE is flat in B at fixed
  n (724–822 cycles at n = 16 for every B; the nodes-series slope predicts 22.66 × 32 = 725 at the low
  end exactly) and the fill is flat in n at fixed B (474,817 at every n for B = 256 KiB); each term
  depends only on its own variable, which licenses writing, for this RTL and store path,
  **release ≈ 1.81·B + 22.7·nd + 58.9·n + O(1)** (bytes filled, nodes killed, leaves cleared) — a model a
  reader can apply to their own geometry, where a grid fit could not be, and the thing to put where
  the manuscript's constant-time claim stands.
  The fill is 94.5 % (n = 256) to 99.5 % (n = 1) of the release at B = 256 KiB and 99.9 % at 1 MiB; the
  node-linear REVOKE overtakes it only past nd ≈ 20,900 nodes at B = 256 KiB (B-conditional).
* **INIT itself costs 5–6 cycles** — 0.001 % of the release. The expensive thing is the write-through
  the specification's UNINIT rule requires before INIT succeeds, not the instruction: a rule, and so
  addressable by design (an INIT that accepts a region without the capability-grained walk, or a bulk
  clear), which the number argues for — 474,817 cycles per 256 KiB release to establish a property.
* Bookkeeping is ~58 cycles per leaf (a capability-slot clear plus a pointer null; the `stc` itself is
  ~29), reissue (a take, a checked store and load in the returned region) ~800–1,000.
* For the manuscript's two formulations of the release cost (the intro's "time independent of the
  objects inside it", the motivation's O(children)): the first is refuted outright — three terms
  scale, none is flat; the second is consistent with REVOKE and bookkeeping; but the term that
  dominates, the byte-linear fill, is one the paper does not model — the sharper statement is "95–99 %
  of measured release time is a dimension the paper's cost argument does not discuss", not "R3 is
  wrong". The lead's framing call, as with tab:safety.
* The n = 1 fill (474,921) is 104 cycles above the others' 474,817 (0.02 %); noted, to be read against
  repetitions 2–5.

**Harness findings on the way to this boot** (registry M-9, M-10): a pool released while
top-of-stack is popped and the next `create_region` in the boot faults inside the monitor — hence the
`--tables` region above every arena; the region slots are never reclaimed within a boot, ~14
region-bearing invocations before `create_region` fails — hence twelve per boot; the emulator console
truncates a long guest command silently — hence the invocation list in a script.

**Boot 2 of 8 (01:52–02:09): the unrelated-heap, depth and object series under Sublet, and the spatial
arm's second repetition.** Invocation lines 13–24 (Sublet bytes/combined, heap shared + combined, depth
shared + combined, object shared; spatial nodes/bytes/heap shared + combined, repetition 2), the same
image `6a569a7e5e34178b`, controls 4 and 4, 57 point lines and 12 end lines, no refused line, no bad
column, no `create_region` failure; the bundle regenerated over both boots holds 114 records with no
warning.

* **The release cost does not depend on the unrelated heap.** With n = 16, B = 256 KiB and U = 64 KiB,
  256 KiB, 1 MiB, 4 MiB of live, unrelated regions beside the released one: 477,207 / 477,200 /
  477,198 / 477,203 cycles (combined), 477,377 / 477,353 / 477,353 / 477,355 (shared) — flat to
  0.005 % across a 64× range of heap. The walk is over the handle's descendants, not the heap. The
  U = 0 point reads 477,744 / 477,872, ~550 higher, and it is the first point of its invocation: its
  bookkeeping and reissue carry the cold cost (bk 1,242 vs 1,095, re 1,018 vs 731; the fill also +106,
  as boot 1's n = 1 point was) — a warm-up, not a heap effect, and the reason every series here is read
  from its second point on.
* **Depth does not change the cost at a fixed set of affected nodes.** Nesting depth 1 / 2 / 4 / 8 with
  the same n = 16, B = 256 KiB: nd = 47 at every depth, REVOKE 852 / 815 / 812 / 845 (combined) and
  958 / 1,004 / 1,021 / 1,003 (shared), totals 478,340 / 478,011 / 477,986 / 477,989 and 478,658 /
  478,130 / 478,082 / 478,067 — within 0.1 %, the depth-1 point again first in its invocation (and the
  only one whose INIT reads 62 cycles instead of 5: cold). The revocation walk is priced per node it
  kills, not per level it crosses.
* **A single object's release-to-reuse under Sublet is 578–658 cycles at 64 B–4 KiB** (bookkeeping
  166–224, REVOKE 45–62, fill 7–12, INIT 2, reissue 347–375) against the spatial arm's 192–215 — the
  discipline adds ~390–440 cycles per object, almost all of it the reissue's take and checked access,
  and the byte-linear fill is ABSENT: an object goes out as a non-linear alias, so its revoke hands the
  region back initialised (fill bytes 0, the 7–12 cycles are the loop's test), exactly the behaviour
  the SQLite counters show (`init` climbs with merges, not with frees). The 16 B point (1,013 vs 527)
  is first-in-invocation. So the cost the paper's object-free series wants is a few hundred cycles,
  and the fill that dominates the region series does not enter it.
* The Sublet bytes series in the combined pattern: 9,851 / 31,784 / 120,868 / 477,248 / 1,902,631
  cycles for 4 KiB … 1 MiB, its fill 7,221 / 29,402 / 118,490 / 474,827 / 1,900,250 — the shared
  pattern's fill to within 25 cycles at every B, so the combined pattern's freed-and-reissued leaves
  change bookkeeping (1,017–1,202 vs 914–1,274) and nothing else.
* **The spatial arm repeats to the cycle.** Repetition 2's medians equal repetition 1's at every point
  that is not first in its invocation (nodes/shared bookkeeping 303 / 215 / 431 / 1,665 / 7,217, reissue
  347 / 169 / 285 / 460 / 486; the bytes and heap points 704–759 in total), so the board's per-point
  noise is a few cycles and the repetitions the protocol asks for are a check on the RTL's
  determinism, not on a distribution.

Boots 3–8 (repetitions 2–5 of the Sublet series, 3–5 of the spatial) are queued behind one bisection
boot of the E2 -O2 question (§7s); the bundle is regenerated after each.
