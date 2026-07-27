# Silicon results for the paper

Consolidated, paper-facing extract of every measurement taken on the CapliFive CVA6
FPGA. The dated notes in `history/` are the investigation trail; **this file is what
a paper author lifts from**. Each entry states the number, the exact conditions, what
it supersedes in the current draft, and what it does **not** establish.

Vehicle throughout: Genesys 2 CVA6, bitstream `working-caplifive-captype-fixed.bit`,
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
and at 32 KiB. Not yet silicon-validated. Lifting it is what would make full CoreMark
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
| **CPI** | **C**ycles **P**er **I**nstruction = `cycles ÷ instret`. How expensive the average instruction was. CPI 1 = one instruction finishes per tick (a perfect pipeline). CPI 2 = each instruction costs two ticks on average — stalls, cache misses, multi-cycle ops. **This CVA6 measures 2.0–3.2, never near 1.** |

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

> ## ⚠ READ FIRST (2026-07-28): the cycle ratios below are UNDER-STATED
>
> A controlled probe (`ctrsanity`, issue **I-2**) shows the **Linux baseline runs identical
> work ~1.21x slower than a domain** — 6.00 vs 7.25 cycles for the same 5-instruction
> register-only loop, verified instruction-for-instruction in the disassembly. Cause: timer
> interrupts inside the measurement bracket (the baseline's excess scales 3.92x for 4x the
> work, at 14 cycles per excess instruction). It inflates the **denominator**, so every
> affected row **understates capability overhead** — the error runs in our favour.
>
> **Do not publish a cycle ratio without per-row triage, and do not apply a blanket 1.214x
> correction either.** Short kernels may be interrupt-free: `beebs_bs` (1,912 cyc baseline)
> and `beebs_recursion` both retired byte-identical instret across passes and are probably
> clean. But byte-identical instret is **necessary, not sufficient** — `beebs_cnt` passed that
> test and still shows an impossible 0.684x.
>
> **The instruction ratios are far less affected (0.982 vs 0.824) and the paper's central
> claim — the cost is the ABI, not the enforcement — is an instruction-count argument.
> Lead with instructions.**

## 2. NEW — Pervasive spatial safety: 3.2% scalar, 5.0% array, 80% recursive

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
> **When new rungs land, edit `tab:spatialcost` — it is built to take more rows.**

Method: each kernel compiled **twice from the identical source header** by the
**same clang at the same `-O`** — once for `-target capstone64` as a pure-capability
domain, once for `-target riscv64` with no capability flags — and run on the same
board. A static gate fails the build if a capability instruction reaches the
baseline. Baseline is the **warm** pass (the capability domain has no paging). Both
halves bracket the compute only, so domain entry/exit is excluded from both.

The `capability` and `baseline` columns are **cycles**. The two bold columns are the
overhead ratios (capability ÷ baseline) for cycles and for instructions respectively.

> ### 2026-07-28: THIS TABLE IS WITHDRAWN except for `beebs_bs`
> Measuring each baseline 16× and keeping the least-disturbed pass changed it materially.
> **`beebs_prime` 1.032× is wrong** — its baseline was carrying ~1,900 interrupt-handler
> instructions; min-of-16 gives 29,775 cyc (−33 %) and the ratio becomes **≥1.605×**, still
> uncertified at 5/15 ties. The **"3.2 % scalar / 5.0 % array / 80 % recursive" headline is
> withdrawn** — its cheapest, most quotable component was the most contaminated.
> Trail: `history/28-07-2026_01-30-00_RESULTS-min-of-16-fixes-short-rungs-and-beebs_prime-is-not-1.032x.md`.

| benchmark | opt | capability (cyc) | baseline (cyc) | **cycles** | **instructions** | status |
|---|---|---:|---:|---:|---:|---|
| `beebs_bs` (binary search, read-only table) | −O1 | 2,258 | **1,772** | **1.274×** | **1.058×** | ✅ **CLEAN — 15/15 passes tied at min instret, 45-cyc spread. The only defensible row.** |
| `beebs_prime` (pure scalar) | −O0 | 47,780 | 29,775 | ≥1.605× | — [†] | ⚠ was 1.032×; 5/15 ties, true value likely higher |
| `beebs_cnt` (matrix seed + sum) | −O1 | 128,178 | 110,013 | 1.165× | 1.277× | ⚠ was an impossible 0.773×; 1/15 ties |
| `rv8_primes` (sieve, 16.5M cyc) | −O0 | 17,283,292 | 16,389,191 | 1.055× | 1.103× | ❌ **uncorrected** — too long for a clean pass; carries the full ~1.21× penalty |
| `beebs_recursion` (deep + mutual recursion) | −O1 | 18,957 | 10,523 | 1.801× | 1.458× | ⚠ not re-measured with min-of-16 |

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

## 3. NEW — That overhead is ABI, not hardware

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

- **This is still an estimate, not a measurement of SQLite.** SQLite has not run on the
  board. Both rows depend on modelled borrow counts, which is the larger uncertainty —
  bigger than the CPI input this section fixes.
- **CPI is workload-dependent**, 2.01–3.15 even across five small kernels. Using ≈2 is
  the conservative choice *within* the measured range; quoting 3.15 would halve the
  overhead again but is not defensible.
- **None of the five kernels chases pointers**, and SQLite does heavily. Pointer-chasing
  code typically has *higher* CPI (cache misses), which would push the estimate lower
  still — so ≈2 remains conservative for SQLite specifically.
- `cycles = instructions × CPI` is a definition, not a model; it is exact for whatever
  CPI the program actually exhibits. All the uncertainty is in *which* CPI to use.

---

## 5. What is NOT established — read before citing anything above

- **THE MECHANISM IS NOW KNOWN (2026-07-27).** The four non-measured rungs fail because of a
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
- **Three benchmarks measured, not seven — and it will stay three.** `beebs_crc32` and
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
