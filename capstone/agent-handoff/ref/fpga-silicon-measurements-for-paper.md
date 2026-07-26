# Silicon results for the paper

Consolidated, paper-facing extract of every measurement taken on the CapliFive CVA6
FPGA. The dated notes in `history/` are the investigation trail; **this file is what
a paper author lifts from**. Each entry states the number, the exact conditions, what
it supersedes in the current draft, and what it does **not** establish.

Vehicle throughout: Genesys 2 CVA6, bitstream `working-caplifive-captype-fixed.bit`,
`mcycle`/`minstret` read in-domain (the unprivileged counters are gated for the
domain). Single core.

Last updated 2026-07-26.

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

## 1. Primitive costs — cycle-accurate (already in the draft)

Feeds `tab:primcost-rtl`. Method: each operation is an `N`-iteration inner loop
bracketed by two `mcycle` reads with an empty calibration loop subtracted.

| primitive | cyc/op |
|---|---:|
| `load` | 2 |
| `shrink` | 1 |
| `mrev` (mint revocation node) | 50 |
| `delin` + `revoke` | 121 |
| **`mrev`+`delin`+`revoke`** (reclaim per lend) | **171** |
| **borrow** = reclaim + load | **~173** |

Super-operation view:

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

## 2. NEW — Pervasive spatial safety: 3.2% scalar, 5.0% array, 80% recursive

The draft claims spatial safety is pervasive ("every pointer is a bounded
capability, always on"), demonstrates it is **correct**, and never prices it.
This is that price, measured.

Method: each kernel compiled **twice from the identical source header** by the
**same clang at the same `-O`** — once for `-target capstone64` as a pure-capability
domain, once for `-target riscv64` with no capability flags — and run on the same
board. A static gate fails the build if a capability instruction reaches the
baseline. Baseline is the **warm** pass (the capability domain has no paging). Both
halves bracket the compute only, so domain entry/exit is excluded from both.

| benchmark | opt | capability | baseline | **cycles** | **instructions** |
|---|---|---:|---:|---:|---:|
| `beebs_prime` (pure scalar) | −O0 | 47,780 | 46,306 | **1.032×** | — |
| `rv8_primes` (sieve, 16.5M cyc) | −O0 | 17,283,292 | 16,459,057 | **1.050×** | **1.102×** |
| `beebs_recursion` (deep + mutual recursion) | −O1 | 18,957 | 10,523 | **1.801×** | **1.458×** |

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

Measured on `rv8_primes`, both counters, same session:

| | cycles | instret | CPI |
|---|---:|---:|---:|
| capability domain | 17,375,220 | 8,773,753 | **1.98** |
| baseline (warm) | 16,459,057 | 7,960,829 | **2.07** |
| ratio | **1.056** | **1.102** | |

The capability build retires **10.2% more instructions** but costs only **5.6% more
cycles**, and its **CPI is lower**. The 812,924 extra instructions cost 1.13 cycles
each against a program average of 2.07.

**Claim this supports:** capability enforcement is essentially free per instruction
on this CVA6. The overhead is the **ABI** — the gp cap-table routes every global
through `ldc rd, i*16(gp)` — and those extra instructions are simple loads that
pipeline better than average. A tuned ABI with fewer cap-table indirections would
reduce it.

**One benchmark.** This is the caveat that matters most.

Source: `history/26-07-2026_15-58-45_overhead-decomposed-and-fault1-reproduces-in-perf-rungs.md`.

---

## 4. NEW — Measured CPI replaces an assumption in `tab:appoverhead`

The draft computes SQLite boundary overhead as
`borrows × 171 cyc / (instr × CPI)` and assumes **CPI = 1**, calling it "the
conservative upper bound", with a 1.5 sensitivity case. Measured on this silicon:

| benchmark | warm cycles | warm instret | **CPI** |
|---|---:|---:|---:|
| `coremark_matrix` | 55,975 | 27,788 | 2.01 |
| `rv8_primes` | 16,459,057 | 7,960,829 | 2.07 |
| `matmult_int` | 71,860 | 29,661 | 2.42 |
| `beebs_insertsort` | 8,398 | 3,410 | 2.46 |
| `beebs_prime` | 46,306 | 14,680 | 3.15 |

**Measured CPI is 2.0–3.2, not 1.** A larger CPI enlarges the baseline and shrinks
the overhead, so the two rows of `tab:appoverhead` should be roughly **halved**:

| workload | draft (CPI=1) | **at measured CPI** |
|---|---:|---:|
| `speedtest1` (whole benchmark) | ~1% | **≈0.5%** |
| in-domain result scan (worst case) | ≤6% | **≈3%** |

An assumption in the paper becomes a measurement, and the claim gets stronger.

---

## 5. What is NOT established — read before citing anything above

- **Three benchmarks, not seven.** `matmult_int` and `coremark_matrix` produce no
  result at ANY reachable configuration: they transfer cleanly, then the `cscall`
  hangs (matmult at −O1 and −O2; coremark at −Os and at −O0 with a 32 KiB window).
  This is a domain-entry fault, not delivery and not the compiler — it is not `-Os`
  codegen (coremark hangs at −O0), not code size (coremark hangs at 1,988 B of text,
  smaller than rungs that pass), and no instruction is present in every hanging build
  and absent from every passing one. `beebs_crc32` cannot build at −O1+ and
  `beebs_insertsort` crashes clang at −O1.
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
