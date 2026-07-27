# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-07-27.

---

## RTL / FPGA

### R-1 — A load through one capability register misses a store through another `CHARACTERISED`
**The blocker for several of the 13 benchmark rungs.** An intervening store through one capability register
causes a later load through a *different* capability register to miss an earlier store to its own
address — though the addresses are distinct and both capabilities are in-bounds derivations of the
same object. Not loop-specific. QEMU executes every probe correctly.

- **Repro:** `/tmp/capstone/capstone-lsu-hazard-repro.tar.gz`; sources
  `tests/runtime-qemu/silicon-ladder/rawhazard{_kernel.h,5,6,7}_fpga_app.c`
- **Evidence:** `history/27-07-2026_17-05-00_RESULTS-culprit-found-register-indexed-load-misses-pending-stores.md`
- **Mitigations tried (7, all failed):** fence before load, fence after every store, register
  hoisting, other store register-indexed, 64 B cache-line separation, constant-offset pointer
  walk, both accesses via pointers. **No general software workaround** — a dynamic array index
  cannot have a compile-time-constant base.
- **Impact:** `matmult_int`, `coremark_matrix`, `beebs_crc32`, `beebs_insertsort` unmeasurable.
- **Confidence it is hardware:** high, not certain. Residual doubt is whether our non-standard
  gp-captable ABI provokes it. **Open question for the board owner.**
- **Predictive record — see the SCORED entry below for the final tally (2 hits, 3 misses,
  1 partial). R-1 is NOT a complete account of the board's behaviour, but its own scope is
  confirmed.** Rungs were written specifically to test its predictions:
  - `beebs_bs` — **predicted PASS, PASSED** (887447230 = oracle, 2264 cyc). This is the
    load-bearing confirmation: `bs_data[mid]` is a genuine register-indexed load through a
    derived capability — the exact addressing form in every failing rung — and it is correct
    because nothing is ever *stored* to the table. **The intervening store is a necessary
    ingredient**, not incidental.
  - **SCORED 2026-07-27 (board): 2 hits, 3 misses, 1 partial — and the same-object clause is
    CONFIRMED.** `beebs_cnt` passes on silicon (oracle exactly), and it is the sharpest
    cross-object control available: its seeding loop keeps stores outstanding to `Array` and
    to `Seed` through two capability registers naming two *different* globals. R-1 predicted
    PASS and it passed. **The "same object" clause in this entry and in the repro README is
    therefore tested, not merely inferred, and needs no correction.**
    `beebs_bs` passed again (2,258 cyc, reproducing 2,264 from the prior session).
    `beebs_fac` and `beebs_duff` HANG; `beebs_fibcall` miscomputes while retiring ~94 % of the
    baseline's instructions (166,539 vs 177,855) — a third signature, distinct from both the
    hangs and from "the compute never ran". R-1 speaks to memory-shape failures and does not
    explain hangs, consistent with the standing ≥2-independent-faults position.
    > **⚠ A mid-run report that "R-1's same-object clause is REFUTED" was WRONG and is
    > withdrawn.** It came from a sweep accidentally run at −O0 (see I-1); at the intended
    > −O1 the cross-object control passes. Nothing in the repro package needs changing.
  - **Four predictions registered 2026-07-27 before the board ran.** Built, QEMU-green through
    the identical controller, oracles fixed, `-O1` to match `beebs_bs`. Written down *before*
    the board speaks so they are tests and not stories:

    | rung | predicted | what it discriminates |
    |---|---|---|
    | `beebs_fibcall` | PASS | no arrays at all — a failure would mean R-1 is not the whole story |
    | `beebs_fac` | PASS | same, plus a 2nd recursion point for the 1.801× headline |
    | `beebs_cnt` | PASS | **the same-object clause.** Stores to `Array` and to `Seed` are outstanding through two capability registers naming two *different* globals |
    | `beebs_duff` | PASS | **pointer-walk into two distinct objects** — the form that failed in rawhazard7 *within one object* |

    `cnt` and `duff` are the load-bearing pair. Every failing rung to date is same-object
    (`C[i*N+j] += …` reads and writes ONE array through two derived registers); no
    cross-object control has ever been run. If they pass, R-1 stays narrow and most of a
    benchmark suite remains measurable. **If either fails, R-1 is wider than written — any two
    derived capability registers — and this file plus the repro README must be corrected
    before the package goes to the board owner.**
  - `beebs_janne` — **predicted PASS, HANGS** (see R-6). Now bisected: the failing loop nest
    contains **no memory operations at all**, so R-1 cannot explain it and the two must not be
    conflated. R-1's scope is unchanged by it; its completeness as an explanation of the whole
    board's behaviour is not.

### R-2 — `delin` in domain code wedges the board `WORKED AROUND`
A `delin` executed in domain code on a capability loaded from the gp cap-table wedges the board
(power-cycle to recover). Proven against a size-matched `addi x0,x0,0` control at the same address,
so it is the instruction and not code layout.

- **Repro:** `/tmp/capstone/capstone-delin-repro.tar.gz` (superseded — now a secondary item in the
  R-1 package); probe knob `LADDER_CM_WITH_DELIN`
- **Evidence:** `history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`
- **Workaround:** the `delin` was ours and unnecessary — removed from the default build, which
  also returns `coremark_matrix` to being a faithful copy of upstream.
- **Probably our bug**, not the platform's: the glue already delins every cap-table entry before
  storing it, and our QEMU was patched to tolerate the redundant case *"rather than faulting"*.
  Only the failure *mode* (full wedge vs catchable trap) is worth the board owner's attention.

### R-6 — `beebs_janne` hangs although R-1 predicts it should pass `OPEN`
BEEBS `janne_complex`: nested data-dependent loops whose conditions are computed **entirely from
locals**, with one `.bss` counter (`jc_iters++`) touched through a single capability register.
R-1 requires a load through one capability register with an intervening store through *another*;
that never occurs here, so R-1 predicts PASS. **The board hangs it.**

- **Repro:** `tests/runtime-qemu/silicon-ladder/beebs_janne_{kernel.h,fpga_app.c,host.c}`,
  `-O1`, oracle 484656629, QEMU-correct through the identical controller.
- **BISECTED 2026-07-27 (`janne_diag`), and the result does NOT fit R-1.** Safety bounds turned
  the hang into a returned diagnostic:

  | slot | board | correct |
  |---|---|---|
  | outer trips | **200** (its safety bound) | 9 |
  | inner trips | **500** (its safety bound) | 12 |
  | final `a` | **2** | 31 |
  | final `b` | **-339** | 27 |
  | `jc_iters` | 700 (= 200+500, self-consistent) | 21 |

  Neither loop terminates, and `a` is frozen at 2 — after 200 outer iterations of `a = a + 2` it
  should be ≥ 400. The board state is internally consistent (`a`=2 and `b`=−339 keep both
  conditions true forever), so the loops behaved *exactly* as if `a` stopped accumulating.

  **The damning part: the loop nest is pure register arithmetic.** Verified in both the emitted
  assembly and the shipped `.dom` — `a`=`a3`, `b`=`a2`, the counter accumulates in `a6`, and
  `jd_iters` is stored **once after** the loops. There are **no memory operations inside the
  nest**. R-1 is a memory hazard and therefore cannot explain this.

- **Status: mechanism UNKNOWN. Do not fold this into R-1.** Candidate explanations, none tested:
  a control-flow/branch-resolution issue on this RTL (the nest is unusually branch-dense); an
  interrupt landing inside the measured bracket (the measurements doc notes ~16k cycles when one
  does; this rung ran 11,167); or the emitted code differing from what actually executes.
- **Next probe RUN (`regloop_diag`, 2026-07-27) — and it PASSES, which deepens the puzzle.**
  A staircase of register-pure loops, no memory in any body:

  | probe | board | correct |
  |---|---|---|
  | simple counted loop, 100 trips | 100 | 100 ✅ |
  | nested 10x10 | 100 | 100 ✅ |
  | data-dependent branch in body | 100 | 100 ✅ |
  | multiply in body | 100 | 100 ✅ |
  | **janne's EXACT nest, bounded** | **21** | 21 ✅ |

  So pure-register looping is fine, and **janne's algorithm itself runs correctly on this board**.

- **The open puzzle: two register-pure implementations of the same nest, one fails, one works.**
  Verified by counting memory ops in the loop *region* (not the whole function): `janne_diag`'s
  nest has **0**, and it fails; `regloop_diag`'s dbg4 nest also has 0, and it passes. The
  differences are incidental — three counters vs two, bounds 200/500 vs 400, and dbg4 executes
  after four other loops. Nothing algorithmic.
- **Most likely reading: this is the known code-layout / perturbation sensitivity**, the same
  phenomenon as the 2026-07-26 controlled A/B where **four added instructions flipped a passing
  rung from correct to wrong**. That makes R-6 a *symptom class* rather than a distinct fault, and
  means **a passing rung is not stable ground** — already the standing caveat in
  `ref/fpga-silicon-measurements-for-paper.md` §5.
- **Do not merge R-6 into R-1** (R-1 is a memory hazard; these nests touch no memory), and do not
  claim it is understood. The honest status is: janne's algorithm works, one particular build of
  it does not, and the discriminator is not algorithmic.

### R-3 — Second domain at the same entry VA hangs within one boot `OPEN`
A domain reused at entry VA `0x10000` within a single boot silently hangs its `cscall` — missing
icache invalidate on the domain switch. Forces **one full power-cycle + firmware reload per rung**
(~2.5 min), which is the dominant board-time cost.

- **Evidence:** `ref/HOW-TO-LAUNCH-ON-FPGA.md`; fix sketch `plans/curried-crunching-gizmo.md`
- **Note:** the domain-boundary `fence.i` was long suspected to be the fix for R-1 as well; board
  test #63 disproved that. It remains the right fix for **this** issue only.

### R-4 — A shared-region word is silently corrupted `OPEN`
`rv8_primes` returned the *correct* result while a word of its shared region held a stray DRAM
address. Passing rungs were only ever clean where someone looked.
- **Evidence:** `ref/fpga-silicon-measurements-for-paper.md` §5

### R-5 — Illegal/meaningless capability ops wedge rather than trap `OPEN`
M-mode appears to spin (`capstone_error` = `while(1)`); only a power-cycle recovers. Seen for
`C_GEN_CAP` (QEMU-only op), for the R-2 `delin`, and for an `scc`-derived load.
- **Evidence:** `history/22-07-2026_18-05-00_gp-free-silicon-smoke-*.md`

---

## Infrastructure / procedure

### I-1 — A sweep silently rebuilds at −O0 and discards your pre-built set `FIXED`
`run_ladder_perf_fpga.py` **rebuilds every artifact by default** (the 25-07 anti-stale fix),
shelling out to `build-ladder-fpga.sh` with the inherited environment. Setting `LADDER_OPT`
on a *pre-build* and omitting it from the *sweep* means the runner rebuilds everything at its
`-O0` default and measures that — against baselines specified at another level.

- **Cost when it fired (2026-07-27):** five rungs reported as silicon failures, including one
  that had passed before; a false conclusion that **R-1's same-object clause was refuted**,
  which would have gone to the board owner as a correction to the bug report; and a nearly
  published §5 claim that *an ordinary rebuild flips a passing rung*. All three withdrawn.
- **Caught only by the in-sweep control.** `beebs_bs` was included purely as a stability
  check; its failure is what made the sweep suspect instead of informative.
- **Rules:**
  1. Set `LADDER_OPT` on the **runner** invocation, not just the pre-build.
  2. Keep a **known-good rung in every sweep**. It is the only thing distinguishing
     informative failures from a misconfigured harness.
  3. `LADDER_REBUILD=0` is **required** to run a specific pre-built binary — pointing
     `LADDER_FPGA_DIR` at it does not stop the rebuild from overwriting it.
  4. Compare the static shape (`.text` size, `ldc gp[i]` count) against the known-good build
     before believing a flipped result.
- **Static signature of the mistake** (`beebs_bs`): −O0 → 2,100 B text, 4 `ldc gp[i]`, 2
  cap-table globals, FAILS; −O1 → 1,408 B, 2, 1 global, PASSES. The function-local
  `static const int probes[18]` becomes a delivered cap-table global at −O0 — the C-4
  boundary moving under an optimization flag.
- **Evidence:** `history/27-07-2026_22-40-00_RESULTS-two-new-silicon-rungs-and-an-O-level-procedure-bug.md`

### I-2 — The Linux baseline is ~1.21x slow on identical work `CONFIRMED`
**Every published cycle ratio understates capability overhead.** Proven 2026-07-28 with a probe
whose measured region is a 5-instruction register-only loop -- no loads, stores, pointers or
globals -- verified in the disassembly to emit the identical `srai/xor/addi/add/bne` on both
targets.

| probe | cap cyc | base cyc | **cyc ratio** | ins ratio | cap CPI | base CPI |
|---|---:|---:|---:|---:|---:|---:|
| `ctrsanity` (100k) | 600,309 | 728,727 | **0.824** | 0.982 | 1.201 | 1.431 |
| `ctrsanity4` (400k) | 2,400,310 | 2,884,826 | **0.832** | 0.982 | 1.200 | 1.417 |

Cycles per iteration: domain **6.003 / 6.001** (metronomic), baseline 7.287 / 7.212.

- **Cause: timer interrupts inside the bracket.** The baseline's excess scales **3.92x** in
  instructions and **3.77x** in cycles for **4x** the work -- proportional to elapsed time, which
  is what a periodic interrupt looks like and a fixed entry cost does not -- at **14 cycles per
  excess instruction**, far above this core's 1.2-1.4 CPI, as expected for interrupt entry/exit
  plus cache disruption.
- **Direction matters: it inflates the DENOMINATOR, so it flatters us.** Our overheads are too
  low, not too high.
- **Do NOT apply a blanket 1.214x correction.** Short kernels may take no interrupts at all --
  `beebs_bs`'s baseline is 1,912 cyc (~40 us) with byte-identical instret across passes
  (827/827), as has `beebs_recursion` (2,019/2,019); those rows are probably clean.
- **The "certified clean" test is necessary but NOT sufficient.** `beebs_cnt`'s passes were
  byte-identical (67,140/67,140) and its ratio is still an impossible **0.684x** -- two passes can
  take the *same* number of interrupts and both be contaminated. Identical instret proves
  reproducibility, not absence. `cnt`'s implied factor (1.46x) exceeds this probe's 1.214x, so
  **`cnt` is still not fully explained**.
- **Repro:** `tests/runtime-qemu/silicon-ladder/ctrsanity{,4}_kernel.h` -- both halves registered;
  `-O1`.
- **Evidence:** `history/28-07-2026_00-10-00_RESULTS-I-2-confirmed-the-linux-baseline-is-1.21x-slow-so-our-overheads-are-understated.md`
- **Fix, ranked:** (1) run the baseline **bare-metal**, removing the confound instead of
  modelling it -- real work, the baseline kernels currently link into a Linux userspace binary;
  (2) **lead with instruction ratios**, far less contaminated (0.982 vs 0.824) and the paper's
  central *ABI-not-enforcement* claim is instruction-based anyway; (3) per-row triage before
  publishing any cycle ratio.
- **Impact:** blocks the 4-row cycle table as it stands. Instruction ratios unaffected.

---

## Compiler / toolchain (ours)

### C-2 — `Cannot select: i128 = or` / `= xor` `OPEN`
`lowerScalarI128Logical` bails when the two operands are not *matching* extends (both sign or both
unsigned). Blocks `rv8_qsort` and `rv8_miniz` at −O1/−O2.
- **Why unfixed:** closing it requires deciding what the high 64 bits mean in the mixed case —
  capability metadata vs a genuine 128-bit integer. A semantics call with miscompile risk, left
  alone deliberately under deadline.

### C-3 — RV8 fails at runtime at −O1/−O2 `OPEN`
Five RV8 benchmarks now *build* at −O1/−O2 but fail 10/10 at runtime: `primes`/`aes`/`dhrystone`
hang silently; `sha512`/`norx` take deterministic capability faults (cause 5 OOB / cause 24, same
PC at both levels). −O0 controls all pass. **Not regressions** — code that never compiled cannot
regress.
- **Evidence:** `history/27-07-2026_12-59-35_three-codegen-fixes-*.md`
- **Leads:** `sha512` faults with bounds visibly too small; `norx` with an untagged capability
  reaching a load. Both smell like a bounds/provenance codegen bug at −O1+.

### C-4 — Large read-only data cannot be delivered into a domain `OPEN`
The cap-table glue cannot deliver a global that is both large and a **private** (`.L`) constant:
too big for the unrolled 12-bit store path, and the large-RO copy path needs a *linkable* symbol
to `lla` from the glue's separate TU.
- **Hit by:** `beebs_crc32` (the optimizer constant-folded its runtime-generated table into a
  2048 B private constant at −O1+). **SQLite's const tables will hit the same thing.**
- **Workaround:** make the source opaque to the optimizer so the table stays runtime-generated.
  Per-benchmark, not general.

### C-9 — Redundant `mv rd, rd` around inline-asm register constraints `OPEN`
The Capstone backend emits **no-op self-moves** around an `asm volatile("" : "+r"(x))`
tie. A 5-instruction loop body became 7 — `srai / xor / add / **mv a4,a4** / addi /
**mv a4,a4** / bne` — where plain riscv64 emits 5 for the same source.

- **Found:** 2026-07-27, while building the I-2 counter-sanity probe. It is logged because
  it **silently defeated that probe**: the measurement depends on both targets retiring the
  same instruction count, and the compiler manufactured a 1.4× difference out of nothing.
- **Repro:** `tests/runtime-qemu/silicon-ladder/ctrsanity_kernel.h` with the inner
  `__asm__ volatile("" : "+r"(acc))` restored; disassemble
  `--triple=riscv64 --mattr=+m` and compare against `ladder-base/obj/base_ctrsanity.o`.
- **Impact:** small in isolation (two wasted instructions per tie), but the register-pinning
  idiom is used throughout the ladder kernels to defeat constant folding, so it inflates
  the capability instruction count of **any** rung that uses it — i.e. it can bias an
  overhead ratio upward. Worth a look before the next measurement round.
- **Workaround:** keep inline-asm ties out of measured loops; use an opaque trip count and
  a consumed result instead.

### C-5 — 4 KiB code window `OPEN`
`link-gpfree.ld` forces globals to image offset `0x1000`, capping `.text` at 4096 B. One
hardcoded number, QEMU-validated at 16 KiB and 32 KiB and silicon-validated at 32 KiB. Lifting it
is what full CoreMark and Dhrystone need. Task #62.

---

## Archive — fixed, kept for provenance

**Move an entry here as soon as it is fixed**, with the fix and how it was validated.
Keep the id so older notes that cite it still resolve.

### Fixed 2026-07-27 (evening)

| id | issue | fix | validated by |
|---|---|---|---|
| **C-1** | `Cannot select: i128 = sign_extend_inreg` — an `int` index feeding capability address arithmetic crashed the backend at −O1+. The `Custom` action only runs during Legalize, and `performSIGN_EXTEND_INREGCombine` deliberately handles **only** the `any_extend(i64)` shape because expanding the general case in a combine ping-pongs against `visitSIGN_EXTEND` forever. Every other shape reached ISel unselectable. | Selected directly in `CapstoneDAGToDAGISel::Select` (`CapstoneISelDAGToDAG.cpp`), where there is no combiner to fight: `PseudoTRUNC_CAP` to XLen → `SLLI`/`SRAI` pair to sign-extend the source field → `PseudoSCALAR_COPY_I128` to widen. | repro clean at −O0/−O1/−O2/−O3; new lit `i128-sext-inreg-int-index.ll`; **Capstone lit 42/42** |
| **I-1** | A sweep silently rebuilt at −O0 and discarded the pre-built set, running capability halves at a different −O than their baselines. Cost five bogus "silicon failures", a false refutation of R-1, and a nearly published claim that a plain rebuild flips a passing rung. | Both build scripts now record the per-rung level to `<OUT_DIR>/optlevels.txt`; `run_ladder_perf_fpga.py` logs the effective levels and **hard-fails** on any capability/baseline mismatch, naming the rungs and telling you to set `LADDER_OPT` on the runner. | mismatch path exercised; runner parses; levels appear in the run log |

### Fixed 2026-07-27 (daytime)

| id | issue | fix |
|---|---|---|
| C-6 | CodeGenPrepare zero-extended a **negative** address offset into the 128-bit pointer carrier (`AddrMode.BaseOffs` is `int64_t`, `ConstantInt::get` defaults to `IsSigned=false`). Produced a **wrong address**; latent on any wide-pointer target. | `/*IsSigned=*/true` at 3 sites |
| C-7 | `APInt::getSExtValue()` asserted on an i128 constant in `SelectionDAGAddressAnalysis::matchLSNode` | `fitsInOffset` guard at 3 sites |
| C-8 | `Cannot select: i128 = and` — the dispatch returned the constant-mask helper unconditionally, so its bail left the node unlowered | fall through to `lowerScalarI128Logical` |

Validated: Capstone lit 41/41, BEEBS 82/82, CoreMark, authority 32/32, RV8 −O0 5/5, full X86 +
RISCV lit (6 `emutls*` failures **verified pre-existing** by stash-rebuild-reproduce).

---

## How to add an entry

One heading per issue with: a one-line statement of the behaviour, a **runnable repro** (path or
tarball), the evidence note, what has been tried, and the impact. An issue without a reproducer is
a rumour — write the probe first. Every probe must be **QEMU-verified before the board** so a
board deviation is unambiguous, and must **return a diagnostic rather than hang** (a hung domain
reports nothing at all).
