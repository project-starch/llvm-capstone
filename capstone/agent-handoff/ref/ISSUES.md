# Open issues registry — RTL/FPGA and compiler

Single index of everything currently broken, with a pointer to a reproducer for each.
**Update this file whenever an issue is found, characterised, worked around or closed.**

Convention: **R-n** = RTL/hardware, **C-n** = our compiler/toolchain, **I-n** = infrastructure.
Status: `OPEN` · `CHARACTERISED` (mechanism known, unfixed) · `WORKED AROUND` · `FIXED` · `CLOSED`.

Last updated 2026-07-27.

---

## RTL / FPGA

### R-1 — A load through one capability register misses a store through another `CHARACTERISED`
**The blocker for 4 of 7 benchmark rungs.** An intervening store through one capability register
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
- **Predictive record (2026-07-27): 1 hit, 1 miss — R-1 is NOT complete.**
  Two rungs were written specifically to test its predictions.
  - `beebs_bs` — **predicted PASS, PASSED** (887447230 = oracle, 2264 cyc). This is the
    load-bearing confirmation: `bs_data[mid]` is a genuine register-indexed load through a
    derived capability — the exact addressing form in every failing rung — and it is correct
    because nothing is ever *stored* to the table. **The intervening store is a necessary
    ingredient**, not incidental.
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
- **Next probe (cheap, decisive):** a *minimal pure-register loop* — `long i=0; while (i<100) i++;`
  returned raw, with no memory access anywhere in the loop. If that fails on the board the finding
  is far larger than R-1. If it passes, janne's specific branch structure is implicated and the
  bisect should continue by simplifying the nest one condition at a time.

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

## Compiler / toolchain (ours)

### C-1 — `Cannot select: i128 = sign_extend_inreg` `OPEN`
An `int` index feeding capability address arithmetic fails ISel. Same family as C-5.
- **Repro:** `rawhazard3_fpga_app.c` with `int` (not `long`) index variables
- **Workaround:** use `long` indices. Applied in the probes only.

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

### C-5 — 4 KiB code window `OPEN`
`link-gpfree.ld` forces globals to image offset `0x1000`, capping `.text` at 4096 B. One
hardcoded number, QEMU-validated at 16 KiB and 32 KiB and silicon-validated at 32 KiB. Lifting it
is what full CoreMark and Dhrystone need. Task #62.

---

## Fixed (2026-07-27) — kept for provenance

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
