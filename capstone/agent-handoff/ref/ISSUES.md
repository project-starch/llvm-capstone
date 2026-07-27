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

### R-3 — Second domain at the same entry VA hangs within one boot `WORKED AROUND`
A domain reused at entry VA `0x10000` within a single boot silently hangs its `cscall` —
a missing icache invalidate on the domain switch. This forced **one full power-cycle +
JTAG firmware reload per rung** (~2.5 min), the dominant cost of every board sweep.

- **RESOLVED IN PRACTICE 2026-07-28: the fault is ADDRESS-KEYED.** Domains linked at
  *different* entry VAs run back to back in one boot. `beebs_bs` @`0x10000` then
  `beebs_prime` @`0x20000`, no power-cycle between them, both returned their oracles.
  Nobody had tested this; the per-rung power-cycle was an assumption, not a measurement.
- **Validated as measurement-safe, not merely correct.** The obvious risk was that a
  second domain runs with an icache warmed by the first, so cycle counts would not be
  comparable to the published first-domain numbers. A reversed-order control says no:

  | rung | as 1st domain | as 2nd domain | spread |
  |---|---:|---:|---|
  | `beebs_bs` | 2,258 / 2,246 | 2,263 | 0.75 % |
  | `beebs_prime` (−O1) | 9,746 | 9,749 | **0.03 %** |

  `instret` was byte-identical in both positions (875, 2,708).
- **A wedged rung poisons the rest of the sweep unless recovery is enabled.** On
  2026-07-28 `rv8_primes` hung and the runner kept "reusing" the dead boot, losing the
  **four** rungs after it — all of which had worked minutes earlier. Fixed: a rung that
  times out clears the boot flag so the next one power-cycles. One failure now stays one
  failure. Anyone re-implementing one-boot mode must include this.
- **How to use it:** `LADDER_DISTINCT_VA=1` on the build (assigns `0x10000`, `0x20000`, …
  64 KiB apart) **and** `LADDER_ONE_BOOT=1` on the runner. Both are opt-in: if the
  address-keying assumption ever fails the symptom is a silent hang that looks like a
  rung result, so this must not become a default without a control rung in the sweep.
- **Impact:** a 13-rung sweep goes from ~13 boots (~35 min) to **1** (~5 min).
- **Not a root fix.** The monitor still lacks the icache invalidate on domain switch, so
  same-VA reuse still hangs. Sidestepped, not repaired — the fix sketch remains in
  `plans/curried-crunching-gizmo.md`.
- **Mechanism note:** the domain-boundary `fence.i` was long suspected to fix R-1 as well;
  board test #63 disproved that. It remains the right fix for **this** issue only.

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

### I-2 — Linux baseline served interrupts inside the bracket `FIXED`
**Fixed 2026-07-28 by removing the OS**, not by modelling the error. The baseline now runs
as a bare-metal S-mode OpenSBI payload (`build-ladder-base-bare.sh`,
`fpga_driver/run_base_bare_fpga.py`).

- **Proof:** the `ctrsanity` control -- identical 5-instruction loop on both sides -- reads
  **600,041 cyc bare vs 600,309 cyc capability, ratio 1.000** (Linux was 728,727, 1.21x).
  Quality went from 1/15 passes tied at min instret to **15/15 with spread 0**.
- **Consequence: every published overhead ratio rose.** `beebs_prime` 1.032x -> **1.683x**,
  `rv8_primes` 1.050x -> **1.263x**, `beebs_recursion` 1.801x -> **1.955x**,
  `beebs_bs` 1.274x -> **1.530x**. Pervasive spatial safety costs **26-96 %**, not 3-5 %.
- **And it refuted a second claim:** with a clean baseline, `rv8_primes` cycles grow
  (1.263x) FASTER than instructions (1.130x) and CPI RISES 1.762 -> 1.970, inverting the
  "overhead is ABI, not enforcement" section.
- **Side benefit:** firmware 15.4 MB -> 2.1 MB, so the JTAG reload that dominates every
  boot is much faster.
- **Bring-up trail (3 silent board sessions):** legacy SBI console absent; DBCN impossible
  (board reports SBI 1.0, DBCN needs 2.0) and the probe read `a0` instead of `a1` anyway;
  fixed by direct ns16550a MMIO with parameters taken from the firmware's **device tree**
  (`/soc/uart@10000000`, `reg-shift=2`). **The FDT had the answer on disk the whole time.**
- **Evidence:** `history/28-07-2026_02-30-00_RESULTS-bare-metal-baseline-works-*.md`

---

## Compiler / toolchain (ours)

### C-2 — `Cannot select: i128 = or` / `= xor`, mixed extends `OPEN (partially widened)`
Blocks `rv8_qsort` and `rv8_miniz` at −O1/−O2 (both still fail 2026-07-28; −O0 passes).

**The semantics question was malformed, and the answer is now settled.** It was framed as
"do the high 64 bits mean capability metadata or a genuine 128-bit integer?" — neither.
`lowerScalarI128Logical` computes the op in XLen and re-extends, which is exact **only while
the i128 carrier's high half is an extension of its low half.** Matching extends preserve that
invariant. Mixed extends break it: for `sext(a) OR zext(b)` the true 128-bit high half is
`sign(a)`, which is **not a function of the low-half result**, so re-extending the narrow
result under *either* rule is a **miscompile**. **The bail is correct. Do not "fix" it by
picking an extension rule.**

- **Widened safely 2026-07-28** (`CapstoneISelLowering.cpp`): when the sign-extended operand is
  **known non-negative** (`DAG.SignBitIsZero`), its sign extension and a zero extension are the
  same bits, so both operands agree and the invariant holds. Covers indices/sizes the optimizer
  has already proven `>= 0`, without assuming anything about meaning.
  Lit `i128-logical-mixed-extend.ll`; **Capstone lit 43/43**.
- **Does NOT unblock rv8.** Re-verified with exit codes: `qsort` −O1/−O2 still
  `Cannot select: i128 = xor`, `miniz` still `i128 = or`. Their signed operand is not provably
  non-negative, so they are the genuinely unrepresentable case.
  > ⚠ An intermediate report that both benchmarks "now build" was **wrong** — that check
  > grepped output for error strings without testing the exit code, so a failing build read as
  > success. Always gate on exit status.
- **What the real fix needs, and why it is not a lowering patch:** the remaining case cannot be
  represented while i128 is carried in a single capability register. Either (a) genuine
  128-bit integers get a register-pair representation distinct from the capability carrier, or
  (b) find why a **64-bit** `or`/`xor` is being widened to i128 at all — if the source only does
  64-bit logic, the i128 node is an artifact upstream of this lowering and should be prevented
  rather than lowered. **(b) is the cheaper investigation and should come first.**

### C-3 — RV8 fails at runtime at −O1/−O2 `OPEN`
**Now also reaches the ladder (2026-07-28):** the `rv8_primes` *rung* runs at −O0 and
**HANGS at −O1** on silicon, so it is the one row in the overhead table that cannot be
measured at the uniform level. Same family as the RV8 −O1/−O2 failures below.
Five RV8 benchmarks now *build* at −O1/−O2 but fail 10/10 at runtime: `primes`/`aes`/`dhrystone`
hang silently; `sha512`/`norx` take deterministic capability faults (cause 5 OOB / cause 24, same
PC at both levels). −O0 controls all pass. **Not regressions** — code that never compiled cannot
regress.
- **Evidence:** `history/27-07-2026_12-59-35_three-codegen-fixes-*.md`
- **Leads:** `sha512` faults with bounds visibly too small; `norx` with an untagged capability
  reaching a load. Both smell like a bounds/provenance codegen bug at −O1+.

### C-4 — split into a FIXED half and a remaining domain-creation bug
Renamed from "large read-only data cannot be delivered": size was never the variable.

#### C-4a — constant pools are unreachable in a domain `FIXED 2026-07-28`
**Root cause, with the emitted sequence:**
```
.LCPI0_0: .quad 81985529216486895        ; .rodata.cst8 -- a CONSTANT POOL entry
  auipc a2, %pcrel_hi(.LCPI0_0)
  addi  a1, a2, %pcrel_lo(...)
  scc   a1, gp, a1     ; set gp's cursor to a .rodata address
  ld    s6, 0(a1)      ; FAULTS
```
A pool entry is **not** a `GlobalVariable`, so it gets no cap-table slot (correctly);
`lowerConstantPool` then falls back to `LGA` → `scc gp`. Under gp-captable `gp` is bounded
to the **cap table itself**, so the cursor lands out of bounds. The tell in the fault line
is that the reported bounds are exactly the table:
`cursor = 0x101561000, bounds = (0x10157ffd0, 0x101580000)`.

**Fix:** `CapstoneSubtarget::useConstantPoolForLargeInts()` returns **false** whenever the
gp-free/gp-captable ABI is active, so the constant is materialised inline instead. Forming
a pool in a domain is always a miscompile, never an optimisation — the same reason
`-fno-jump-tables` is already mandatory (a jump table is `.rodata` too).

**Validated:** the previously-faulting `rv8_sha512` configuration now returns its oracle
(`__CAPSTONE_LADDER_RV8_SHA512_PASSED__`); 0 `.LCPI` entries remain in the emitted asm;
Capstone lit **43/43**; `beebs_bs`, `beebs_prime`, `beebs_cnt` still pass QEMU parity.

> **Two wrong turns on the way, both worth remembering.** First this was called a
> *large-data delivery* problem, because bigger constants are the ones that get pooled.
> Then, on seeing that all named globals DID have cap-table slots, the constant-pool
> explanation was **retracted as refuted** — but the faulting object was never a global,
> so the descriptors could not have refuted it. The lesson is to identify the faulting
> OBJECT before reasoning about the mechanism: a symbolised `-S` listing settled in one
> step what two rounds of inference got wrong.

#### C-4b — the large-RO COPY PATH in the generated glue is broken `OPEN — trigger identified`
**Not a domain-creation bug, and not about size.** Earlier notes here (now corrected) chased
image geometry through the loader and kernel module. That was the wrong component:

> `Created domain ID = 0` appears **before** the assertion in the serial log. Domain
> creation **succeeds**; `helper_cssplit: rs1_v->tag && !rs2_v->tag` fires afterwards, in
> the **entry glue**.

**The actual trigger is a threshold in the glue generator, not a size limit.**
`gen-gp-captable-glue.py` has `COPY_THRESHOLD = 256` and picks between two paths:

| initializer size | glue path | result |
|---|---|---|
| 640 B (`sha512_k[80]`) | **large-RO copy loop** (`stor > 256`) | **FAILS** |
| 128 B (`sha512_k[16]`) | unrolled `li`/`sd` immediates (`stor <= 256`) | **passes** |

So every "size-dependent" symptom was just this threshold selecting a different code path.
The large-RO copy path is the thing that is broken; it is emitted for exactly one global in
the ladder today, which is why nothing else has hit it.

**The suspect sequence** (from the generated `.inc`):
```
lla t4, sha512_k
lla t5, __gpfree_globals_base
sub t5, t4, t5               /* blob offset = sym - base */
cincoffset(t4, sp, t5)       /* src */
cincoffset(t3, t2, x0)       /* dst */
```
`lla` on a Capstone target may not yield a plain integer, so `sub` of two such values --
and hence the operand feeding a later `split` -- is where a stray tag most plausibly comes
from. **Verify by dumping tags, not by reading:** that inference is exactly the kind that
has been wrong three times on this issue.

**Refuted along the way, recorded so nobody repeats them:** (a) `tot_size` invariant --
both images give `tot_size` 8192 and satisfy `tot_size > code_size + 1536`; (b) `code_len`
carrying the exec segment -- it is `image_size`, the whole loadable image
(`libcapstone.c:197`); (c) `dom_pages_log2` rounding -- it rounds **up** correctly
(`dom_pages == 1 ? 0 : ilog2(dom_pages - 1) + 1`).

**Experiment RUN (2026-07-28): the unrolled path is not a viable stopgap, and C-4b is
entangled with C-5.** Raising `COPY_THRESHOLD` above 640 so the big table takes the
unrolled `li`/`sd` path fails at link time:

```
ld.lld: error: unable to place section .text at file offset [0x1000, 0x2E77]
```

`.text` reaches **11,895 B** against the 4 KiB window — 640 B of data costs ~8 KB of
immediate-materialisation code, exactly the reason the copy path exists. So:
- The copy path is **necessary**, not an optimisation — it cannot simply be disabled.
- **C-4b cannot be worked around without first lifting C-5** (the 4 KiB window), or by
  fixing the copy path itself.
- Threshold reverted to 256; no code change kept from this experiment.

**So the remaining work is to fix the copy path directly.** Dump capability tags through
the `lla`/`lla`/`sub`/`cincoffset` sequence to find where a tagged value reaches an operand
that must be untagged. Do not infer it from reading — inference has been wrong three times
on this issue.

**Related hazard — CHECKED 2026-07-28, NOT a bug.** `getGpCaptableIndex` derives its index
from a global's *position* in `M.globals()`, and GlobalMerge mutates that list (it merged
`sha_chain` + `sha_w` into one 192 B entry here), which raised the possibility of an access
lowered against the pre-merge order loading the **wrong capability slot** — silent wrong
data rather than a fault. It cannot happen: **GlobalMerge runs in `addPreISel`**
(`CapstoneTargetMachine.cpp`), i.e. before instruction selection, so `lowerGlobalAddress`
during ISel and `emitGpCaptableTable` in the AsmPrinter both see the same post-merge list.
Confirmed empirically as well — the merged-global `rv8_sha512` build and the 6-global
`beebs_cnt` both return their exact oracles, which mismatched indices would break.
**Recorded because the reasoning is the useful part:** any future pass that adds or removes
globals *after* ISel would silently break this positional scheme.

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
