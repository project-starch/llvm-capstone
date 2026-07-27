# Current Capstone state

Minimal snapshot. Read first in every session.

## Latest (2026-07-27) — silicon status: 3 rungs measured, 2 hang, blocker still open

**Paper-facing source of truth is `ref/fpga-silicon-measurements-for-paper.md`
(§5 is the authoritative "what is NOT established" list). This section is the
short state; the dated `history/` notes are the trail.**

### The perf table as it stands (these 3 are quotable)

Method: each kernel built twice from the identical source header by the same
clang at the same `-O` — once `-target capstone64` as a pure-cap domain, once
`-target riscv64` with no capability flags — run on the same board, compute-only
bracket, warm baseline. Static gate fails the build if a capability instruction
reaches the baseline.

| rung | opt | capability cyc | baseline cyc | **cycles** | **instr** |
|---|---|---:|---:|---:|---:|
| `beebs_prime` (pure scalar) | −O0 | 47,780 | 46,306 | **1.032×** | — |
| `rv8_primes` (sieve, 16.5 M cyc) | −O0 | 17,283,292 | 16,459,057 | **1.050×** | **1.102×** |
| `beebs_recursion` (deep + mutual recursion) | −O1 | 18,957 | 10,523 | **1.801×** | **1.458×** |

**Headline: pervasive spatial safety costs 3.2 % scalar / 5.0 % array / 80 %
recursive. The spread IS the result — report the range and the mechanism, never
an average.** The recursion outlier is **ABI cost** (gp-free call/return + cap
spills), not hardware.

Two further results that upgrade the paper: measured **CPI 2.0–3.2** (the draft
assumed 1, so `tab:appoverhead`'s SQLite figures roughly halve), and on
`rv8_primes` **+10.2 % instructions but only +5.6 % cycles with a *lower* CPI** ⇒
capability enforcement is near-free per instruction; the overhead is the
gp-captable ABI. That is one benchmark — caveat it.

### 2026-07-27 — two blocked rungs UNBLOCKED by compiler fixes (board-free)

`beebs_crc32` and `beebs_insertsort` now **build at −O0/−O1/−O2 and pass the QEMU parity leg**,
taking the ladder from 3 to 5 *buildable* rungs. **They were then measured on the board and BOTH
FAIL — the measured set stays at 3.** `beebs_crc32` hangs at −O1; `beebs_insertsort` returns
957879052 against an oracle of 271779359 with only 560 retired instructions, i.e. the compute
never ran. Both were already wrong on silicon at −O0 in the 25-07 sweep, so the compiler fixes
were **necessary but not sufficient** — they removed the build blocker and exposed the same
unexplained silicon divergence underneath. Not a regression. Trail:
`history/27-07-2026_15-48-02_RESULTS-the-two-newly-buildable-rungs-fail-on-silicon-too.md`.
The fixes themselves remain worth having:

1. **`beebs_crc32` was never a compiler bug.** The kernel generates its CRC table at runtime to
   avoid a large initialized global; −O1+ **constant-folds the loop** and re-materialises a
   2048 B *private* constant `.L.crctable`, which the cap-table glue cannot deliver (over the
   12-bit unrolled path, and the large-RO copy path needs a *linkable*, non-`.L` symbol visible
   from the glue's separate TU). Fixed by making the polynomial opaque to the optimizer — one
   line, no runtime change. **Generalises: any hand-rolled table meant to dodge the large-RO
   limit can be silently undone at −O1+, SQLite included.**
2. **`beebs_insertsort` — the clang crash was hiding a real defect.** Guarding an
   `APInt::getSExtValue()` assert in `SelectionDAGAddressAnalysis` exposed
   `Constant:i128<0xFFFFFFFFFFFFFFFC>` — **CodeGenPrepare zero-extends a negative address
   offset** into the pointer carrier (`AddrMode.BaseOffs` is `int64_t`, `ConstantInt::get`
   defaults to `IsSigned=false`). Invisible on ≤64-bit-pointer targets; on a 128-bit capability
   `−4` becomes a huge positive offset. It was producing a **wrong address**, caught only by our
   backend's fatal guard. Latent for any wide-pointer target, CHERI included.
3. **`i128 = and` was unlowerable** — the dispatch `return`ed the constant-mask helper
   unconditionally, so its bail left the node unlowered instead of falling through to the general
   path OR/XOR use.

**RV8 is NOT fixed — do not quote "0/7 → 5/7".** Five RV8 benchmarks now *compile* at −O1/−O2,
then **fail 10/10 at runtime** (3 silent hangs; `sha512`/`norx` take deterministic capability
faults, cause 5 OOB and cause 24, same PC at both levels). −O0 controls all pass. These are not
regressions — you cannot regress code that never compiled — they are pre-existing −O1+ codegen
defects newly exposed, and root-causing them is the next real compiler task.

**Regression status: clean.** Capstone lit 41/41, BEEBS 82/82, CoreMark, authority 32/32, RV8
−O0 5/5, full X86 + RISCV lit. The only failures are 6 `emutls*`/`tls-android` tests, **verified
pre-existing** by stashing the changes, rebuilding `llc`, and reproducing them identically.
Trail: `history/27-07-2026_12-59-35_three-codegen-fixes-unblock-two-ladder-rungs-and-rv8-at-O1.md`.

### The other 4 rungs

| rung | status on silicon |
|---|---|
| `matmult_int` | **HANGS** the `cscall` at −O1/−O2 — no result at any reachable config |
| `coremark_matrix` | **HANGS** at −Os and at −O0 @32 KiB — localized to `core_init_matrix` (#66) |
| `beebs_crc32` | cannot **build** at −O1+ (2048 B folded table overflows a 12-bit store offset) |
| `beebs_insertsort` | **crashes clang** at −O1 |

### RETRACTED — do not carry these forward

The 2026-07-25 sweep table below this section reported **4 rungs miscomputing**
under an "array-store-with-live-accumulator" framing. **Both the framing and that
rung classification are withdrawn:**

- The rungs contain **zero `shrink`** instructions, so the documented
  `shrink`→store root cause cannot apply; and `beebs_recursion` has no array at
  all. Bounds-representability is refuted too (the rung with the *largest* global
  passes). **Do not escalate the shrink story to the board owner.**
- **"Scalar rungs pass, array rungs fail" is too strong.** A controlled A/B showed
  two builds of the same rung differing only in `domain_main` — *with* the minstret
  instrumentation `beebs_prime` returns 1087631800 (wrong, deterministic across two
  sessions); *without* it, 582955588 = the oracle. **Four instructions, none inside
  the computation, flip a passing rung.** A passing rung is not stable ground —
  re-gate on the oracle after ANY domain change.
- **"Domain-entry fault" is dead** (#63, `LADDER_INSTR_MODE=7`): the entry path runs
  and both hanging rungs complete a full domain round-trip when the compute is
  branched over. The domain-boundary `fence.i` (#61) is therefore the wrong layer.
- **"Fragile `bne` loop exits" is dead** (#65). It was observed statically that
  `matmult_int` at −O1 emits 8 conditional branches **all `bne`** while −O0 emits 8
  **all `blt`**, suggesting one fault whose symptom the branch kind selects. A −O1
  build with ordered exits forced — verified 0 fragile / 8 ordered, QEMU-correct
  through the *same* controller — **still hangs, identically.** The codegen split is
  real but is a **correlate, not the cause**. Do not restate "one fault, two
  symptoms".

**The pattern to inherit:** two hypotheses died in two days, both by promoting a
strong *static* correlation to a *mechanism* before a board test could speak. At
~2.5 min/boot with days left, prefer a **bisect that needs no mechanism guess**
(mode 7 and #66 paid off; #65 did not).

### RESOLVED 2026-07-27 (board #67a–#67f) — `delin` in domain code wedges the RTL

**`coremark_matrix`'s first fault is NAMED, with a size-matched control.** Six boots,
each build QEMU-correct through the identical controller first. Full trail:
`history/27-07-2026_04-33-58_RESULTS-delin-wedges-the-RTL-controlled-and-second-fault-isolated.md`.

| probe | delta | board |
|---|---|---|
| #67a | while loop only | **RETURNS 9** |
| #67c | + **`delin`** (one instruction) | **HANGS** |
| #67e | #67c with `addi x0,x0,0` **instead** (size-matched) | **RETURNS 9** |
| #67f | `B = A + N*N`, **no `delin`** | **RETURNS 9** |
| #67d | **full** benchmark, **no `delin`** | **HANGS** |

1. **The `delin` opcode is the fault, not code layout.** #67c and #67e differ only in a
   4-byte instruction's *encoding* — same position, same `"+r"(A)` plumbing. This control
   was mandatory: the 26-07 A/B showed 4 added instructions can flip a rung.
2. **Not "`delin` is unimplemented".** The glue `delin`s several caps in *every* domain and
   passing rungs work. The difference is the operand: glue delins a cap **fresh from
   `split`**; domain code delins one **loaded by `ldc` from the cap-table** — which the glue
   already delin'd before `stc`, so on a type-preserving machine it is **NONLIN→NONLIN**.
   That is exactly the case `capstone-qemu` `f4d416c265` patched to be idempotent
   *"rather than faulting"*. Same QEMU-permissive / RTL-enforces shape as `C_GEN_CAP`.
   **Caveat:** instrumented QEMU reports that operand as **LIN**, so QEMU and the glue
   disagree about type after `stc`→`ldc`. Which side is wrong is a **board-owner question**.
3. **Dropping the `delin` is safe but insufficient.** #67f returns (the `rd != rs1`
   derivation does not consume `A` on hardware) and QEMU still gives 14343 — but the full
   rung still hangs (#67d). **≥2 independent faults.** Fault 2 is in the **seeding loop or
   later**, which revives the surviving static candidate: `coremark_matrix` is the only rung
   doing **narrow (`sh`) accesses through the block cap**. Next: phase-bisect inside the
   seeding loop, or widen `MATDAT` to 32-bit.
4. **`matmult_int` has no `delin` at all** — fault 1 cannot explain it. Still possibly two bugs.
5. **A minimal silicon repro now exists** (two 4-byte instructions, both QEMU-correct) — the
   *paper-acceptable* outcome: a documented hardware limitation, not an unexplained one.

### What survives, cumulatively

- The hang is **inside the compute**, not at domain entry.
- For `coremark_matrix` it is inside **`core_init_matrix`** — bisected against mode 7
  at the same −O0 @32 KiB config: entry-only **RETURNS**, entry + `core_init_matrix`
  **HANGS**, everything **HANGS**. That is one ~40-line function. Two candidates
  remain, **not yet separated**: the dimension loop
  `while (j < blksize) { i++; j = i*i*2*4; }` (`bgeu` `0x10428` / `mulw` `0x10444`),
  and the N×N seeding loop running `seed = ((order*seed) % 65536)` per element,
  writing `A[]`/`B[]` **through the gp-delivered block capability**.
- It is **not** the loop-exit condition, and **not** discriminated by instruction
  mix (M-extension ops included, re-checked properly), code size, global count, or
  `.bss` size.
- **Do not assume the two hanging rungs share a mechanism.** `matmult_int` has no
  data-dependent bound at all; `coremark_matrix` is built around one.
- **Three further framings refuted board-free (2026-07-27, lane C** —
  `history/27-07-2026_02-45-07_core_init_matrix-codegen-audit-three-framings-refuted.md`**):**
  (a) *"an extra capability load/store in a loop is the trigger"* — `rv8_primes`
  reloads its block cap from the cap-table **and** stores through a dynamically
  derived cap in its hottest loop, and is silicon-correct; (b) *"the block cap gets
  round-tripped through memory"* — the **passing** `beebs_prime` spills and reloads
  its block cap; (c) *"a redundant NONLIN→NONLIN `delin` faults on the RTL"* —
  instrumented QEMU shows **zero** redundant delins in the whole coremark run, so
  the cap is genuinely LIN at that site and the in-kernel `delin` is necessary.
  Also: at −O0 `core_init_matrix` keeps **no** live capability across the loop — it
  reloads **both** `A` and `B` from stack slots every iteration.
  **Surviving candidate, for `coremark_matrix` only:** it is the sole rung doing
  **narrow (`sh`/`sb`/`lh`/`lb`) accesses through the block capability** (4 stores +
  9 loads at −Os); all three passing rungs use word-or-wider only, and `matmult_int`
  has none — so it cannot be a shared mechanism. Treat as a candidate, not a cause.
  **⚠ Probe #67 as specified is a 3-way, not a 2-way:** the `delin` + `B = A + N*N`
  derivation block sits *between* the dimension loop and the seeding loop, so
  "return `N` before the seeding loop" leaves two candidates on its HANG branch.
  Move the split point before the `delin`, or make it 3-way.
- The corruption is a **silicon divergence** — QEMU runs the identical binaries
  correctly — and is **NOT proven a compiler bug**. If our code is ISA-legal and
  QEMU-correct, this is an RTL divergence to hand to the board owner with a minimal
  repro: a **paper-acceptable** outcome (documented hardware limitation).

Trail: `history/27-07-2026_00-58-47_RESULTS-65-falsified-66-localizes-hang-to-core_init_matrix.md`,
`history/27-07-2026_00-28-51_loop-exit-condition-splits-hang-from-miscompute.md`,
`history/26-07-2026_23-56-07_the-hang-is-in-the-compute-not-at-domain-entry.md`,
`history/26-07-2026_17-43-17_controlled-ab-four-instructions-flip-a-passing-rung.md`,
`history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`.
Memory `project_gp_captable_codegen`.

### Tooling traps that silently corrupt this analysis

- **The Capstone-triple disassembler cannot decode M-extension instructions.**
  Domains build `-Xclang -target-feature -Xclang +m`, but `llvm-objdump` on a
  `capstone64` binary prints every `mul`/`div`/`rem` as `<unknown>`. Any
  mnemonic-keyed analysis must pass `--triple=riscv64 --mattr=+m`. (Re-run properly,
  the "no discriminating instruction" conclusion still **stands** — a trap, not a
  retraction.)
- **`<sym+0xNN>` in disassembly is not a branch target.** Regexes grabbing the last
  hex number on the line invert forward/backward branch classification. Strip `<...>`.
- **At −O0 clang emits a forward exit test plus an unconditional `j` backedge.** A
  metric counting only *conditional backedges* reports zero for every −O0 build.
- **A domain that hangs reports nothing at all** — the controller prints `res[]` only
  after the `cscall` returns, so "write a marker and read it back" probes are unusable
  on a hang. Design probes around *does it return at all*.

---

## Latest (2026-07-26) — xlang cross-language repro corpus (separate track from the board work)

**14 of 15 rows reproduce; 12 of 15 reproduce the temporal-borrow class the
benchmark is about.** All 15 `run.sh` pass and assert their expected outcome.
Stock toolchain only — no Capstone compiler, no QEMU fork, no board.

- Artifacts: `xlang/` (start at `xlang/README.md`).
- Full state, evidence and open decisions:
  `history/26-07-2026_18-04-21_xlang-phase1-state.md`.
- **Do not quote "14/15" for the temporal benchmark** — rows 6 and 11 reproduce as
  *spatial* heap-buffer-overflows, not UAFs, which contradicts the companion note's
  "every defect here is a temporal borrow" claim. Row 7 does not reproduce and
  appears not to exist as specified.
- Corpus is no longer monolingual: Lua↔Rust 2/2 and Rust→C 1/1 now reproduce, so
  the paper's "two subjects" framing is backed by artifacts for the first time.

## Superseded (2026-07-25) — silicon-ladder perf sweep, original table

**Kept for provenance. Its rung classification and explanation are RETRACTED by the
2026-07-27 section above — read that first.**

| rung (fresh dom) | silicon | oracle | mcycle | verdict as reported then |
|---|---:|---:|---:|---|
| rv8_primes | 99991 | 99991 | 17,283,292 | ✅ PASS |
| beebs_prime | 582955588 | 582955588 | 47,804 | ✅ PASS |
| matmult_int | 1166210317 | 774662735 | 76,498 | ❌ reported miscompile |
| beebs_crc32 | 1568735421 | 1703161001 | 311,902 | ❌ reported miscompile |
| beebs_insertsort | 255001740 | 271779359 | 10,463 | ❌ reported miscompile |
| beebs_recursion | 2095861164 | 1579141629 | 30,263 | ❌ reported miscompile |
| coremark_matrix | — | 14343 | — | transfer never landed |

Each was verified on a dom rebuilt after the 24-07 memcpy fix (`d078839`) and each
was **QEMU-correct** with that same fresh binary — that part stands, and it is why
this is a silicon divergence rather than a build artifact. `beebs_insertsort`'s
255001740 coinciding with the pre-fix memcpy signature was a **red herring**.

Two process findings from that sweep, both still valid:

1. **The runner could run stale binaries** — it reused pre-built `.dom`s and read a
   different dir than the build script wrote. Now rebuilds-by-default + hard-fails on
   stale (`4be78cb`/`bd03316`). It did not explain any of the miscompiles.
2. **Board transfer improved** (`fast_xfer`: Ctrl-C resync to escape the `> `
   continuation prompt a dropped char leaves; catch the wedge timeout and escalate
   instead of aborting; third slower tier). This recovered 2 of 3 previously
   unverifiable rungs. `coremark_matrix` was later shown **not** transfer-blocked.

Full table + mechanics + correction trail:
`history/25-07-2026_03-58-47_fpga-ladder-perf-sweep-results.md`.

Runner: `tests/rtl-smoke/fpga_driver/run_ladder_perf_fpga.py` — one full
power-cycle + JTAG reload per rung (each rung runs as first domain / clean icache;
warm `reset halt` does NOT re-enter OpenSBI), tier-1 `fast_xfer.fast_put`
transfer, `insmod /capstone.ko` (UP image doesn't auto-load it). The `-b` LLVM was
rebuilt from scratch with `-capstone-gp-captable` (system `/usr/bin/clang++`,
`RISCV;Capstone`); all 7 perf domains build `cjalr=0 ldc-gp≥1`.

## Latest (2026-07-24) — CoreMark matrix on the silicon ladder (QEMU)

CoreMark 1.01's **matrix** benchmark now runs as silicon-ladder **rung 7** in a
pure-cap domain on QEMU: domain crc16 `14343` == native `cc -O0` oracle, static
gate `cjalr=0 ldc-gp=1`, `__CAPSTONE_LADDER_COREMARK_MATRIX_PASSED__`. Files in
`tests/runtime-qemu/silicon-ladder/coremark_matrix_{kernel.h,app.c,host.c}` +
`run-coremark-matrix-qemu.sh`. Matrix only (list/state CRCs are pointer-size-
dependent → wouldn't match a native oracle); driven standalone with CoreMark's
validation-run matrix params (N=9). Built `-Os` (pinned in the wrapper): CoreMark
matrix is ~4.7 KiB `.text` at `-O0` and overflows the 4 KiB PCC window; ~1.5 KiB
at `-Os`. **Note:** the `-b` clang is stale (predates the merged
`-capstone-gp-captable` flag); validated with a sibling checkout's current clang
driving the `-b` runtime — the `-b` LLVM build config was restored to shared +
`clang;lld` but the rebuild is deferred. Trail:
`history/24-07-2026_14-14-09_coremark-matrix-silicon-ladder-rung.md`.

## Latest (2026-07-22) — gp-free domain bring-up (silicon-shaped ABI)

On branch **`capstone-gp-free`** (off `capstone-bootstrap`; not merged/pushed): a
real globals-using integer app now runs **correctly** in a pure-capability domain
**gp-free / cjalr-free** on QEMU with the `gp = PCC(cursor 0)` fabrication
**disabled** — `gp` is an image-covering data cap the **monitor** delivers via the
cscratch stack region (board owner's confirmed channel; same as `capstone-c`).

- **Compiler `-capstone-gp-free`** (default off, byte-identical off; lit 40/40):
  plain `jal`/`jalr` calls/returns within PCC (no `cjalr`); global data via `SCC`
  (absolute in-bounds cursor) not `cincoffset gp` (which needs the unrepresentable
  cursor 0). Files: `CapstoneAsmPrinter.cpp`, `CapstoneISelDAGToDAG.cpp`
  (`selectCall`), `CapstoneExpandPseudoInsts.cpp` (`expandCapGlobalBase`).
- **Monitor** `create_domain` mints `gp` with `C_GEN_CAP` + stashes it at the
  cscratch region top slot; **glue** `start-gpfree-cscratch.S` loads it. **QEMU**
  `op_helper.c` gates the 4 gp-fabrication sites behind `CAPSTONE_GP_FABRICATE`
  (default on) + a `CAPSTONE_GP_STANDIN` monitor stand-in.
- Proof + repro: `tests/runtime-qemu/gp-free-domain/` (`build-and-run.sh` →
  `__CAPSTONE_GPFREE_DOMAIN_PASSED__`); default domains still pass with the rebuilt
  monitor. Trail: `history/22-07-2026_16-09-12_gp-free-domain-bringup-qemu-proof.md`;
  guidance memory `project_silicon_gp_delivery_boardowner_guidance`.
- **Remaining:** same `create_domain` change on the FPGA (caplifive-system) copy +
  board image rebuild + a silicon smoke/cycle run (Experiment A). QEMU + monitor
  submodule edits kept as local experiments (no submodule-source commits).

## Latest (2026-07-15) — read this first; sections below predate it

Since 2026-07-03 the active work shifted from C1/C2 to the **performance
reframe** (2026-07-13): eager CHERI matches our temporal security, so the
separating axis is **performance**. That comparison is now **DONE** and in the
paper.

- **CHERI-vs-Capstone temporal-safety perf comparison — DONE (QEMU-to-QEMU, two
  workloads).** Eager CHERI (the config that matches our security) pays
  **~14–17 M instr per free** (address-space sweep); our revoke-at-free is
  **O(1), +5 instr/op**; async CHERI is 1.9–6.4× but blocks **0/11** UAF at the
  contract point. Paper `evaluation.tex` §`sec:eval-perf-compare` filled
  (`tab:perfcompare` microbench + `tab:perftree` real-workload BST). CHERI stack
  is fully local at `~/cheri` (`tests/cheri-perf/`, `tests/cheri-baseline/`).
  Full report: `history/15-07-2026_00-20-00_cheri-capstone-perf-comparison.md`;
  plan `plans/perf-cheri-vs-capstone-qemu.md`.
- **`-O2`/`-O1` capability-select ICE — FIXED (2026-07-15).** `lowerSELECT`
  crashed on an i128 cap select with non-null constant arms; fixed in
  `CapstoneISelLowering.cpp` (rematerialize constant arms as `li` via
  CopyToReg). This unpinned the Capstone BST tree probe from `-O0`; it now builds
  **and runs clean at `-O2`** (revoke-at-free +5, matching the microbench).
  Backend lit 39/39, clang 6/6, authority **26/26**. Trail:
  `history/15-07-2026_03-43-21_cap-select-o2-ice-fixed.md`.
- **Nightly orchestrator added:** `capstone/tests/run-nightly.sh`
  (build → lit → QEMU suites serially → report to `/tmp/capstone/`).
- **Corrections to the sections below:** the authority suite is now **26 domains**
  (not 20); `-capstone-shrink-stack` is **default ON** since 2026-07-03 (covering
  varargs save-area + dynamic alloca, so those are no longer "not yet"); the
  task-005 FastCC-i128 and revoke-intrinsic-DCE codegen defects are **resolved**.
- **Standing next step:** the Capstone **RTL cycle-accurate** number
  (human-in-the-loop; **postponed** pending the board owner's answer on automation).

## SQLite in-memory bring-up

SQLite 3.53.3 compiles, links, **and runs end to end** as a
`capstone64-unknown-elf` pure-capability domain using memsys5 over the static
arena and the runtime-initialized SQLite VFS skeleton. `run-sqlite-memory.sh`
executes `CREATE TABLE` / `INSERT` / `SELECT` and the domain returns correct rows
(`row name=alpha value=11 / beta=22 / gamma=33`, `__CAPSTONE_SQLITE_MEMORY_PASSED__`).
The pinned fetch/build/run workflow is in `capstone/benchmarks/sqlite/README.md`.

**Bring-up is complete — all 8 gaps resolved:**
- Gaps 1–2 (compiler): `CapstoneCapGlobalInit` recurses nested global aggregates
  (#71); clang memcpy-from-private-template of cap aggregates handled (#72).
- Gaps 3–4 (QEMU): untagged `ldc`/`stc` made bit-preserving over the full 128-bit
  word, enabling a tag-preserving `memcpy` (#73/#74).
- Gap 5 (compiler ISel): `cscincoffset` int+ptr operand order (#79).
- Gap 6 (SQLite alignment): 16-align `sqlite3NestedParse`'s `saveBuf` so the
  tag-preserving `memcpy` fast path carries Parse-tail caps (#80).
- Gap 7 (compiler): materialize interior-pointer capability globals
  (`&global[N]`) — `sqlite3aLTb/aEQb/aGTb` (#81).
- Gap 8 (SQLite alignment): 16-align the `BtCursor` embedded by `allocateCursor`
  (#82).

Full per-gap detail in `history/` (dated notes) and
`design/sqlite-gap6-memcpy-tag-preservation-proposal.md`. Follow-ups: the SQLite
8-byte-alignment class (gaps 6/8) may surface more instances under wider workloads.

**In-domain cap-fault delivery — abort retired (2026-07-03).** QEMU no longer
aborts on an in-domain capability fault: `riscv_cpu_do_interrupt`'s
`assert(env->priv < PRV_C)` is replaced (for `env->priv == PRV_C`) by a clean halt
— a structured `[CAPSTONE] domain halted by capability fault: cause=…` line then
`fflush`+`exit(0)`. This preserves the domain's serial output (`abort()` didn't
flush stdio — the gaps 8/9 "no serial output" cause) and turns a SIGABRT into a
named halt. The monitor host-trap path (`priv < PRV_C`) is unchanged. Validated:
full authority suite all-PASS, SQLite base+extended PASS, no abort in logs. Step A
proved the `ctvec` horizontal-trap path can't deliver this (a domain installs no
`ctvec`). **Return-to-host** delivery (domain terminates, host continues) is the
remaining, monitor-side step — see
`design/domain-fault-delivery-proposal.md` + `history/03-07-2026_00-00-03_*`.

## Verified baseline

All of the following pass on the `capstone-bootstrap` branch:

- LLVM Capstone backend builds the sample domain; `ld.lld` links native `EM_CAPSTONE`
- `capstone/caplifive-buildroot/build/local.mk` present — keeps the image on the Capstone-enabled OpenSBI path
- All HostCall probes pass: shared-region, stdout, filewrite, fileread, full file-handle
  lifecycle (open/write/read/sync/stat/truncate/close), path ops, combined file-object
- `run-nullblk-baseline.sh`, `run-nullblk-split-io.sh`, and
  `run-nullblk-split-rmmod.sh`
- `run-hostcall-all.sh`, `run-nullblk-all.sh`, and `run-all-beebs.sh` provide
  aggregate gates for reproducible full reruns; keep individual wrappers as the
  diagnostic entry points. The HostCall, `null_blk`, and full BEEBS aggregates
  have passed end to end; BEEBS has also passed with `RUN_ALL_BEEBS_JOBS=4`.
  `run-all-beebs.sh` is serial by default
  (`RUN_ALL_BEEBS_JOBS=1`) and has opt-in isolated parallelism via
  `RUN_ALL_BEEBS_JOBS=N`. It keeps child output in per-benchmark logs by default
  and prints compact pass/fail lines; set `RUN_ALL_BEEBS_VERBOSE=1` for streamed
  child output. It retries structured QEMU infra flakes before benchmark
  execution twice by default (`RUN_ALL_BEEBS_BOOT_RETRIES=0` disables this) and
  caps aggregate boot-to-login waits at 90 seconds by default
  (`RUN_ALL_BEEBS_LOGIN_TIMEOUT`), but does not retry benchmark marker failures.
- QEMU runtime smoke tests use snapshot mode, so repeated runs do not mutate `rootfs.ext2`
- Buildroot getty is pinned to `ttyS0`, avoiding intermittent boot-to-login hangs through `/dev/console`
- QEMU runtime smoke tests force `-smp 1`, avoiding intermittent boot stalls under the current OpenSBI/QEMU setup
- `run-coremark.sh` - all three algorithms, "Correct operation validated."; CoreMark now uses
  compiled C `domain_main`, not `coremark_domain_entry.S`
- `capstone/benchmarks/beebs/run-beebs-fac.sh` - first BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-insertsort.sh` - second BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fibcall.sh` - third BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cnt.sh` - fourth BEEBS benchmark runs end to end
  and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bubblesort.sh` - fifth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-prime.sh` - sixth BEEBS benchmark runs end to
  end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-recursion.sh` - seventh BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-janne-complex.sh` - eighth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-tarai.sh` - ninth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-cover.sh` - tenth BEEBS benchmark runs end
  to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-duff.sh` - eleventh BEEBS benchmark runs
  end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-levenshtein.sh` - twelfth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-jfdctint.sh` - thirteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fdct.sh` - fourteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-strstr.sh` - fifteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ndes.sh` - sixteenth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraybinsearch.sh` - seventeenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-queue.sh` - eighteenth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listinsertsort.sh` - nineteenth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-listsort.sh` - twentieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-expint.sh` - twenty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-compress.sh` - twenty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-md5.sh` - twenty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-cast128.sh` - twenty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-matmult.sh` - twenty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc32.sh` - twenty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-mergesort.sh` - twenty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-stringsearch1.sh` - twenty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-bs.sh` - twenty-ninth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-fir.sh` - thirtieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-lcdnum.sh` - thirty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ns.sh` - thirty-second BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ud.sh` - thirty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nsichneu.sh` - thirty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arraysort.sh` - thirty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayheapsort.sh` - thirty-sixth
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-arrayquicksort.sh` - thirty-seventh
  BEEBS benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-dllist.sh` - thirty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-hashtable.sh` - thirty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-aes.sh` - fortieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-picojpeg.sh` - forty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-sha256.sh` - forty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-huffbench.sh` - forty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-rijndael.sh` - forty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-crc.sh` - forty-fifth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-statemate.sh` - forty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-arcfour.sh` - forty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-nettle-des.sh` - forty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-aha-mont64.sh` - forty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-dijkstra.sh` - fiftieth BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-stack.sh` - fifty-first BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-vector.sh` - fifty-second BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-edn.sh` - fifty-third BEEBS benchmark
  runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ctl-string.sh` - fifty-fourth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-qrduino.sh` - fifty-fifth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-sglib-rbtree.sh` - fifty-sixth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-miniz.sh` - fifty-seventh BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-slre.sh` - fifty-eighth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-wikisort.sh` - fifty-ninth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-trio-sscanf.sh` - sixtieth BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-compress.sh` - sixty-first BEEBS
  benchmark runs end to end and validates its adapted LZW-state checksum marker
- `capstone/benchmarks/beebs/run-beebs-cubic.sh` - sixty-second BEEBS
  benchmark runs end to end with the soft-float/libm runtime and root oracle
- `capstone/benchmarks/beebs/run-beebs-sqrt.sh` - sixty-third BEEBS
  benchmark runs end to end and validates its correctness marker
- `capstone/benchmarks/beebs/run-beebs-ludcmp.sh` - sixty-fourth BEEBS
  benchmark runs end to end with the local const-array source workaround
- `capstone/benchmarks/beebs/run-beebs-minver.sh` - sixty-fifth BEEBS
  benchmark runs end to end and validates its adapted matrix checksum marker
- `capstone/benchmarks/beebs/run-beebs-frac.sh` - sixty-sixth BEEBS
  benchmark runs end to end with shared soft-float/libm support
- `capstone/benchmarks/beebs/run-beebs-st.sh` - sixty-seventh BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-nbody.sh` - sixty-eighth BEEBS
  benchmark runs end to end with correctly-rounded software `sqrt`
- `capstone/benchmarks/beebs/run-beebs-qsort.sh` - sixty-ninth BEEBS
  benchmark runs end to end with a widened 1-indexed array and sorted-region hash
- `capstone/benchmarks/beebs/run-beebs-qurt.sh` - seventieth BEEBS benchmark
  runs end to end and validates all three quadratic root cases
- `capstone/benchmarks/beebs/run-beebs-select.sh` - seventy-first BEEBS
  benchmark runs end to end with a widened 1-indexed array and return-value oracle
- `capstone/benchmarks/beebs/run-beebs-newlib-sqrt.sh` - seventy-second BEEBS
  benchmark; self-contained `__ieee754_sqrtf`, upstream exact verifier with
  `exp[]` moved to `static const` (Bug #9), soft-float builtins only
- `capstone/benchmarks/beebs/run-beebs-newlib-exp.sh` - seventy-third BEEBS
  benchmark; self-contained `__ieee754_expf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-log.sh` - seventy-fourth BEEBS
  benchmark; self-contained `__ieee754_logf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-newlib-mod.sh` - seventy-fifth BEEBS
  benchmark; self-contained `__ieee754_fmodf`, oracle tail vs host reference
- `capstone/benchmarks/beebs/run-beebs-stb_perlin.sh` - seventy-sixth BEEBS
  benchmark; 3-D Perlin noise, self-contained oracle (`benchmark()` compares a
  10x10 plane against a `static const` table and returns 0 on full match);
  only external dep is `floor`, added to the shared soft-float libm
- `capstone/benchmarks/beebs/run-beebs-matmult-float.sh` - seventy-seventh BEEBS
  benchmark; `matmult` source built `-DMATMULT_FLOAT` (float[10][10]), soft-float
  builtins only, FNV-1a checksum of the global `ResultArray` vs a host reference
  (`--gc-sections` drops the dead `values_match`/`frexpf`/`fabsf`)
- `capstone/benchmarks/beebs/run-beebs-whetstone.sh` - seventy-eighth BEEBS
  benchmark; classic Whetstone over the shared libm (added `atan`); built
  `-DPRINTOUT` with a capturing `POUT` that FNV-folds every module's outputs,
  compared exactly to a same-libm host reference

Most BEEBS correctness-marker wrappers now share `beebs_simple_domain.c` and
`beebs_simple_host.c`. Keep separate per-benchmark domain/host files only when
the marker ABI or host behavior is genuinely different; currently the older
`fac`, `fibcall`, and `insertsort` wrappers keep custom markers.

Most Capstone-specific benchmark source adaptations live in explicit `.c` files
under `capstone/benchmarks/beebs/adapted/`; shell scripts generally orchestrate
fetch/build/link/run rather than embedding C source. Full-replacement adapted
files (bubblesort, prime, cnt, duff, janne_complex, tarai, levenshtein,
recursion) are compiled directly. Prefix/tail files (crc32) and tail-append
files (strstr, insertsort, jfdctint, fdct, aha-compress, nettle-md5,
nettle-cast128, nettle-arcfour, nettle-des) are concatenated with the stripped
upstream source at build time. `huffbench` uses checked-in adapted C snippets
for its freestanding prefix and RNG replacement. `aha-mont64` uses a checked-in
rewrite helper for constant hoisting. `ndes` uses a checked-in rewrite helper
for pointer-based aggregate passing and explicit table delinearization.
`ctl-string`, `qrduino`, `miniz`, `slre`, and `trio-sscanf` are generated as
scratch sources under `$CAPSTONE_TMP_ROOT/beebs-build` because their adaptations
are local include/stub/allocation/verifier rewrites rather than reusable
replacement translation units.  `slre` additionally uses a checked-in tail file
(`adapted/beebs_slre_capstone_tail.c`) to avoid the `char *regexes[]` global
pointer array that would require caprelocs.  `wikisort` uses a checked-in tail
file to keep the upstream prefix while replacing the Range/sort/test tail.
`trio-sscanf` strips hosted includes, builds with `TRIO_SSCANF`,
`TRIO_EMBED_STRING`, float/file/dynamic-string features disabled, a minimal set
of embedded `triostr` helpers, and checked-in freestanding libc stubs.
`compress`, `cubic`, `minver`, `qsort`, `qurt`, and `select` use adapted
oracle tails because the upstream verifiers return `-1`. FP benchmarks use
compiler-rt soft-float builtins and, where needed, the shared
`adapted/beebs_softfloat_libm.c` domain libm.

`build-beebs-simple-capstone-common.sh` now supports `BEEBS_EXTRA_DEFINES`
(array of `-D` defines, e.g. `BEEBS_EXTRA_DEFINES=(QUICK_SORT)`),
`BEEBS_STRIP_FROM_REGEX` plus `BEEBS_ADAPTED_TAIL_SRC` for single-source
tail-replacement adaptations, and includes `-fno-jump-tables` unconditionally
(jump tables use raw integer addresses which fault on Capstone since loads
require capabilities).

## Resolved blocker

The 2026-06-09/10 split `null_blk` unload blocker is resolved. The hang was
diagnosed as lost timer progress after split-domain activity: QEMU traces showed
that the final timer H-interrupt was taken while `mie.MTIP` was disabled, after
which OpenSBI did not reprogram the timer and RCU/percpu-ref progress stopped.

The fix is in `capstone/capstone-qemu`:

- Capstone H-interrupt selection in `riscv_cpu_local_irq_pending()` now considers
  only interrupts enabled by `env->mie`.
- `rmw_mie64()` calls `riscv_cpu_check_interrupts()` after `mie` changes so a
  pending H-interrupt becomes deliverable when software reenables it.

The split null_blk package also keeps the safer fixes found during investigation:
metadata is borrowed per domain call instead of permanently shared, and
`null_validate_conf()` copies back only validated scalar configuration fields.

All temporary Linux/OpenSBI/QEMU trace and printk diagnostics were removed before
the verified run.

## Important distinction

The validated path is the **split host/domain runtime path**, not a full hosted
`capstone64-unknown-linux-gnu` Linux userspace. The helper is ordinary guest Linux;
the domain is a Capstone-loaded domain.

## Known backend bugs (stable workarounds in place)

The prologue frame-lowering bug is fixed and validated. Three remaining LLVM backend
workarounds from CoreMark bring-up stay in `capstone/benchmarks/coremark/build-coremark-capstone.sh`
and should only be removed after focused root fixes. Details: `plans/backend-compiler-fixes.md`.

The `va_list` capability-tag-loss backend bug is fixed and validated: `va_start`/
`va_arg`/`va_copy` now lower with capability ops (`stc`/`ldc`, 16-byte `cincoffset`
stride). The CoreMark `ee_printf_asm.S` trampoline is removed — `ee_printf` uses a
standard C `va_list` and CoreMark still validates. This unblocks the `va_list`
prerequisite for `trio`.

The `sub i128` pointer-decrement backend blocker is fixed and validated:
`ptr - integer` and `ptr + (-offset)` now lower through `cincoffset` with a
negated XLEN offset.

The `sub i128` pointer-difference backend blocker is also fixed and validated:
`ptr - ptr` now lowers by extracting both capability cursors with `lcc ..., 2`,
subtracting the XLEN cursor values, and sign-extending the integer result back
through the `i128` carrier when needed. `ctl-string` is the proof benchmark.

Stack-passed capability arguments are fixed: a function with >8 args whose extra
args are pointers had its stack-slot address computed with an integer `ISD::ADD`
(→ `addi`, tag-stripping), delivering the callee an untagged capability.
`CapstoneTargetLowering::LowerCall` now uses a capability `CIncOffset` for the
slot address (test `stack-cap-arg.ll`; repro `tests/runtime-qemu/stack-cap-arg-repro/`).
This unblocked RV8 `norx` and is the same class as the `va_list` fix.

The i128 non-vector-shift assertion (Bug #3) is fixed (`lowerScalarI128Shift`
general constant-shift fallback). **Capability globals are now auto-tagged**: the
`CapstoneCapGlobalInit` ModulePass synthesizes a per-module `__capstone_cap_init`
(called from `my_first_domain/start.S` before `domain_main`) that materializes
initialized capability globals in place at runtime — a tag cannot live in the
static image. Validated via `static-cap-typed-load-repro` + lit
`static-cap-global-init.ll`. Design:
`design/capability-globals-init-decision.md`.

## Capability granularity & provenance (C1/C2 — paper track)

After the three benchmark suites completed, work pivoted to the paper's security
contributions. **An external audit (2026-06-29,
`history/29-06-2026_15-08-22_granularity-provenance-audit.md`) reviewed this whole
direction; its findings are folded in below — read it before paper-facing work.**
Current state on `capstone-bootstrap`:

- **Bounds model** (`design/capability-bounds-model.md`): the narrowing op is
  **`SHRINK`** (`int_capstone_cap_shrink`); `SPLIT`/`SHRINKTO` exist in the ISA
  but are unwired. **Audit correction:** the `<4 KiB exact / grain-above`
  representability rule is **spec-derived, NOT measured** — this QEMU keeps exact
  fat bounds in a side table (`cm_map`) and restores them on load, so observable
  `SHRINK` is **exact at all sizes**. Un-narrowed bounds are segment-granular
  (single `PT_LOAD` ≈ whole image).

- **C1 object-granularity narrowing — INITIAL SLICES (not a spatial-safety
  theorem; broad `gp`/`sp` roots remain, permissions stay RWX):**
  - **Globals** — `selectLGA` (`CapstoneISelDAGToDAG.cpp`) narrows each sized data
    global to `[&g, &g+sizeof(g))`. Flag `-capstone-shrink-globals` (**default on**);
    functions / unsized externs not narrowed.
  - **Heap** — NOT a libc policy: only **two benchmark-local allocators**
    (`rv8_malloc.c`, dtoa `malloc_beebs`) `cap_shrink` returns; trio left
    un-narrowed (its `realloc` over-reads); CoreMark uses stack storage. Do not
    call this "heap default-on."
  - **Stack** — fixed stack objects narrowed to `[&obj, &obj+size)` via the
    shared `narrowToFrameObjectBounds` helper, now covering **both** the
    bare-`FrameIndex` address **and** interior pointers / load-store bases
    (`materializeFrameIndexAddrBase`), flag `-capstone-shrink-stack`
    (**still default off** pending the empirical default-on matrix). Not yet:
    varargs save-area, dynamic `alloca` (variable-size + spill slots excluded by
    design). Object- not subobject-granularity.
  - Validation is **functional only**: **CoreMark ✓, RV8 7/7 ✓, BEEBS 82/82 ✓**
    with global+heap on; stack-on smoke = CoreMark + 9 stack-heavy BEEBS ✓. Found
    a **real OOB bug**: rijndael wrote 8 bytes through a `char r[4]` (patched).
    **Code-size overhead measured across all 90 domains (CoreMark + 7 RV8 + 82
    BEEBS, 2026-07-01):** globals narrowing costs a near-constant **~15.6 bytes
    per narrowed global**; as % text, **median 1.83%, mean 4.17%, range 0%
    (no sized globals) – 46% (`statemate`, generated WCET tables)**; no
    correctness regression — matrix + full table in
    `design/c1-coverage-matrix-and-overhead.md`. **Runtime/cycle overhead still
    NOT measured** (functional QEMU, no cycle-accurate path) — don't claim it.
  - **Negative pointer difference fixed:** exact signed element scaling now
    restores `srai` after narrowing the i128 pointer-difference carrier to XLEN;
    genuine logical shifts remain `srli`. Positive and negative runtime probes
    pass, including `low - high == -7`.

- **Provenance/authority evidence suite** (`capstone/tests/capstone-authority/`,
  `run-authority-suite.sh`): 20 domains pinning runtime behavior (source + asm +
  QEMU trap/no-trap vs an oracle). forge/ptr→int→ptr **tag-fault**; global/heap/
  stack edge/index `_oob` **bounds-fault**; positive/negative pointer differences
  and last-valid-byte controls pass. A struct-field over-read is
  **no-trap-today**, confirming the subobject-bounds gap. The additive opt matrix
  passes all 12 eligible domains at `-O1/-O2/-O3`; 8 assembly-verified O0-only
  probes are explicitly skipped. Runtime fact:
  a domain-mode capability fault currently **aborts the QEMU model** (a
  `riscv_cpu_do_interrupt` assertion) after emitting the diagnostic.

- **Regression tests:** lit `cap-shrink-globals.ll`, `cap-shrink-stack.ll`
  (on/off A/B), `ptr-diff-signed.ll`, and updated
  `static-cap-global-init.ll`. Full Capstone lit suite green (32 tests).

- **C2 (provenance verifier) — REDESIGNED (v2, 2026-07-01), awaiting reviewer
  sign-off before implementing.** The audit found v1 (`UNKNOWN`-accepting,
  opcode-only) was a hygiene checker, not a proof. The redesign in
  `design/c2-provenance-verifier-proposal.md` §"Design (v2)" folds in all three
  fixes: no permissive `UNKNOWN` (`ROOT`/`CAP`/`INT`/`TAINTED` lattice, TAINTED-as-
  authority flagged), IR→MIR intent + calling-convention arg/return seeding,
  precise per-opcode transfer functions (LDC propagates memory tag; tied-operand
  ops inherit+validate; integer-as-base is a fault not a forge), two separated
  properties (P1 non-forging / P2 preservation), and a small hand-proved formal
  model with the corpus as validation. v1 retained in the doc for history. Do NOT
  implement until the reviewer signs off on v2.

- **Audit's strategic reframing (for the reviewer):** object bounds re-derive
  CHERI; Capstone's novelty is linearity/revocation/`SPLIT`/**root-elimination**.
  Proposed stronger frame: **provenance + attenuation + root-elimination** (trusted
  `SPLIT` removes the ambient broad root from application code). A
  research-direction decision, not yet acted on.

## Where to go next

- Next milestone: `state/current-next-step.md`
- Test entry points: `ref/testing-matrix.md`
- Deep design docs: `design/`
