# SQLite on silicon — scoping plan (the paper's comprehensive-benchmark number)

**Status:** PROPOSED — for review before building. Created 2026-07-23.

## Mandate (settled 2026-07-23)

- **PI:** the comprehensive benchmark must **run on the FPGA hardware**, full stop.
  A model-based silicon estimate (QEMU counts × measured primitive latencies) is
  **not** acceptable for the claim. → We do a real end-to-end SQLite run on the
  board and read cycle-accurate `mcycle`.
- **Board owner (transfer):** the domain ships **inside the buildroot rootfs** and
  the whole firmware image is loaded over **JTAG** (`monitor load_image`, the same
  path the 15 MB firmware already uses). **No per-run UART transfer.** This is the
  expected, standard route — the earlier UART-speed worry is retired.
- **Board owner (limits):** create_domain image/region bounds and the revoke-node
  pool size live in the **RTL / capstone module** and are **adjustable**. Changing
  them = re-synthesize the bitstream (heavier than a firmware reflash), so we fit
  within current limits where possible and read the actual constants from RTL
  before assuming a ceiling.
- **PI (deliverables):** the paper needs **performance results AND correctness** on
  hardware. Performance = the benchmark suite (CoreMark, BEEBS, RV8) on silicon with
  cycle counts; correctness = those programs producing correct results under
  capability protection on real hardware. So the benchmark ladder below is **not just
  de-risking toward SQLite — it is itself the perf+correctness deliverable.** SQLite
  adds the security-relevant host↔domain boundary-marshalling workload on top.

## Delivery mechanism — JTAG `load_image` per domain (no per-benchmark reflash)

Board owner's model is "domain in the image, loaded over JTAG." For a *suite* of
domains we refine that to **JTAG `load_image <ADDR> dom.bin` into a reserved RAM
region + a controller that reads the domain from there** (the "tier-2b" variant),
rather than baking each domain into the rootfs and rebuilding firmware per benchmark.
Rationale: it lets us run **many benchmark domains in a single board session** (load →
run → read `mcycle` → load next), which is exactly what a perf *suite* needs — a
firmware reflash per benchmark would make the suite impractical. Baking into the
image (tier-2a) is reserved for a final frozen artifact if we want one. Confirming the
reserved region with the board owner is a one-line ask (`reply-boardowner-jtag-limits.md`).

## What is already in hand (do not rebuild)

- SQLite 3.53.3 **runs end-to-end in a pure-cap domain on QEMU** — CREATE/INSERT/
  SELECT correct, all 8 bring-up gaps closed (`state/current-state.md` "SQLite
  in-memory bring-up"; `benchmarks/sqlite/`).
- **HostCall** delegates syscalls to the host → **no full libc** needed (same model
  used for every QEMU domain run).
- **umm_malloc** heap is vendored and PureCap-safe.
- **Silicon ABI validated:** a compiler-built, globals-using domain creates, runs,
  and returns on the captype-fixed CVA6 via the gp-captable / gp-free path
  (retval 554745961 — `history/22-07-2026_18-05-00_*` UPDATE 23-07).
- **shrink→store RTL hazard** is root-caused with a **build-time workaround**:
  `-capstone-shrink-stack=false -capstone-shrink-globals=false` → correct on
  silicon (coarser whole-array/frame bounds, still capability-confined).
  `history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`.
- **Firmware build recipe** (fw_payload with our gp-delivery monitor, embedded
  FDT+kernel) — memory `project_fpga_fw_payload_build_recipe`.
- **Per-primitive silicon cycle numbers** already measured (borrow(N)≈75+3N/2,
  load2/shrink1/mrev50/delin+revoke121) — memory `project_fpga_silicon_measurement_status`.
- **Boundary frequency on QEMU:** ~1 borrow / ~19k instr on speedtest1
  (memory `project_sqlite_boundary_overhead_tier1_tier2`).

## The gap to close

Everything above is either QEMU-only or a *hand-written* silicon probe. What's new:
compile the **real SQLite engine** (hundreds of globals, a full call graph) for the
**silicon ABI** and run it in a domain on the board, then measure it. Three real
integration risks, called out per stage.

## Strategy — climb a benchmark ladder to SQLite (do NOT jump straight to SQLite)

SQLite is the hardest possible first target: hundreds of globals, heap, and boundary
marshalling all at once — any silicon-only miscompute would land in the least
debuggable program. Instead validate the **silicon build config**
(`-capstone-gp-captable` + gp-free call/ret, **shrink OFF**) on a ladder of programs
we already have exact oracles for, cheapest first, fixing any board-only issue before
climbing. Payoff: (1) each risk is isolated on a small program; (2) the paper gets a
silicon perf *curve across a suite*, not a single SQLite point — a stronger
"runs on real hardware" story and closer to PI's "comprehensive benchmark".

**The ladder (increasing complexity):**
1. **matmult-int** — tiny; an array-store-in-loop kernel = the exact pattern that hit
   the shrink→store hazard. First compiler-built (not hand-written) program to
   validate the **shrink-off workaround** on silicon. Replaces the hand-written
   `rc_const0` probe with real codegen.
2. **CoreMark** — real call graph + globals; the "we ran a real app on silicon under
   capability protection" existence proof.
3. **BEEBS subset (~4–5)** — pattern-diverse, chosen to stress the codegen most likely
   to break on silicon: `crc32`/`crc` (global lookup tables), `matmult` (array stores),
   `nettle-aes` or `rijndael` (byte arrays; rijndael historically caught a real OOB),
   `sglib-hashtable` or `sglib-rbtree` (pointer structures + heap). Not all 82 — a
   representative slice (board time is human-in-the-loop and serialized).
4. **RV8 subset** — heavier, real `malloc`/heap (umm_malloc): e.g. `qsort`, `sha512`,
   `aes`, `miniz`. Exercises the allocator on silicon.
5. **SQLite** — the headline. Minimal first (existence proof), then the boundary
   workload (the paper's actual measurement).

Which rungs make the *paper* is a call we make after seeing the numbers; the ladder's
first job is de-risking. All rungs = single-domain compute, **no boundary
borrow/revoke** → the rev-node pool (Risk B) is a SQLite-boundary concern only, not a
ladder concern.

## Stages (each gated; stop and report at the gate)

### Stage 0a — Cap-table builder-glue GENERATOR (the missing "compiler-side" piece)
The cap-table entry glue (`start-gpfree-captable-app.S`) is currently **hand-baked**
per app (`N=1, size[0]=32`). Silicon forbids a runtime descriptor read from the code
image (execute-cap can't read data — the wedge we already hit), so sizes must be
**baked as immediates**. Build the generator its own comment promises: read the
compiler's `.capstone_gp_table` descriptor (per-global index + size) from a compiled
domain and emit that domain's `BUILD_GP_CAPTABLE` block (per-global `split`/`delin`/
zero/`stc`, sizes baked). This is the reusable machinery every rung (matmult →
CoreMark → SQLite) depends on.
- CONFIRMED (2026-07-23): the compiler already emits `.capstone_gp_table`
  (`CapstoneAsmPrinter.cpp:849`). Layout, in `getGpCaptableIndex` order:
  `u64 count`, then per global `{ u64 size, u64 align, i64 init_off }` where
  `init_off` is a PC-relative `sym - .` diff to the initializer template, or **0 for
  a zero-init (.bss) global**. Generator reads this from the built `.dom` and emits
  the `BUILD_GP_CAPTABLE` block.
- **INITIALIZED GLOBALS — RESOLVED (design, 2026-07-24), no board dependency.**
  Earlier framed as a board question ("copy the image template" → a data-read from
  the code image = the wedge). The `capstone-c` reference settles the model: its
  prologue (`codegen.rs:264-301`) carves + `stc`s each global exactly like our glue,
  emits **no `.data` section**, and never reads an initializer from the image (the
  hardware forbids it) — comments `TODO/FIXME` confirm globals are carved storage,
  and no sample uses an initialized global. So the silicon model is **carve + zero +
  initialize-in-code, never an image data-read.** Our resolution: the generator reads
  the initializer bytes from the built object **at build time** (host-side, fine) and
  emits **`li reg,<word>; sd reg,off(storage)` per word** (`sd x0` for zero words).
  The constants ride as **instruction immediates** (fetched via PCC/execute — always
  allowed), so there is **no runtime data-read from the image**. Silicon-correct, no
  monitor change, no board round-trip. Caveat: large constant tables (SQLite) become
  many `li/sd` → code-size bloat; optimize later (RLE / or a monitor-readable image
  cap) only if it bites. This unifies `.bss` and `.data` in one code path.
- **RISK A also:** cap-table size vs. the stack/data region it's carved from at
  CoreMark/SQLite global counts.
- Gate: matmult-int builds with an auto-generated glue and runs on QEMU.
- **DONE 2026-07-24:** generator + generic glue under `tests/runtime-qemu/silicon-ladder/`
  (`gen-gp-captable-glue.py`, `start-gp-captable-generic.S`, `build-ladder-domain.sh`,
  `run-ladder-qemu.sh`). Both global classes handled + validated on QEMU (gp
  fabrication OFF):
  - `.bss`: rung 1 **matmult-int** (3 globals, non-inlined `mm_cell`, `+m` mul) →
    `retval == oracle 774662735`, gate `cjalr=0 ldc-gp=6`.
  - initialized: **init_probe** (const `LUT[8]` + `.bss`) → `retval == oracle
    4093668916`; `li/sd` materialization packs the LUT int32s correctly.
  Generator scaled N=1→N=3 automatically. Commits `05451cd8`, `2ffd621a`.
- **Rung 2 DONE 2026-07-24: BEEBS `insertsort`** (a *found* benchmark; single-TU;
  `is_a[11]` .bss global + a function-local const `expected[11]`) → `retval ==
  oracle 271779359` on QEMU, silicon config. Files `beebs_insertsort_{kernel.h,
  app.c,host.c}`. Surfaced + fixed TWO real bugs (full trail:
  `history/24-07-2026_03-57-54_ladder-rung2-insertsort-memcpy-stc-miscompile.md`):
  (1) **generator** — pick the initialized-global template by *reloc type*
  (`R_RISCV_ADD` 33..36), not "skip `.L`-prefixed": a function-local const is
  promoted to `.L__const.*` and was wrongly zero-filled. (2) **compiler
  (unconditional)** — `findOptimalMemOpLowering` sub-capability memcpy fix: a copy
  that is neither 16-aligned+16-multiple nor 8-aligned+8-multiple (e.g. 44-byte
  `int e[11]`, 4-aligned) fell to the generic lowering, which used a 16-byte `stc`
  while the under-aligned source loaded only 8 bytes → upper 8 bytes of each unit
  dropped. Generalized to scalarize any non-tag-capable copy. **Full regression
  gate GREEN** (lit 41/41; QEMU: CoreMark, BEEBS 82/82, authority 26/26, RV8 7/7,
  SQLite 9/9 rows). Also noticed, NOT fixed: `-O2` on this kernel crashes clang
  (APInt assertion, store→load forwarding).
- **Rung 3 DONE 2026-07-24: BEEBS `crc32`** → `retval == oracle 1703161001` on
  QEMU, silicon config. Two globals (a 1 KiB table + function-local `static
  seed`), nested loops, real `crc32pseudo→rand_beebs→UPDC32` call graph indexing
  the table 1024x. Files `beebs_crc32_{kernel.h,app.c,host.c}`. Validates a 1 KiB
  `.bss` array reached via `ldc gp[i]`. (Value differs from upstream 1207487004
  only because `unsigned long` is 64-bit here; domain == native oracle by
  construction, which is the correctness claim.)
- **Rung 4 DONE 2026-07-24: BEEBS `recursion`** → `retval == oracle 1579141629`.
  Deep self-recursion (fib) + mutual recursion (anka<->kalle) under gp-free plain
  call/ret (tall reentrant stack), a `volatile int In` global + `static n`.
  `cjalr=0`. Files `beebs_recursion_{kernel.h,app.c,host.c}`.
- **Rung 5 DONE 2026-07-24: BEEBS `prime`** → `retval == oracle 582955588`.
  Primality call graph (prime->even->divides) + `swap(&x,&y)` taking the ADDRESS
  of two globals (pointers into the gp cap-table region) + a volatile `result`.
  3 .bss globals, `cjalr=0`. Files `beebs_prime_{kernel.h,app.c,host.c}`.
- **Rung 6 DONE 2026-07-24: RV8 `primes`** (first RV8-family rung) → `retval ==
  oracle 99991`. Faithful to rv8-bench's sieve at the committed RV8 oracle's
  reduced limit (100000 → largest prime 99991) with the same `1ull << (p&0x3f)`
  shift-UB fix; single-TU, sieve as a **12.5 KB `.bss` bitmap** (runtime-written,
  no big *initialized* table), 64-bit shift arithmetic, nested sieve loop,
  `cjalr=0`. Files `rv8_primes_{kernel.h,app.c,host.c}`. **This rung forced a
  real generator fix** — see next item.
- **DONE 2026-07-24: generator now zeroes large `.bss` with a runtime loop.**
  `gen-gp-captable-glue.py` previously unrolled `sd x0, k(t2)` per 8-byte word,
  which for primes' 12.5 KB sieve (a) overflowed the 12-bit store/`addi` immediate
  and (b) would balloon `.text` by ~1560 stores past the 0x1000 PCC code window.
  Fix: an all-zero global is now zeroed by a compact `li count; cincoffset ptr;
  loop{sd x0; cincoffsetimm ptr,8; addi count,-8; bnez}` (any size, tiny code);
  the reserve `addi t1,-stor` falls back to `li;sub` when >2047. Initialized
  (non-zero) storage stays unrolled and is now explicitly capped at 2040 B with
  the large-RO message. Re-verified no regression: matmult / crc32 (1 KiB `.bss`)
  / insertsort (initialized const path) / primes all still PASS. **This directly
  de-risks SQLite's large `.bss`** (only large initialized `.rodata` remains open).
- **`dijkstra` DEFERRED — blocked by the big-table item.** Its `AdjMatrix[10][10]`
  (400 B) and the function-local `expected[100]` (400 B) are genuine INPUT data
  (a random adjacency matrix + a result vector), so unlike crc32 they cannot be
  computed at runtime -- ~200 li/sd immediates plus a 10-function body would
  overflow the 0x1000 code window. Confirms the big-table limit bites real
  benchmarks with legitimate input data; revisit once that mechanism exists.
- **OPEN — large initialized read-only tables don't fit the silicon-gp model.**
  crc32's upstream 256-entry `const` table exposed it: the generator materializes
  initialized globals as `li/sd` instruction immediates, so a 1 KiB const table
  balloons `.text` to ~2 KiB, which (a) collides with the fixed globals offset in
  `gp-free-domain/link-gpfree.ld` and (b) overflows the monitor's PCC code window
  (all code must fit `[base, base+0x1000)` for the silicon image SPLIT). Runtime
  table-gen sidesteps it for crc32, but **SQLite's static tables will hit this**:
  needs a real large-RO-table delivery mechanism (candidates: a monitor-provided
  data cap over an image .rodata region; a data-region copy set up by the glue;
  or raising GPFREE_GLOBALS_OFFSET + PCC window — the last needs a firmware
  rebuild). Decide before Stage 3 (SQLite firmware). [[project_gp_captable_codegen]]
- **RISK A concretized — per-module cap-table indices.** `getGpCaptableIndex`
  (`CapstoneISelDAGToDAG.cpp:112`) numbers globals **per module**, so multi-TU
  domains collide on the single gp cap-table + emit multiple descriptor headers.
  Single-TU domains are fine (matmult, init_probe, **the SQLite amalgamation**, most
  BEEBS/RV8). **CoreMark (multi-file) needs amalgamation or `-flto`** to present one
  module — the next step. (Whole-program-index compiler fix is the alternative.)

### Stage 0b — QEMU parity for the whole ladder in the *silicon* build config
With the generator, build every ladder rung `-capstone-gp-captable` + gp-free +
**shrink off** (`-capstone-shrink-stack=false -capstone-shrink-globals=false`) and
prove each still passes on QEMU (fast loop, zero board time). Capture each rung's QEMU
**instruction count** as the cross-check reference for its later silicon `mcycle`.
- The ladder shows *where* on the global-count curve any cap-table problem first bites.
- Gate: every rung emits its correct oracle marker on QEMU under this config.

### Stage 1 — Silicon ladder run (Opus main session; human-in-the-loop)
Run the ladder on the board cheapest→hardest, **stopping to debug any board-only
miscompute before climbing** (that's the whole point of the ordering):
- matmult-int → CoreMark first: existence proofs + shrink-off validation on real
  compiler-built code.
- Then the BEEBS + RV8 subsets.
- Read `mcycle` around each; cross-check against the Stage-0 QEMU instruction count.
- **RISK C (a board-only miscompute):** handle exactly like the shrink case — dump
  caps with `lcc`, compare to QEMU, localize — never guess. Isolating it on a small
  rung (not SQLite) is why we climb.
- **RISK (image size):** RV8/SQLite images are larger — read the create_domain image
  limit from RTL early (board owner said it's in the RTL/capstone module); if a rung
  overflows, that's a concrete bitstream-bump ask.
- Gate: the benchmark ladder runs correctly on silicon with cycle numbers; the
  silicon build config is proven on real codegen before SQLite.

### Stage 2 — Resource-limit recon for the boundary workload
(Only needed before the SQLite *boundary* run — the ladder doesn't touch rev-nodes.)
- Read the RTL for the **revoke-node pool size** and create_domain region max.
- Count the boundary workload's **borrow/revoke events**. Pool is a fixed bump
  allocator with **no slot reclamation** (drop only invalidates) — memory
  `project_fpga_silicon_measurement_status`.
- **RISK B (rev-node exhaustion):** size the first boundary workload under the pool;
  if a representative run needs more, spec an RTL pool bump (bitstream re-synth).
- Gate: a written boundary-workload budget that fits — or a concrete RTL-bump ask.

### Stage 3 — SQLite firmware packaging (bake into rootfs, JTAG)
- Add the SQLite domain binary (+ the controller that create_domain's it) to the
  buildroot rootfs overlay; rebuild **fw_payload** via the known recipe (embedded
  FDT+kernel — silent boot fail otherwise); JTAG `monitor load_image`.
- Board owner's model exactly: image built in, loaded over JTAG, no UART.
- Gate: board boots to a shell, `/dev/capstone` present, controller sees the SQLite
  dom in the rootfs (no transfer step).

### Stage 4 — SQLite silicon run + measurement (Opus main session)
- **Minimal first** (your call): CREATE/INSERT/SELECT (`run-sqlite-memory.sh` shape) in
  the domain on the board; confirm correct rows — real software on silicon under
  capability protection.
- **Then the boundary workload:** the host↔domain marshalling path
  (`sqlite_boundary_cost_domain.c`, the paper's subject). Read `mcycle` around the run
  and around the borrow/revoke primitives.
- Gate: correct SQLite result on board + cycle numbers, cross-checked against QEMU.

### Stage 5 — Paper integration
- Silicon `mcycle` numbers (ladder + SQLite boundary) into the perf table as the
  "on real hardware" layer over the existing QEMU CHERI-vs-Capstone comparison
  (`evaluation.tex` §`sec:eval-perf-compare`). No re-run of the CHERI/HFI side.

## Config note for the paper (be upfront internally)
The silicon perf run is built **shrink-off** (whole-array bounds), while the QEMU
security results used shrink-on (per-element). This does **not** weaken the boundary
claim: shrink is *intra-domain subobject* narrowing, orthogonal to the *host↔domain
boundary* borrow/revoke that this paper measures — bounds are still enforced, at
whole-array granularity. State the config; don't conflate the two axes.

## What needs the board owner (vs. what we do)
- **Us:** Stages 0–2 (compiler config, RTL constant read, firmware rebuild), and
  driving Stages 3 board sessions.
- **Board owner:** only if Stage 1 shows the workload overflows the RTL pool /
  image limit → a bitstream re-synth with bumped constants. Otherwise nothing.

## Constraints (repo hygiene — unchanged)
Board sessions stay in the main Opus session (etiquette + token). No real-person
names anywhere committed/shared. Submodule *source* stays uncommitted (firmware/
buildroot edits are local experiments). shrink-off + gp-captable are existing gated
flags — no new default flips. Bug-fix/root-cause notes → `history/` dated.
