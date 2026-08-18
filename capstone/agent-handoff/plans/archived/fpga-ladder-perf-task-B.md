# B-lane task: FPGA cycle-count perf numbers for the silicon-ladder rungs

**Autonomy: high.** Goal + guardrails, not steps — you know the FPGA path already (agentB-016/
017). This doc gives you (a) the **full cold-start context** — what the A lane changed in the
compiler/monitor and where the blockers are, so you can take over this workstream — (b) the
deliverable, and (c) the FPGA mechanics (how to run, how to run *faster*, and the tier-2b suite
path).

**First, adopt this repo's operating rules as your own.** Read the repo-root `CLAUDE.md` and
`capstone/agent-handoff/history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md` (hard constraints + the permanent rules restated at the
bottom of this doc), then `source capstone/tests/capstone-test-env.sh` and read
`state/current-state.md`. **Note the board rule** (permanent): the FPGA is open to either lane,
but **serialized** — never two board sessions at once.

---

## 0. Cold-start context — what's changed and where the blockers are

You are picking up a workstream the A lane has been driving. Absorb this before running anything;
it is the "why" behind the ready set and the guardrails.

### What we changed (compiler)
The paper needs benchmarks that run in a pure-capability **domain on the CVA6 silicon**, where
the ABI differs from our QEMU model in two ways the compiler now handles under a **gate**
(`-capstone-gp-free`, default off → byte-identical corpus by construction; lit 40/40):

- **gp-free global addressing.** Stock codegen emits `cincoffset $rd, gp, …; delin` for globals,
  needing `gp` = an image-covering cap at *cursor 0*, which is **unrepresentable** under cap
  compression on real silicon. Under the flag we instead emit an absolute in-bounds `SCC`
  (`-capstone-gp-captable` is the silicon-correct variant: globals via a cap-table `ldc gp[i]`).
  Files: `CapstoneExpandPseudoInsts.cpp` (`expandCapGlobalBase`), `CapstoneISelDAGToDAG.cpp`.
- **cjalr-free calls/returns.** Stock codegen lowers `PseudoCALL/RET/TAIL` to `CJALR`; a cap
  return needs `ra` = a code-cap that can't be formed gp-free on a `cscall` entry. Under the flag
  calls/returns are plain `jal`/`jalr` within PCC. File: `CapstoneAsmPrinter.cpp` (`selectCall`).
- **shrink OFF for silicon.** `-capstone-shrink-stack=false -capstone-shrink-globals=false`:
  there is a **shrink→store RTL hazard** on this CVA6 (root-caused; whole-array/frame bounds are
  the build-time workaround, still capability-confined). Rung binaries are built `-O0` + native
  `+m`. See `history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`.

**Net: the silicon config is `-capstone-gp-captable` + gp-free call/ret + shrink-off + `+m`, `-O0`.**
The ladder harness already bakes this in.

### What we changed (monitor / delivery)
`create_domain` (OpenSBI `sbi_capstone.c`) mints the domain's `gp` as a real **data** cap and
stashes it in the cscratch region top slot; the domain's entry glue loads it (no QEMU-style
fabrication, no `ctvec`). This is the board owner's confirmed channel and is **already validated
on silicon** — a compiler-built, globals-using domain creates, runs, and returns on the
captype-fixed CVA6 (retval `554745961`). See `history/22-07-2026_18-05-00_*` UPDATE 23-07 and
memory `project_silicon_gp_delivery_boardowner_guidance`.

### The blockers (know these so you don't trip them)
1. **Monitor cannot be regenerated to a booting image** (the big one). Rebuilding `fw_jump.elf`
   (QEMU) or the FPGA firmware monitor from the *current* `capstone-c` produces a **boot-hang**
   (zero serial). The working firmware is an **unreproducible prebuilt** (built by a compiler
   state that allocates smaller frames, `s0–s6/−368`, vs current `s0–s11/−464`). **So: use the
   existing working firmware prebuilt as-is; do NOT rebuild the monitor.** This is a *separate*
   task (`plans/monitor-regen-audit-task-B.md`) — fixing it unblocks large-`.rodata`/SQLite on
   silicon, but the 7 rungs below need **no** monitor change. Memory:
   `project_opensbi_monitor_rebuild_include_wrapper` (WARNING section).
2. **Large `.rodata`/const tables** can't be delivered to a domain yet (needs the monitor change
   in #1). This blocks SQLite + const-table benchmarks on silicon — out of scope here.
3. **Board is flaky/slow**: websocket drops, ~2 min to JTAG-load 15 MB firmware, no HW
   breakpoints. Use UART-banner stall-probes, not `hbreak`.

---

## 1. Goal (the deliverable — this is the deadline-critical one)
Produce **real FPGA cycle counts** for the silicon-ladder micro-benchmarks — the paper's headline
"on real hardware" perf numbers. For each ready rung: run it in a pure-cap domain on the
**Genesys2 CVA6 Capstone** board, confirm **correctness** (domain retval == the native `cc -O0`
oracle), and record its **cycle count** (`mcycle`). Output = a perf table `rung → cycles (+ ✓)`
for the eval section, plus a fixed per-call overhead noted separately.

## 2. The ready set (no monitor changes needed)
These 7 rungs already run in the silicon config on QEMU and need **no** large-`.rodata`/monitor
work: `matmult_int`, `coremark_matrix`, `rv8_primes`, `beebs_crc32`, `beebs_insertsort`,
`beebs_prime`, `beebs_recursion` (in `tests/runtime-qemu/silicon-ladder/`). Start with
`matmult_int` (simplest; already board-validated as a globals-using domain) to prove the
end-to-end pipeline, then batch the rest **in one board session** (see §4).

## 3. The bridge you'll build (QEMU ladder → board path)
The ladder harness (`run-ladder-qemu.sh`) targets **QEMU**; the board path lives in
`tests/rtl-smoke/` + `fpga_driver/`. Bridge one ladder rung onto the board path:
- Build the rung's `.dom` in the silicon config — reuse `build-ladder-domain.sh` (honors
  `DOMAIN_OPT_LEVEL`, default `-O0`).
- Wrap the workload in `mcycle` reads — `tests/rtl-smoke/fpga_instrument.h`. Measure the **same
  enter/exit points** across rungs so numbers are comparable.
- A board **controller** (`.user`, freestanding, integer-only, raw Linux syscalls) that does DPI
  create+call and prints retval + cycles — adapt `tests/rtl-smoke/borrow_cost_fpga_nogp_ctl.c`.
- Transfer + run via the driver (§4), harvest retval + mcycle.
This is bounded plumbing between two paths you already have.

## 4. How to run on the FPGA (the mechanics)

**Read first** (the map + the runbook you'll adapt):
- `ref/HOW-TO-LAUNCH-ON-FPGA.md` — the map + non-negotiables (lock → power-cycle → run →
  **power off + unlock in `finally`**).
- `ref/gp-free-silicon-smoke-runbook.md` — the closest working precedent (build domain +
  controller, boot, transfer, run, harvest). Your rungs are the same shape (globals + call graph,
  integer-only) minus the monitor rebuild — you **skip its Step 1** (use the prebuilt firmware).
- `history/22-07-2026_18-05-00_*` — full board findings + etiquette.

**Board access.** Browser/websocket CVA6+Capstone FPGA — **no SSH**. Drive it via the Python
driver (`tests/rtl-smoke/fpga_driver/`, venv `/tmp/capstone/fpga-venv`). URL+token in
`~/.config/capstone/fpga-board-url` (secret — never commit/echo; comes from the user out-of-band).
A local `.bit` is not needed — the resident bitstream is server-side
`working-caplifive-captype-fixed.bit`; verify it's resident before measuring.

**The flow per session:** lock the board → power-cycle → JTAG-load the firmware prebuilt (~2 min,
15 MB) → gdb-boot to Linux (image exposes `/dev/capstone`) → for each rung: transfer `<ctl>` +
`<rung.dom>`, run `<ctl> <rung.dom>`, read back retval + mcycle → **power off + unlock**.

**Signal of a live domain** = the controller prints its first line (that's *after*
`IOCTL_DOM_CREATE` returns). `C_PRINT` (`csrw 0x800`) goes to the **RTL trace, not UART** — don't
use it as a UART probe.

**Correctness gate** stays: `retval == native oracle`, and the static gate `cjalr=0`
(already true for these rungs). A cycle number without the correctness check is not a result.

## 5. Run it faster — fast_xfer + one-session batching

The old debug bottleneck was UART transfer (char-by-char / 3 round-trips per chunk). Two levers:

- **Tier-1 `fast_xfer` (DONE, board-validated ~3×).** Use `fpga_driver/fast_xfer.py`
  `fast_put` — direct-append base64 chunks, single final-sha guard, safe-retry on mismatch. This
  is the default transfer for every domain; a controller is now ~30 s vs ~4 min. Memory:
  `project_board_transfer_tiers`.
- **Batch all 7 rungs in ONE board session.** The firmware JTAG-load (~2 min) dominates; pay it
  **once**. Boot once, then loop `fast_put rung.dom → run → read mcycle → next rung`. The domain
  binaries are tiny; only the firmware is big. This alone makes the 7-rung suite a single ~15-min
  session, no per-rung reflash.

## 6. tier-2b — the suite/SQLite scaling path (yes, we'll use it — but not required here)

**tier-2b** = JTAG `load_image <ADDR> dom.bin` into a **reserved RAM region** + a resident
controller that reads the domain from there (vs baking each domain into the rootfs = tier-2a, or
per-run UART transfer). It's the delivery mechanism for a **large suite + SQLite** — load → run →
read `mcycle` → load next, many domains per session, no reflush, no UART bottleneck. This is the
board owner's endorsed model ("domain in the image, loaded over JTAG"). See
`plans/sqlite-on-silicon-scoping.md` §"Delivery mechanism".

**For THIS 7-rung task, tier-2b is NOT required** — UART `fast_xfer` in one session (§5) already
runs the whole batch comfortably. Use tier-2b only if you find the UART loop too slow at this
scale (you won't for 7 tiny integer domains). **Before using tier-2b** you must confirm the
reserved-region address/size with the board owner (one-line ask, tracked in
`plans/` / `reply-boardowner-jtag-limits.md`) — do not guess a RAM address. Treat tier-2b as the
on-ramp to the *next* task (SQLite on silicon), and note in your results whether the UART loop was
a bottleneck (that's the signal for whether to build tier-2b next).

## 7. Hard guardrails
- **DO NOT rebuild the monitor / firmware.** Confirmed toolchain gap (§0 blocker #1) — every regen
  boot-hangs. Use the existing working firmware prebuilt as-is. If a rung seems to need a monitor
  change, it's out of scope (that's the blocked large-`.rodata` track → `monitor-regen-audit-task-B.md`)
  — pick another rung. All 7 need no monitor change.
- **Board serialized across lanes.** Either lane may run it, but **never two at once** —
  coordinate timing with the A lane before you start, hand back when done. Lock → power-cycle →
  run → **power off + unlock in `finally`** (never leave it locked/on). The secret token comes
  from the user out-of-band — never commit or paste it anywhere.
- **Correctness gate:** `retval == native oracle` + `cjalr=0`. A cycle number without the
  correctness check is not a result.
- Measure `mcycle` around the domain workload **consistently** across rungs; note fixed per-call
  overhead separately.

## 8. Deliverable format
A committed results note (`history/DD-MM-YYYY_HH-MM-SS_fpga-ladder-perf.md`) + a compact table
(rung, cycles, correctness ✓) suitable for the paper's perf section, and a one-line note on
whether the UART transfer loop was a bottleneck at 7 rungs (tier-2b signal). Manager/PI-facing
summary → `/tmp/capstone/`, not the repo.

## 9. Permanent repository rules — adopt as your own (non-negotiable)
1. **Never mention any real person by name — anywhere** (PI/board owner/collaborator → neutral
   roles), in every committed/shared file, commit, doc, report. (Upstream `lldb/`,`llvm/` files
   are not ours.)
2. **No `Co-Authored-By:`**; no worker/agent identity in commits; **don't rewrite pushed
   history**; **commit only when the user asks.**
3. **Never commit debug/report/session-note files.**
4. **Manager/PI-facing summaries → `/tmp/capstone/`**, never the repo.
5. **Serialize the QEMU suites** (shared `rootfs.ext2` lock) — and **serialize board sessions
   across lanes** too.
6. **`ninja -j90`**, never `-j112`.
7. **No commits into submodule source.**
8. **Bug-fix/root-cause/audit notes → `history/`** (dated), not `design/`. Active plans → `plans/`.
9. Commit to **`capstone-bootstrap-b`** only. The board is **not** off-limits to you, but is
   serialized across lanes (coordinate first).

## 10. Start here
`git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap` to pick up the ladder
rungs, the board-rule update, and this task. Read §0, then coordinate a board window with the A
lane before your first run. Start with `matmult_int` to prove the pipeline, then batch the rest.
