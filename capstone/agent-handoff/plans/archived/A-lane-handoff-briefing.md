# A-lane handoff briefing — the in-flight state a successor needs

**Purpose.** One doc a successor (peer lane B, or a fresh A session) reads to take over the A
lane's in-flight work. It is an **index + the load-bearing "why"**, not a re-derivation — the
canonical detail lives in the files it points to. Read it after `CLAUDE.md`,
`state/current-state.md`, and `state/current-next-step.md`.

Last refreshed: 2026-07-24.

---

## 1. The project in three lines
Cycle-accurate RTL/FPGA + QEMU evaluation of Capstone memory-safety (PureCap RISC-V, 128-bit caps)
on CVA6/Genesys2, for an NDSS/USENIX paper (double-blind, **deadline end-July-2026**). The PI
mandates the headline benchmarks **run on FPGA hardware** (a QEMU-count × primitive-latency model
is not acceptable for the claim). Security story is settled; the **separating axis is performance**
(eager CHERI matches our temporal safety, so we win on cost).

## 2. Lanes, branches, clones
- **A lane** → branch `capstone-bootstrap`, clone `/home/alexey/dev/llvm-capstone` (this one).
  Owns: compiler/codegen + capability-ABI, the paper, commits, synthesis, anything with
  real-person names. Single-writer of `state/current-state.md` / `current-next-step.md`.
- **B lane** → branch `capstone-bootstrap-b`, clone `/home/alexey/dev/llvm-capstone-b`, remote
  `bwork`/`origin`. Peer Opus session. Has its own `*.B.md` state files. Already owns the **FPGA
  web-console driver** (task 017: HTTP+Socket.IO hybrid, validated on live hardware).
- **`capstone-gp-free`** → the branch where the silicon-shaped compiler work lives (gp-free /
  cjalr-free codegen; committed `88054a14`). Not merged to `capstone-bootstrap`.
- The board is **one shared physical resource**, serialized across lanes (never two sessions at
  once). Built-in subagents never touch it.

## 3. What is DONE (don't rebuild — pointers only)
- **Three benchmark suites on QEMU:** CoreMark ✓, BEEBS 82/82 ✓, RV8 7/7 ✓ (only C++ `bigint`
  deferred). `state/current-state.md` "Verified baseline".
- **SQLite 3.53.3 runs end-to-end in a pure-cap domain on QEMU** (all 8 bring-up gaps closed;
  CREATE/INSERT/SELECT correct). `benchmarks/sqlite/`.
- **CHERI-vs-Capstone perf comparison DONE + in the paper** (QEMU-to-QEMU, microbench + BST):
  eager CHERI ~14–17 M instr/free vs our O(1) +5 instr/op. `evaluation.tex`.
- **C1 spatial narrowing** (globals default-on, stack default-on, real umm_malloc heap) shipped +
  functionally validated. **C2 provenance verifier** = redesigned v2, awaiting reviewer sign-off,
  do NOT implement verbatim.
- **Silicon ABI validated on the board:** a compiler-built, globals-using domain creates, runs,
  returns on the captype-fixed CVA6 via the gp-captable / gp-free + cscratch-`gp` path
  (retval `554745961`). `history/22-07-2026_18-05-00_*` UPDATE 23-07.
- **Per-primitive silicon cycle numbers** measured (borrow(N)≈75+3N/2, load2/shrink1/mrev50/
  delin+revoke121). memory `project_fpga_silicon_measurement_status`.

## 4. What is IN FLIGHT (the live fronts)
- **Silicon micro-benchmark perf numbers** — the deadline-critical deliverable. 7 ladder rungs are
  QEMU-green in the silicon config and ready to run on the board. → `fpga-ladder-perf-task-B.md`.
- **Monitor regen is broken** — the central blocker. Rebuilding `fw_jump.elf` (QEMU) or the FPGA
  firmware monitor from the current `capstone-c` **boot-hangs**; the working firmware is an
  unreproducible prebuilt (older compiler state, smaller frames `s0–s6/−368` vs current
  `s0–s11/−464`). This blocks **large-`.rodata`/const delivery → SQLite on silicon**. We do NOT
  need the board owner to fix it (two self-service paths). → `monitor-regen-audit-task-B.md`;
  memory `project_opensbi_monitor_rebuild_include_wrapper`.
- **SQLite on silicon** — the paper's comprehensive-benchmark number. Blocked behind the monitor
  regen (needs the large-`.rodata` monitor change). Delivery = tier-2b (JTAG `load_image` into a
  reserved RAM region + resident controller; confirm the region with the board owner first). →
  `sqlite-on-silicon-scoping.md`.
- **xlang (cross-language FFI temporal-safety) case study** — the 2nd case study. Phase-1
  stock-toolchain bug reproduction is decoupled + handed to an external collaborator; Phase-2
  (capability mechanism) stays in-lane. → `xlang-repro-task.md`, master plan
  `ndss-pivot-master-plan.md`, memory `project_xlang_benchmark_direction`.

## 5. Blockers, distilled
| Blocker | Impact | Path |
|---|---|---|
| Monitor regen boot-hangs (unreproducible prebuilt) | large-`.rodata`/SQLite on silicon | bisect capstone-c OR root-cause the miscompile — `monitor-regen-audit-task-B.md` |
| Board is flaky/slow, human-in-loop | all board runs | fast_xfer + one-session batching; tier-2b for suites |
| C2 verifier is a hygiene checker, not a proof | paper security claim | redesigned v2, gated on reviewer sign-off |
| `-O1+` codegen gaps (i128 xor/or ISel, fp128, cscincoffset) | RV8 0/7 at `-O1+` | pre-existing, orthogonal; silicon rungs are `-O0` |

## 6. Priority queue for the successor (B)
1. **FPGA micro-benchmark perf numbers** (`fpga-ladder-perf-task-B.md`) — deadline-critical; run
   the 7 ready rungs on the board, `mcycle` + `retval==oracle`, one session.
2. **Monitor regen audit** (`monitor-regen-audit-task-B.md`) — board-free; unblocks (3).
3. **SQLite on silicon** (`sqlite-on-silicon-scoping.md`) — after (2) lands the large-`.rodata`
   monitor change; tier-2b delivery.
4. (Parallel, external) **xlang Phase-1** repro is with the collaborator; Phase-2 is in-lane later.

## 7. Hard rules (the ones that bite)
- **No real-person names in any committed/shared content** — neutral roles only. Person-facing
  notes → `/tmp/capstone/` only.
- **Never commit or share the FPGA console URL/token** — env/`~/.config` only, placeholder
  `<FPGA-CONSOLE-URL>` in committed text.
- **Never rebuild the monitor** for a board run (use the working prebuilt); the regen fix is its
  own task.
- **Serialize** QEMU suites (shared `rootfs.ext2` lock) and board sessions (across lanes).
- `ninja -j90` never `-j112`; no submodule-source commits; commit only when asked; no
  `Co-Authored-By`; bug-fix/audit notes → `history/` (dated), not `design/`.

## 8. Canonical pointers
- Rules: `CLAUDE.md`, `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md` (archived). State: `state/current-state.md`, `current-next-step.md`.
- FPGA: `ref/HOW-TO-LAUNCH-ON-FPGA.md`, `ref/gp-free-silicon-smoke-runbook.md`,
  `fpga_driver/PROTOCOL.md`.
- Tasks: `plans/fpga-ladder-perf-task-B.md`, `plans/monitor-regen-audit-task-B.md`,
  `plans/sqlite-on-silicon-scoping.md`, `plans/xlang-repro-task.md`,
  `plans/ndss-pivot-master-plan.md`.
- Key memories: `project_opensbi_monitor_rebuild_include_wrapper`,
  `project_silicon_gp_delivery_boardowner_guidance`, `project_gp_captable_codegen`,
  `project_fpga_silicon_measurement_status`, `project_board_transfer_tiers`,
  `project_fpga_fw_payload_build_recipe`.
