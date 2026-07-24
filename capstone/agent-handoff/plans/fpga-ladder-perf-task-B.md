# B-lane task: FPGA cycle-count perf numbers for the silicon-ladder rungs

**Autonomy: high.** Goal + guardrails, not steps — you know the FPGA path already (agentB-016/
017). The guardrails below are the cold-start context + the few things that changed.

**First, adopt this repo's operating rules as your own.** Read the repo-root `CLAUDE.md` and
`capstone/agent-handoff/DELEGATION.md` (hard constraints + the permanent rules restated at the
bottom of this doc), then `source capstone/tests/capstone-test-env.sh` and skim
`state/current-state.md`. **Note the updated board rule** (below): the FPGA is now explicitly
open to either lane, but serialized.

## Goal (the deliverable — this is the deadline-critical one)
Produce **real FPGA cycle counts** for the silicon-ladder micro-benchmarks — the paper's
headline "on real hardware" perf numbers. For each ready rung: run it in a pure-cap domain on
the **Genesys2 CVA6 Capstone** board, confirm **correctness** (the domain returns the same
value as the native `cc -O0` oracle), and record its **cycle count** (`mcycle`). Output = a
perf table `rung → cycles (+ correctness ✓)` for the eval section.

## The ready set (no monitor changes needed)
These 7 rungs already run in the silicon config on QEMU (gp-captable + gp-free call/ret +
shrink-off + `+m`) and need **no** large-`.rodata`/monitor work:
`matmult_int`, `coremark_matrix`, `rv8_primes`, `beebs_crc32`, `beebs_insertsort`,
`beebs_prime`, `beebs_recursion` (in `tests/runtime-qemu/silicon-ladder/`). Start with
`matmult_int` (simplest; already board-validated as a globals-using domain) to prove the
end-to-end pipeline, then batch the rest.

## What's actually new here (the bridge you'll build)
The ladder harness (`run-ladder-qemu.sh`) targets **QEMU**; the board path lives in
`tests/rtl-smoke/` (`build-borrow-cost-fpga*.sh`, `fpga_driver/` — `fast_xfer.py`,
`fpga_console.py`, `run_rtl_smoke.py`, `PROTOCOL.md`; `fpga_instrument.h` for `mcycle`). Your
job is to **bridge a ladder rung onto the board path**: build the rung's `.dom` in the silicon
config (reuse `build-ladder-domain.sh`), wrap it with the `mcycle` instrumentation
(`fpga_instrument.h`), transfer + run it via the existing `fpga_driver` / board runbook
(agentB-016/017), read back the retval + cycle count. This is bounded engineering, mostly
plumbing between two paths you already have.

## Hard guardrails
- **DO NOT rebuild the monitor / firmware.** There is a confirmed **toolchain gap**: the QEMU
  monitor can't currently be regenerated to a booting image (every regen boot-hangs; see
  `plans/large-ro-delivery-completion-task-A.md` 1-STATUS v3 and the memory warning). Use the
  **existing working FPGA firmware prebuilt** as-is. If a rung seems to need a monitor change,
  it's out of scope for this task (that's the blocked large-`.rodata` track) — pick another
  rung. All 7 above need no monitor change.
- **The board is a single shared resource, serialized across lanes.** Per the updated rule,
  either lane may run it, but **never two board sessions at once** — coordinate timing with the
  A lane before you start, and hand back when done. Board etiquette + the secret token apply
  (the token comes from the user, out-of-band — never commit or paste it anywhere).
- Correctness gate stays: `retval == native oracle`, and the static gate `cjalr=0` (already
  true for these rungs). A cycle number without the correctness check is not a result.
- Measure `mcycle` around the domain workload consistently across rungs (same enter/exit
  points) so the numbers are comparable; note any fixed per-call overhead separately.

## Deliverable format
A committed results note (`history/DD-MM-YYYY_HH-MM-SS_fpga-ladder-perf.md`) + a compact table
(rung, cycles, correctness) suitable for the paper's perf section. Manager/PI-facing summary →
`/tmp/capstone/`, not the repo.

## Permanent repository rules — adopt as your own (non-negotiable)
1. **Never mention any real person by name — anywhere** (PI/board owner/collaborator → neutral
   roles), in every committed/shared file, commit, doc, report. (Upstream `lldb/`,`llvm/` files
   are not ours.)
2. **No `Co-Authored-By:`**; no worker/agent identity in commits; **don't rewrite pushed
   history**; **commit only when the user asks.**
3. **Never commit debug/report/session-note files.**
4. **Manager/PI-facing summaries → `/tmp/capstone/`**, never the repo.
5. **Serialize the QEMU suites** (shared `rootfs.ext2` lock) — and, per above, **serialize
   board sessions across lanes** too.
6. **`ninja -j90`**, never `-j112`.
7. **No commits into submodule source.**
8. **Bug-fix/root-cause/audit notes → `history/`** (dated), not `design/`. Active plans →
   `plans/`.
9. Commit to **`capstone-bootstrap-b`** only. The board is **not** off-limits to you, but is
   serialized across lanes (coordinate first).

## Start here
`git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap` to pick up the ladder
rungs + the board-rule update + this task. Then coordinate a board window with the A lane before
your first run.
