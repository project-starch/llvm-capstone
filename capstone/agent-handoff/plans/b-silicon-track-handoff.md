# B-lane handoff: own the silicon / FPGA / firmware track

**This hands you (B lane) the whole silicon track — not a single task. Run it with high autonomy.**
The A lane is context-constrained and stepping back; you own planning, board sessions, the firmware,
the silicon-compat compiler work, and the silicon perf numbers. A stays available for review and for
the paper. Goal + guardrails below; no step-by-step, no per-task sign-off — decide and move.

## 0. First, load context (don't skip — it's the whole point of this doc)
```
git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap
source capstone/tests/capstone-test-env.sh
```
Read, in order: `CLAUDE.md`, `history/29-07-2026_ARCHIVED_DELEGATION-lane-a-b.md` (archived), then **`plans/A-lane-handoff-briefing.md`** (the map),
then this doc. The briefing indexes everything; this doc is the *current* strategy on top of it.

## 1. What the board owner's two answers give us (READ THIS — it reshapes the plan)

We got two answers from the board owner this week. Together they **substantially de-risk the whole
silicon track**. Here is what each one actually means for us.

### Answer A — "the monitor build reference is `caplifive-system`" (the monitor-regen unblock)
**What we were stuck on:** rebuilding the OpenSBI monitor (`fw_jump.elf` / FPGA firmware) from our
tree's `capstone-c` produced a boot-hanging image; the working firmware was a "mysterious
unreproducible prebuilt." We (wrongly) treated this as a compiler bug to reverse-engineer, and even
drafted a long question to the board owner about it.

**What the answer means:** the answer was *in our own tree the whole time.* `caplifive-system` is the
canonical, self-consistent firmware build tree, and it **pins its own `capstone-c`** at branch
**`bugfix`@`508342a`** — which is **not** our tree's `master`@`8cda52c`. They diverge right after
`4899cf9`, and the `bugfix` side carries `3780447 "Fixed overly large alignment for gct"` (gct = the
global-constructor table the monitor relies on) — the very plausible reason the regenerated monitor
had bigger stack frames (`s0–s11/−464` vs the good prebuilt's `s0–s6/−368`) and boot-hung. So the
"toolchain gap" is almost certainly **just the wrong compiler**, not a real miscompile.

**What it unlocks:** if you rebuild the monitor with `508342a` and it boots, we **regain the ability
to change the firmware**, which reopens two things that were both blocked on it:
- The **`fence.i` domain-boundary fix** (`plans/curried-crunching-gizmo.md`) — this is what actually
  kills the **~2.5 min/rung power-cycle cost** (today a second domain at the same VA within one boot
  hangs its `cscall`; the fix lets you run many domains per boot).
- Any monitor-side feature we might still want (though see Answer B — we may need far less monitor
  change than we thought).

**The meta-lesson (adopt it):** `caplifive-system`'s submodule pins are the source of truth for the
firmware build. **Check `.gitmodules` / `git submodule status` there before assuming anything about
the toolchain** — and before asking the board owner. (Also: keep board-owner messages *short and
human* — no long AI-generated context dumps. See memory `feedback_boardowner_short_msgs_check_intree`.)

### Answer B — "let the host userspace process do it, not the monitor" (large-RO delivery)
**What we were stuck on:** SQLite's big `const` tables overflow the domain's `[base, base+0x1000)`
PCC window if materialized as `li`/`sd` code, so we need another way to get initialized read-only
data into a domain. We had prototyped this as an **M-mode monitor** memcpy.

**What the answer means:** the board owner's endorsed design is for the **host userspace process**
(the Linux process that creates the domain via the `/dev/capstone` ioctls) to copy the initializer
bytes into a fresh domain **data** region and hand the domain a **data-authority cap** to it — *not*
the M-mode monitor. Rationale: keep M-mode minimal; the host already has the image bytes. He added
*"for now whatever works is fine,"* so you can prototype on QEMU with the simplest thing first.

**What it unlocks:** the SQLite-on-silicon large-RO path likely needs **little or no new monitor
code** — the work moves to the host userspace controller (which you already build for every board
run). That's a big de-risk: it decouples SQLite-on-silicon from the monitor-rebuild entirely. Factor
any prototype so the copy lives in (or can move to) host userspace, not the monitor.

### Net effect of A + B on our plan
- The monitor is **rebuildable** (Answer A) → the per-rung cost and any firmware iteration are back
  on the table.
- The comprehensive-benchmark (SQLite) delivery **mostly sidesteps the monitor** (Answer B) → less
  firmware risk than we feared.
- **What is still genuinely open and on the critical path is the compiler** — the gp-captable
  silicon miscompile (next section). Neither answer addresses it; it's ours to crack.

## 2. The critical path to the deadline (priority order, with the "why")

**#1 — Root-cause the gp-captable silicon miscompile (highest value, board-light).**
This is the tightest deadline gate, and here is the reasoning: the paper's silicon perf table needs
correct-and-measured rungs, but **4 of our 6 measurable rungs currently miscompute on hardware**
(fresh binaries, shrink OFF, QEMU-correct) — see the results table at the top of `current-state.md`
and `history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`. So the perf
numbers **can't be completed until this is fixed**, *and* it blocks the silicon-compatibility claim
*and* the `capstone-gp-free` merge. The shape is consistent (store into a global array while a live
accumulator is in flight → the accumulator returns address-contaminated), we have **4 fresh hardware
reproductions + 2 passing controls + the minimal `rc_const0`/`rc_p1` pair**. Suggested attack: diff
the `-O0` register schedule / codegen of a PASS vs a FAIL rung; if our codegen looks clean, it
becomes a tight localization request to the board owner with the 4 reproductions (kept *short*).
Everything else waits behind this for the deadline table.

**#2 — Rebuild the monitor with `508342a` (fast, board-free, keystone unblock).**
Per Answer A. `plans/monitor-regen-audit-task-B.md` has the fast path (build with `508342a`,
QEMU-boot-test, pin if green). Do this early — it's quick and, if it works, unlocks the `fence.i`
fix that removes the 2.5 min/rung cost, which pays for itself immediately in board throughput. If
the fast path does *not* boot, the deeper audit (Paths A/B in that doc) is the fallback.

**#3 — Finish the silicon perf table.** Once #1 lands: the 4 fixed rungs give real cycle numbers
(add to the 2 already trustworthy: `rv8_primes` 17,283,292; `beebs_prime` 47,804). Get
`coremark_matrix` a verdict (transfer-blocked today — split payload or avoid typing base64 into the
shell). With #2's `fence.i` fix, run the batch in one boot instead of one power-cycle per rung.

**#4 — SQLite on silicon (the comprehensive benchmark).** Per Answer B: prototype the large-RO
delivery in the **host userspace controller** (copy image bytes → domain data region → hand a
data-authority cap), simplest-thing-first on QEMU, then port to the board. `plans/sqlite-on-silicon-
scoping.md` (updated with Answer B). This is the paper's comprehensive number; it comes after the
micro-bench table is correct.

## 3. What's already in your hands (don't rebuild)
The FPGA driver (HTTP+Socket.IO, board-validated), `fast_xfer` (~3×), the ladder→board bridge, the
mcycle instrumentation, and the results harness — all yours, all committed (you built most of them).
`ref/HOW-TO-LAUNCH-ON-FPGA.md` has the board mechanics, transfer tiers, the multi-domain-hang and
stale-dom gotchas. Per-primitive silicon cycle numbers are already measured. The 2 correct perf
points are quotable now.

## 4. Guardrails (the ones that bite)
- **No real-person names in any committed/shared content**; person-facing notes → `/tmp/capstone/`.
- **Never commit/share the FPGA console URL or token** (out-of-band from the user only).
- **Board serialized across lanes** — coordinate a window with A; never two board sessions at once;
  lock → power-cycle → run → **power off + unlock in `finally`**.
- **Monitor rebuilds use `caplifive-system`'s `capstone-c` (`508342a`)**, not our tree's `master`.
  A monitor rebuild swaps the *shared* `fw_jump.elf` — coordinate with A and restore the good
  prebuilt (`/tmp/capstone/fw_jump.elf.orig.bak2`) after every QEMU probe.
- `ninja -j90` never `-j112`; serialize QEMU suites (shared `rootfs.ext2` lock); no submodule-source
  commits; commit only when the user asks; no `Co-Authored-By`; bug-fix/audit notes → `history/`
  (dated), not `design/`. Commit to `capstone-bootstrap-b`.

## 5. Pointers
- Strategy/index: `plans/A-lane-handoff-briefing.md`. This doc: current strategy on top.
- Tasks: `plans/monitor-regen-audit-task-B.md` (Answer A fast path), `plans/sqlite-on-silicon-
  scoping.md` (Answer B), `plans/fpga-ladder-perf-task-B.md` (perf mechanics),
  `plans/curried-crunching-gizmo.md` (the `fence.i` fix).
- Open bug: `history/23-07-2026_17-30-00_gp-captable-silicon-array-loop-miscompute-OPEN.md`;
  latest sweep `history/25-07-2026_03-58-47_fpga-ladder-perf-sweep-results.md`.
- Memories: `project_opensbi_monitor_rebuild_include_wrapper` (Answer A + 508342a),
  `project_gp_captable_codegen` (the open miscompile), `project_silicon_gp_delivery_boardowner_
  guidance`, `project_fpga_silicon_measurement_status`, `project_board_transfer_tiers`,
  `feedback_boardowner_short_msgs_check_intree`.

## 6. Start here
Kick off #2 (monitor rebuild with `508342a` — fast, board-free, and if it boots it changes your
board economics immediately), and in parallel start #1 (the miscompile is the deadline gate — pure
analysis to begin, no board needed). Coordinate a board window with A before your first hardware run.
