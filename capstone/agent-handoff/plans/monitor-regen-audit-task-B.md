# B-lane task: recover a rebuildable monitor (audit the boot-hang, no board)

**Autonomy: high. Board-free. Priority: SECOND** — behind the deadline-critical FPGA perf numbers
(`fpga-ladder-perf-task-B.md`). This unblocks large-`.rodata`/const-table delivery → **SQLite on
silicon**, which is the paper's comprehensive-benchmark number, so it matters — but the micro-bench
perf table comes first.

## UPDATE 2026-07-25 — the toolchain answer is KNOWN; try the fast path first

The board owner confirmed the monitor build reference is **`caplifive-system`** (the source of the
working firmware). That repo **pins its own `capstone-c` submodule** (`sw/capstone-c`) at branch
**`bugfix`, commit `508342a`** — *not* our tree's `master`@`8cda52c`. The two diverge right after
`4899cf9`; the `bugfix` side carries two commits we never had, including **`3780447 "Fixed overly
large alignment for gct"`** (gct = the global-constructor table the monitor uses) — the very
plausible reason the regen frames differ (`s0–s11/−464` vs the good prebuilt's `s0–s6/−368`).

**So the likely fix is not archaeology — it is using the right, already-pinned compiler.**
**Fast path (do this first):**
1. Build the monitor with `capstone-c` at `caplifive-system`'s pin: check out our
   `capstone/capstone-c` submodule to `508342a` (or build against
   `capstone/caplifive-system/sw/capstone-c`, already at `508342a`). Confirm the commit:
   `git -C capstone/caplifive-system/sw/capstone-c log --oneline -1` → `508342a`.
2. Regen + relink the QEMU monitor with **that** compiler
   (`rm sbi_capstone_dom.c.S`, `make build A=opensbi-rebuild CAPSTONE_CC_PATH=<508342a checkout>`).
3. Boot-test in QEMU: OpenSBI banner + a known-good domain (e.g. `matmult_int`) runs.
4. If it boots → the "toolchain gap" is closed; **pin our `capstone-c` (and the ladder/monitor
   builds) to `508342a`** and record it. Then land the large-`.rodata` monitor change and rebuild.

Only if the fast path does **not** boot do the deeper audit (Paths A/B below). Also note the board
owner asked "did you try reproducing the whole thing?" — i.e. build `caplifive-system` end-to-end
as-is (its pinned submodules) before assuming a miscompile; a clean end-to-end build is the control.

**Adopt the repo rules** (repo-root `CLAUDE.md`, permanent list at the bottom of
`fpga-ladder-perf-task-B.md`). `source capstone/tests/capstone-test-env.sh`; read
`state/current-state.md` and memory `project_opensbi_monitor_rebuild_include_wrapper` (the WARNING
section is the whole problem).

## The problem
The Capstone OpenSBI monitor logic lives in `capstone-sbi/sbi_capstone.c` (two copies: QEMU
`fw_jump.elf` builds the `components/opensbi` copy; FPGA firmware builds the caplifive-system
copy). **The working `fw_jump.elf` is an unreproducible PREBUILT** (md5 `6724bcb3…`, Jul-22).
Regenerating it from the current `capstone-c` yields a **different, boot-hanging** image
(md5 `788f8a1a…`, zero serial, hangs *before* `create_domain` → whole-monitor early miscompile).

**Confirmed NOT commit-drift:** both `capstone-c` `8cda52c` and the tree-pinned `4899cf9` produce
the same broken monitor. Diffing good-vs-regen `.c.S`: the current compiler allocates **more
callee-saved regs / bigger frames** (`s0–s11`, frame −464) than whatever built the prebuilt
(`s0–s6`, frame −368). So the working monitor came from a `capstone-c` state not reachable from
the current checkout.

**Why this matters:** we cannot add the small **monitor-side change** that delivers large
`.rodata`/`const` tables to a domain (needed for SQLite + const-table benchmarks) until the
monitor can be regenerated to a booting image — on both QEMU and, ultimately, the FPGA firmware.

## Goal
Restore the ability to **rebuild a booting monitor from source**, then (only then) land the
large-`.rodata` delivery change. The board owner may be *asked* which commit built the prebuilt
(`/tmp/capstone/boardowner-monitor-toolchain-question.md`), but that's a fragile shortcut — the
prebuilt may have come from a dirty/deleted tree. **We own two self-service paths; prefer them.**

## Path A — bisect `capstone-c` (mechanical, recovers rebuild capability)
The good prebuilt has a date (Jul-22 20:04) and a codegen signature (`s0–s6`/frame −368). Walk
`capstone-c` history, rebuild the monitor at each candidate, boot-test in QEMU, find the last
commit that produces a **booting** `fw_jump`.
- Rebuild recipe (memory `project_opensbi_monitor_rebuild_include_wrapper`): `rm
  components/opensbi/lib/sbi/sbi_capstone_dom.c.S` then `make build A=opensbi-rebuild
  CAPSTONE_CC_PATH="$(realpath ../capstone-c)"` from `caplifive-buildroot`.
- Boot-test: does it print the OpenSBI banner + run a known-good domain (e.g. `matmult_int`)?
- **Safety:** the working prebuilt is backed up at `/tmp/capstone/fw_jump.elf.orig.bak` / `.bak2`
  (`cp` back into `caplifive-buildroot/build/images/fw_jump.elf` to restore). **Never leave a
  broken `fw_jump` in place — it boot-breaks the shared QEMU for BOTH lanes.** Restore after every
  probe. Coordinate with the A lane so you don't clobber a shared run.

## Path B — root-cause the boot-hang (higher value, real bug)
The current-compiler monitor hangs before `create_domain` → a whole-monitor early-boot miscompile.
Find it: gdb-boot the hanging `fw_jump` in QEMU, locate where it diverges from the good one
(likely in the larger-frame prologue / callee-save save-restore or a clobbered reg). This is
squarely our `capstone-c` codegen expertise. **This is the more valuable outcome** — a compiler
that silently miscompiles the monitor is a latent hazard for *all* future firmware work, including
the FPGA firmware you'll eventually rebuild for SQLite. If you find + fix the codegen bug, we can
rebuild the monitor from the *current* toolchain (no pinning fragility).

## After the rebuild capability is restored
Land the large-`.rodata` delivery change (the copy-into-domain-data-region loop; the A lane's
correct implementation is described in `plans/large-ro-delivery-completion-task-A.md` — the code
was right, only the rebuild was broken). Then verify SQLite's const tables reach the domain on
QEMU, and prep the same change for the FPGA firmware monitor (still no board needed for the QEMU
half).

## Guardrails
- **No submodule-source commits.** The monitor `.c` edits stay local experiments; the fix that
  gets committed is in `capstone-c` (Path B) or a pinned-commit note (Path A) — put findings in
  `history/DD-MM-YYYY_HH-MM-SS_*.md`, not `design/`.
- **Never leave the shared `fw_jump` broken** (restore from backup after every probe).
- Board-free task — do not touch the board for this.
- Commit only when the user asks; `capstone-bootstrap-b`.

## Start here
Path B first if you want the durable fix; Path A first if you want the fastest unblock. Either
way, keep the good prebuilt restored between probes. Report which path landed it in a dated
`history/` note.
