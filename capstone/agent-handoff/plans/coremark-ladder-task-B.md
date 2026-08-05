# B-lane task: CoreMark on the silicon ladder (via amalgamation)

**Autonomy:** high. This states the *goal* and the *guardrails*, not the steps —
choose your own method. You are as capable as the A lane; the guardrails below are
context you'd otherwise pay a cold-start tax to rediscover, not a leash.

**First, adopt this repo's operating rules as your own.** Read the **repo-root
`CLAUDE.md`** (the committed project instructions — your Claude Code session
auto-loads it; it's the same file the A lane works under, present at the root of
your own checkout) and **`capstone/agent-handoff/DELEGATION.md`**, and treat their
hard constraints — plus the "Permanent repository rules" restated at the bottom of
this doc — as binding for everything you do here. This includes the **context /
compaction discipline** (manage context deliberately; recommend `/compact` only at a
safe checkpoint and with a short keep-vs-compress brief; never compact unilaterally).
Then `source capstone/tests/capstone-test-env.sh` and skim
`agent-handoff/state/current-state.md` for the live picture.

## Goal (the deliverable)
Get **CoreMark** running as a silicon-ladder rung: it compiles in the silicon
config and, run in a pure-cap domain on QEMU, returns its **correct checksum ==
a native `cc -O0` oracle**, with the static gate showing **`cjalr=0`**. That's the
whole bar — a headline breadth benchmark proven in the silicon config on QEMU.
(Board runs are deferred/batched and stay in the A lane; do **not** drive the board.)

## Why this is a clean, independent task
The six existing ladder rungs (matmult, insertsort, crc32, recursion, prime,
primes) already run in the silicon config. The harness is
`capstone/tests/runtime-qemu/silicon-ladder/` — `run-ladder-qemu.sh <base>` builds
`<base>_host.c` (native oracle) + `<base>_app.c` (domain defining
`domain_main(unsigned *res, unsigned func)`) and asserts `retval == oracle`. The
silicon config = `-capstone-gp-captable` + gp-free call/ret + **shrink off**
(`-capstone-shrink-stack=false -capstone-shrink-globals=false`) + `+m`, built `-O0`;
`build-ladder-domain.sh` wires it, `gen-gp-captable-glue.py` builds the per-app
cap-table glue from the compiler's `.capstone_gp_table` descriptor. This is
disjoint from the A lane's current work (the SQLite large-`.rodata` monitor
mechanism), so we run in parallel with no file collisions.

## The one real obstacle — RISK A — and the suggested sidestep
`getGpCaptableIndex` (`CapstoneISelDAGToDAG.cpp:112`) numbers globals **per
module**, so a multi-TU domain (CoreMark ships several `.c` files) collides on the
single gp cap-table and emits multiple descriptor headers. **Single-TU is fine.**
Suggested path: **amalgamate** CoreMark's TUs into one translation unit (concatenate
/ `#include` into a single `.c`, like the SQLite amalgamation) so it presents as one
module — this sidesteps RISK A with **no compiler edit**. `-flto` presenting one
module is a possible alternative. A whole-program-index *compiler* fix would also
work but **touches the A lane's Capstone backend files — coordinate before doing
that** (avoid two lanes editing the target codegen). Your call on method; the
amalgamation path is the recommended one because it stays in your lane entirely.

## Things you'll likely hit (already solved — reuse, don't re-derive)
- Large **`.bss`** is handled: the generator now zeroes big `.bss` with a runtime
  loop (landed as of `160a7613`), so a big scratch array won't overflow the code
  window. Large **initialized `.rodata`** is NOT yet solved (it's the A lane's
  board-owner-blocked item) — if CoreMark has a large `const` table, either keep it
  small / runtime-computed (as crc32 does) or flag it; don't build a delivery
  mechanism (that's A's critical path). CoreMark is compute-heavy / small-data, so
  this most likely won't bite.
- Oracle convention: a `<base>_kernel.h` with the shared compute + an FNV-style
  checksum, `<base>_app.c` returns it via `*res`, `<base>_host.c` prints the same
  value. Match the existing rungs.

## Permanent repository rules — adopt these as your own (non-negotiable)
These are the full standing rules for anyone working in this repo (canonical:
`CLAUDE.md` + `DELEGATION.md`). Treat them exactly as the A lane does:
1. **Never mention any real person by name — anywhere.** PI / supervisor / board
   owner / collaborator → neutral roles, in every committed/shared file, commit,
   doc, or report. Permanent and absolute. (Upstream `lldb/`, `llvm/` files are not
   ours — leave their names alone.)
2. **No `Co-Authored-By:` lines**; no worker/agent identity in commit messages;
   **don't rewrite pushed history**; **commit only when the user asks.**
3. **Never commit debug/report/session-note files** (`*_DEBUG_CHECKPOINT.md`, etc.).
4. **Manager/PI-facing summaries → `/tmp/capstone/`**, never the repo.
5. **Serialize the QEMU suites** — shared `rootfs.ext2` write-lock, never two
   matrix/QEMU runs in parallel (the A lane may be running QEMU; don't overlap).
6. **`ninja -j90`** (~80% of 112 cores), **never `-j112`** (parallel debug-link
   storm hangs the whole box, no SSH).
7. **No commits into submodule *source*** (`capstone-qemu`/opensbi/buildroot/system);
   submodule source stays uncommitted.

> **WITHDRAWN 2026-08-05:** the "submodule source stays uncommitted" rule above no longer
> applies. Submodule work is now COMMITTED on a branch (see CLAUDE.md). Keeping the live
> monitor uncommitted nearly cost the trace markers every board verdict depends on.

8. **Bug-fix / root-cause / audit notes → `history/`** (dated
   `DD-MM-YYYY_HH-MM-SS_name.md`), **not `design/`** (`design/` = architecture only).
   Active plans → `plans/`.
9. Commit to **`capstone-bootstrap-b`** only. Board/FPGA is off-limits for this task
   (batched, human-in-loop, A lane).

## Start here
`git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap` to pick up
the ladder + the corpus-validated memcpy fix + the gated gp codegen (the A lane just
folded its sandbox into mainline; `origin/capstone-bootstrap` is at `160a7613`).
Then work in `capstone/tests/runtime-qemu/silicon-ladder/`.
