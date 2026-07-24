# B-lane task: CoreMark on the silicon ladder (via amalgamation)

**Autonomy:** high. This states the *goal* and the *guardrails*, not the steps —
choose your own method. You are as capable as the A lane; the guardrails below are
context you'd otherwise pay a cold-start tax to rediscover, not a leash.

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

## Guardrails (hard constraints — inherited, non-negotiable)
- **No real person's name anywhere** in committed/shared content (neutral roles only).
- **Serialize the QEMU suites** — the shared `rootfs.ext2` write-lock means never
  two matrix/QEMU runs in parallel (A may be running QEMU; coordinate / don't overlap).
- **`ninja -j90`**, never `-j112` (a `-j112` debug-link storm hangs the whole box).
- **No commits into submodule *source*** (`capstone-qemu`/opensbi/buildroot/system).
- Commit to **`capstone-bootstrap-b`** only; **no `Co-Authored-By`**, no agent
  identity in messages; don't rewrite pushed history; commit only when asked.
- Bug-fix / root-cause notes → `history/` (dated `DD-MM-YYYY_HH-MM-SS_name.md`),
  not `design/`.

## Start here
`git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap` to pick up
the ladder + the corpus-validated memcpy fix + the gated gp codegen (the A lane just
folded its sandbox into mainline; `origin/capstone-bootstrap` is at `160a7613`).
Then work in `capstone/tests/runtime-qemu/silicon-ladder/`.
