# B-lane task: broaden the silicon ladder within the three benchmark suites

**Autonomy: high.** This states the *goal* and the *guardrails*, not the steps —
pick your own method. You are as capable as the A lane; the guardrails below are
cold-start context you'd otherwise pay to rediscover, not a leash.

**First, adopt this repo's operating rules as your own.** Read the **repo-root
`CLAUDE.md`** (committed project instructions — your Claude Code session auto-loads
it from the root of your own checkout; same file the A lane works under) and
**`capstone/agent-handoff/DELEGATION.md`**, and treat their hard constraints — plus
the "Permanent repository rules" restated at the bottom — as binding. This includes
the **context / compaction discipline** (manage context deliberately; recommend
`/compact` only at a safe checkpoint and with a short keep-vs-compress brief; never
compact unilaterally). Then `source capstone/tests/capstone-test-env.sh` and skim
`agent-handoff/state/current-state.md` for the live picture.

## Goal (the deliverable)
Broaden the silicon ladder with **more integer kernels drawn from the three suites
the paper already commits to — CoreMark, RV8, and BEEBS** (no new suite). Each added
kernel compiles in the silicon config and, run in a pure-cap domain on QEMU, returns
a **deterministic result == a native `cc -O0` oracle**, with the static gate showing
**`cjalr=0`**. Target roughly **3–5 more rungs**; depth within the sanctioned suites,
not a fourth benchmark.

**Stay within the three suites — do not introduce a fourth.** The perf story is
CoreMark + RV8 + BEEBS by design. (Note: `dhrystone` is already an *RV8* kernel we run
on the non-silicon path, so it is not a new suite — but it is heap/large-const-heavy,
so it is not a good early silicon pick; see the guardrails.) If you think a genuinely
new suite is warranted, raise it with the A lane / PI first — don't just add one.

## Suggested picks (your call on the exact set)
Existing rungs: `matmult_int`, `beebs_insertsort`, `beebs_prime`, `beebs_recursion`,
`beebs_crc32`, `rv8_primes`, `coremark_matrix`, `beebs_crc32big`. Good next candidates
are **small, pure-integer, self-contained** kernels that fit the existing path:
- **BEEBS is the richer near-term source**: e.g. `bubblesort`, `fibcall`,
  `janne_complex`, `ns`, and similar small integer kernels — pick ones with small/no
  global tables and no libc/FP dependence.
- **RV8** beyond `primes` is mostly **blocked for now**: the other RV8 kernels
  (`aes`, `sha512`, `norx`, `miniz`, `dhrystone`, `qsort`) pull **large const tables
  and/or the heap allocator**, which need A's not-yet-landed large-RO path (and the
  heap shim). Leave those until large-RO lands; don't build a mechanism for them.
Prefer variety of *shape* (sorting, recursion, bit-twiddling, simple number theory)
over piling on near-duplicates.

## Why this is clean and independent
The A lane is actively finishing the **large-`.rodata` monitor-copy delivery path**
and is editing exactly these files: `gen-gp-captable-glue.py`, `link-gpfree.ld`, the
OpenSBI monitor (`sbi_capstone.c`), and possibly `build-ladder-domain.sh` /
`run-ladder-qemu.sh`. **Your task must not touch any of those.** You only *add* new
rung files — `<base>_kernel.h`, `<base>_app.c`, `<base>_host.c`, and a
`run-<base>-qemu.sh` wrapper if you need a non-default opt level (see rung 7's
`run-coremark-matrix-qemu.sh` for the `DOMAIN_OPT_LEVEL` pattern). Adding files only
= **zero file overlap → zero merge conflict** with A's in-flight work.

## The guardrails that keep us disjoint (read these)
- **Do NOT modify** `gen-gp-captable-glue.py`, `link-gpfree.ld`, `sbi_capstone.c`,
  `build-ladder-domain.sh`, or `run-ladder-qemu.sh`. If a rung seems to *need* a
  change there, **stop and coordinate** — that surface is A's this week.
- **The large-`.rodata` path is A's and is NOT yet landed.** If a kernel has a
  **large initialized `const` table** — a **file-scope `const` array > 256 B**, or
  **any** `static` / function-local (`.L…`) large `const` — it will fail the build or
  hit the half-built copy path. **Do not build or extend a delivery mechanism for
  it.** Pick a kernel whose data is small / runtime-computed instead. (Large all-zero
  `.bss` is fine — the generator zeroes it with a runtime loop, `160a7613`.)
- **Single-TU only (RISK A).** `getGpCaptableIndex` numbers globals *per module*, so a
  multi-`.c` domain collides on the one gp cap-table. Amalgamate into one `_kernel.h`
  (like rung 7 did for CoreMark). Provide tiny **freestanding** helpers in the kernel
  (no libc).
- **Pointer-size independence.** The domain and the native oracle must fold the
  *same* value; don't checksum anything pointer-sized (see rung 7's note). Prefer
  kernels whose result is pure integer arithmetic.
- Match the oracle convention: `<base>_kernel.h` = shared compute + checksum,
  `<base>_app.c` returns it via `*res`, `<base>_host.c` prints the same value.

## Already solved — reuse, don't re-derive
- Rung 7 (`coremark_matrix`) shows the amalgamation pattern, the pointer-size-
  independence reasoning, and the `-Os` wrapper trick for the 4 KiB PCC code window
  (`.text` must fit `[base, base+0x1000)`; drop to `-Os` if a kernel overflows at
  `-O0`, and lean on the oracle-match assertion to catch any miscompile).

## Permanent repository rules — adopt these as your own (non-negotiable)
Full standing rules (canonical: `CLAUDE.md` + `DELEGATION.md`). Treat them exactly as
the A lane does:
1. **Never mention any real person by name — anywhere.** PI / supervisor / board
   owner / collaborator → neutral roles, in every committed/shared file, commit, doc,
   or report. Permanent and absolute. (Upstream `lldb/`, `llvm/` files are not ours —
   leave their names alone.)
2. **No `Co-Authored-By:` lines**; no worker/agent identity in commit messages;
   **don't rewrite pushed history**; **commit only when the user asks.**
3. **Never commit debug/report/session-note files** (`*_DEBUG_CHECKPOINT.md`, etc.).
4. **Manager/PI-facing summaries → `/tmp/capstone/`**, never the repo.
5. **Serialize the QEMU suites** — shared `rootfs.ext2` write-lock, never two QEMU
   runs in parallel (the A lane may be running QEMU for the large-RO end-to-end test;
   don't overlap — coordinate timing).
6. **`ninja -j90`** (~80% of 112 cores), **never `-j112`** (parallel debug-link storm
   hangs the whole box, no SSH).
7. **Bug-fix / root-cause / audit notes → `history/`** (dated
   `DD-MM-YYYY_HH-MM-SS_name.md`), **not `design/`** (`design/` = architecture only).
   Active plans → `plans/`.
8. Commit to **`capstone-bootstrap-b`** only. Board/FPGA is off-limits (batched,
   human-in-loop, A lane).

## Start here
`git switch capstone-bootstrap-b && git merge origin/capstone-bootstrap` to pick up
the current ladder (rungs 1–7 incl. your CoreMark matrix; `origin/capstone-bootstrap`
is at the latest A push). Then work only under
`capstone/tests/runtime-qemu/silicon-ladder/`, adding files.
