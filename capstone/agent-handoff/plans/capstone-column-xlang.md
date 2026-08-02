# xlang — the Capstone column

**Written 2026-08-01.** Supersedes `xlang-phase2-seam-TODO.md`, which planned this
column via *real mruby* (the `mrb_open_allocf` seam). That route is dropped — see
"Why shims, not real mruby" below. Item 2 of that doc (the capability-allocator
contract) is **carried forward unchanged** and is still the highest-leverage item
here; the rest of it is moot.

**Goal.** Produce the Capstone half of the xlang catch/no-catch comparison, so the
corpus has the same two-column story the sqlite corpus has.

---

## Read first — most of this is already done

Do not restart any of these:

| Thing | Where | State |
|---|---|---|
| The 14 defect reproductions | `xlang/repro/<key>/` | done — 14/14 reproduce natively |
| The CHERI column | `xlang/cheri/RESULTS.md` | **done** — 14 rows x 3 configs, reproduced from a clean `CHERI_ROOT`, verdicts byte-identical |
| The shims | `xlang/cheri/shims/` | done — 14 shims, `#define`s over `vm_stack_uaf.h` |
| The fidelity gate | `xlang/cheri/check_shim_fidelity.py` | done — 18/18, proves each shim still triggers its defect natively |
| The Capstone pattern to copy | `capstone/benchmarks/sqlite/sqlite_row*_domain.c` + `run-sqlite-row*.sh` | proven on the sqlite corpus |
| The revoking allocator | `capstone/benchmarks/sqlite/revoke_on_free_alloc.h` | proven — SPLIT / MREV / delin, `xFree` revokes the node |

**The only missing thing is the Capstone column itself.**

---

## Why shims, not real mruby

The sqlite corpus states its vehicle asymmetry plainly (`paper/old-parts/evaluation.tex`,
"A vehicle asymmetry we state plainly"): Capstone ran unmodified upstream SQLite,
CHERI could not, and that porting cost is reported *as a finding*.

For xlang we go the other way — **shims on both sides** — for three reasons:

1. **Speed.** The CHERI column is finished and is shim-based. Matching it costs a
   build target, not a bring-up.
2. **Symmetry.** Identical shim source and identical mock allocator on both sides
   means the shims' known bias (real engines recycle on internal free lists no
   revocation observes) applies *equally* to both columns. Neither absolute number
   is realistic; the comparison between them is. Say exactly that in the writeup.
3. **It collapses three seam mechanisms into one.** The real-mruby route needed
   `mrb_open_allocf` for rows 4-15, Rust's `#[global_allocator]` for rows 1-2, and
   `LD_PRELOAD` interposition into prebuilt `libpulse.so` for row 3. The shims are
   plain C, so all three problems disappear.

**The porting-cost finding survives, and is stronger than sqlite's**, because it was
paid for rather than inferred: purecap mruby needed four changes and five
probe-bisection boots; CHERI clang was *completely silent* on the fatal one (tested,
not assumed — `xlang/cheri/mruby-port/why_warnings_miss_it.c`); only 1 of 9 pinned
mruby trees is proven to boot; and rows 1-3 cannot be ported at all because no
purecap Rust toolchain exists. That evidence is already committed. Use it; do not
redo it.

---

## Deliverable

One table, same shape and taxonomy as `xlang/cheri/RESULTS.md`:

```
Row | Defect (class) | Capstone: bounds-only | Capstone: revoke-on-free | verdict
```

published next to the CHERI table so the two read together. Verdict vocabulary is
the sqlite corpus's, unchanged: **BLOCKED-SYNC** (caught at the contract point) /
**BLOCKED-SWEEP** (only via a sweep) / **MISS**.

### Configs: two map, one does not

| CHERI config | Capstone analogue |
|---|---|
| spatial (bounds + tags, revocation off) | bounds-only arena |
| eager (revoke on every `free`) | `revoke_on_free_alloc.h` |
| temporal async (quarantine sweep) — *CHERI's deployed default* | **none — say so** |

Do **not** invent an async analogue, and do **not** put CHERI's async default in a
head-to-head cell with Capstone's revoke-on-free. Showing all three CHERI columns
beside Capstone's two lets the reader see directly that CHERI's *realistic* config
blocks 0 of 12 temporal defects at the contract point. That is the finding; a
matched-cell table would hide it.

Conversely Capstone's hierarchical subtree revocation (`revoke_on_free_hier_alloc.h`)
has no CHERI counterpart. Report it as a Capstone capability, not a comparison cell.

---

## Work

### 1. The capability-allocator contract  *(carried forward — do this first)*

A short spec in `xlang/shim/`, in Capstone terms rather than prose about intent:

- `alloc` — mint a capability bounded to exactly the request.
- `realloc` that **moves** — derive a capability for the new block and **revoke** the
  old; state explicitly what happens to the returned capability's bounds.
- `free` — revoke.
- And the part that actually decides the design: **what revocation does to a pointer
  already cached in a register or on the stack.** Row 4 is the worked example — its
  UAF is a write through a register-stack pointer cached across exactly the
  `realloc` that frees the old stack. If a revoked capability in a register still
  faults on use, the row is caught; if revocation only invalidates memory-resident
  copies, it is not.

**That distinction is the whole benchmark.** It is the smallest item here and the
highest-leverage, and it applies to shims exactly as it applied to real mruby.

### 2. Five files

- `xlang_shim_domain.c` — **one** generic domain; `-DROW=<key>` selects the shim.
  SQLite needed seven hand-written domains because its rows genuinely differ; these
  are template-driven, so one is enough. Resist writing fourteen.
- `mock_mruby_capstone.c` — the mock's `malloc`/`free` repointed at `rof_malloc` /
  `rof_free`. **Load-bearing**: without it the column measures nothing.
- `xlang_shim_host.c` — host side, modelled on `sqlite_row9_host`.
- `build-xlang-capstone.sh`, `run-xlang-capstone.sh` — loop the 14 rows, `smoke()`
  each, classify on the fault-cause string (`Cap mem access on revoked capability`,
  cause 25 -> BLOCKED-SYNC; clean exit -> MISS).

### 3. Sequencing

Row 10 (the corpus's template row) end-to-end first — domain, allocator, host,
verdict. The other 13 are then `#define` blocks. Doing one row end-to-end and
reporting what breaks is worth more than half-doing fourteen; if the toolchain
fights back, say so early rather than absorbing it.

---

## Traps

- **Build `-O0`.** The CHERI column does, deliberately: at `-O1`+ the compiler hoists
  the load before the `free` or elides the dangling access entirely, so the access
  the mechanism must police is never emitted. A row then "passes" for the wrong
  reason and the output looks identical to a real catch.
- **Commit predictions to a `rows.tsv` before the first run.** That is what made the
  CHERI column a test rather than a description. It is cheap and it is the project's
  standard.
- **`<stdio.h>` in `vm_stack_uaf.h`.** Bare-metal needs `mock_report` routed through
  the hostcall output path — one `#ifdef`, but the piece most likely to be fiddly.
- **Row 2 is the one to watch.** Stack-use-after-return: no allocator is involved, so
  revocation cannot apply on either side. CHERI reports MISS in all three configs.
  Predict MISS and write it down *before* running.
- **Rows 6 and 11 are spatial, not temporal.** Bounds catch them; revocation is
  irrelevant. Of the 14: 11 temporal, 2 spatial, 1 stack-UAR. State that up front —
  "14 of 14" invites the reading that all 14 test the same thing.
- **Silicon is not required.** Per the PI's rule (`tests/cheri-perf/RESULTS.md`):
  QEMU-to-QEMU for the comparison, RTL only for the Capstone absolute. The shims are
  small bare-metal C — the same size class as the beebs rungs that *do* run on
  silicon — so a silicon run is a plausible bonus, not a dependency. R-1 may still
  bite; do not promise it.

---

## Deliberately not in scope

- Real mruby, real Lua, or any real engine on either column. Dropped for speed; the
  purecap-mruby work stays committed as the porting-cost evidence.
- An async-quarantine analogue for Capstone. It does not exist; that is a result.
- Editing the paper. **Ask first** (CLAUDE.md). Results land in
  `agent-handoff/ref/fpga-silicon-measurements-for-paper.md`, which needs no
  permission and is the right default.

## Worth having if there is room

A paper-facing table of **what a bound catches that ASan does not**. The facts are
already in hand from the rows 6/11 reclassification: both read *in bounds* under some
parameters and fault under others, so under those parameters the defect is silent to
ASan *and* to a plain run. A bound on the index is the only mechanism that turns
either into a deterministic trap rather than a silent wrong answer. Close to the
strongest single argument the corpus supports, and right now it exists only as two
paragraphs inside two row READMEs.

## Ground rules

- **No personal names** in anything committed or shared — commit subjects included.
  Use roles ("the collaborator", "the board owner", "the project lead").
- Never commit the FPGA console URL or token; the placeholder is `<FPGA-CONSOLE-URL>`.
- Run `bash capstone/tests/precommit-scan.sh --msg <msgfile>` before every commit and
  push. Exit 1 = blocked.
