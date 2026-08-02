# Plan: CHERI security baseline for the xlang corpus

**Status: PROPOSED (2026-07-31) — for discussion, not yet started.**
Owner: unassigned. No board, no Capstone compiler, no rootfs lock — this is
CHERI-stack-only work, parallel-friendly with everything else.

## 1. Goal

Repeat the sqlite-corpus CHERI comparison (agentB-015,
`capstone/tests/cheri-baseline/`) for the **xlang cross-language FFI corpus**
(`xlang/`, 15 rows): for each row, measure empirically whether CHERI-RISC-V
**purecap** catches the defect and *when*, under the same three revocation
configs, and classify with the same taxonomy. The paper storyline is unchanged:
*not blocked under the CHERI baseline (at the contract point), blocked on our
system.* A refutation is a valid result and is reported as such.

This is **measurement + classification of the CHERI baseline, not our system**.
The Capstone ("our system") column is out of scope here — see §8.

## 2. Prior art being copied

`capstone/tests/cheri-baseline/{README,RESULTS}.md` define the method:

- Each row's minimal vulnerable shim compiled **verbatim, `-O0`, purecap**
  (`-O0` because the dangling access is UB and `-O1+` elides it).
- One CheriBSD purecap boot under CHERI-QEMU; each row run under three configs
  toggled by sysctl and *confirmed* per-process with `malloc_revoke_enabled()`:
  `spatial` (bounds+tags only), `temporal` (async quarantine-and-sweep — the
  realistic default), `eager` (revoke on every `free` — expensive upper bound).
- Verdicts: **BLOCKED-SYNC** (faults under `spatial`), **BLOCKED-SWEEP**
  (`async`/`eager`: caught only by revocation, not at the contract point),
  **MISS** (survives all).
- **Predictions recorded per row in `rows.tsv` BEFORE any run.**
- Sanity probes that must hold or the row data is invalid.

Sqlite-corpus headline (for symmetry): spatial blocked 3/15 (null-derefs only),
the realistic async default blocked 0/10 UAFs at the contract point, and the
reuse-not-free variant (`3r`) was missed by every config.

## 3. Deliverable

`xlang/cheri/` mirroring the sqlite layout:

| File | Role |
|---|---|
| `rows.tsv` | xlang row → shim, oracle, defect class, predicted verdict per config |
| `mock-mruby/` | minimal mruby-lifecycle harness the shims link against (§4) |
| `shims/rowN.c` | one C shim per in-scope row, distilled from `xlang/<row>/` |
| `compile-purecap.sh` | builds the mock + one purecap ELF per row + sanity probes |
| `run-in-guest.sh` / drivers | reused from `cheri-baseline/` (see §7) |
| `RESULTS.md` | config reality + per-row verdict table + what the data says |

Host-side drivers (`cheri-run.py`, `oneshot.py`, `classify.py`,
`cheri_status.c`) are corpus-agnostic and are reused, not rewritten.

`rows.tsv` is also the seed of the corpus's consolidated ledger — the analog of
`capstone/benchmarks/sqlite/cve-repros/PROVENANCE-LEDGER.md`, the sqlite
corpus's per-CVE analysis report (upstream artifact, defect class, verdicts,
primitive family). The xlang table should reach the paper the same way.

## 4. The vehicle — mock lifecycle harness, the sqlite approach

Real mruby will not be the purecap vehicle, for the same reason real SQLite was
not: porting an interpreter to run purecap is a project in itself (mruby's
default word-boxing packs pointers into 64-bit `mrb_value` words, which cannot
carry a 128-bit capability) and is orthogonal to this task's question. The
sqlite precedent applies directly: the CHERI verdict for each row depends only
on the defect's **memory-lifecycle events** — what is allocated, freed or
recycled, and dereferenced, in what order — not on the VM around them.

So, exactly as `mock-sqlite/` did: a minimal **`mock-mruby/` lifecycle
harness** reproduces those events (object/env/stack allocation, GC-arena
recycling, C-heap `free`/`realloc` of VM stacks and buffers, callback entry),
and each row gets a small C shim against it, built `-O0`. One difference from
sqlite is unavoidable and must be handled, not hidden: the sqlite corpus rows
already *were* C shims, compiled verbatim; the xlang rows are interpreter +
trigger script, so each shim is **written by us**, distilled from the phase-1
analysis (`xlang/<row>/target.md` + `asan.txt`, where the mechanism work is
already done). Three fidelity rules keep the shims honest:

1. **Native validation gate.** Every shim must reproduce the same ASan defect
   class **natively** (host, `-fsanitize=address`) before its purecap verdict
   counts. A shim that doesn't reproduce the row's defect on the host measures
   nothing. This is the analog of the sqlite corpus's host oracle.
2. **The allocator route is derived, not chosen.** Whether a row's dangling
   memory reaches `free()` or is GC-arena-recycled is a property of real mruby
   (§6); the shim reproduces what the source/trace shows. GC-arena rows are
   shimmed **reuse-not-free** (the sqlite `3r` pattern) — modeling every death
   as `free()` would manufacture CHERI catches.
3. **Real allocation geometry for the spatial rows.** CHERI bounds are
   `malloc`-granular; mruby's VM stack is one big allocation and its GC packs
   many objects per arena page. An overflow that stays inside the allocation is
   invisible to CHERI in every config. Shims reproduce the real allocation
   sizes and overflow distances, taken from the row's ASan report.
4. **As near to the real software as the mock allows** (review guidance,
   2026-07-31). Lift the actual mruby structures and code fragments — real
   struct layouts, the real loop under test — into the shim rather than
   abstracting to a generic lifecycle model; mock only what must be mocked.

`RESULTS.md` states the residual caveat the same way the sqlite run stated its
mock caveat: the shims are a lifecycle model, and where the model makes frees
visible to the allocator the CHERI-catch numbers are an **upper bound**.

## 5. The Rust rows (1–3)

There is no usable purecap Rust toolchain for CHERI-RISC-V. Precedent already
exists **inside the xlang corpus**: rows 1–3 were excluded from the RISC-V QEMU
leg on exactly these grounds (`xlang/README.md` §"On the QEMU leg").

- Default: **exclude rows 1–3 from the CHERI column** with the same one-line
  justification, leaving **11 measured rows** (4–6 and 8–15; row 7 cannot
  appear, §6 — the operative count is 11, not conditional).
- Optional salvage for **row 3 only**: its stale dereference executes inside
  **libpulse, which is C** — its C-side lifecycle fits the §4 shim approach
  with no Rust involved.

## 6. Allocator visibility (the expected headline)

CHERI's temporal safety is **`free()`-triggered**: quarantine and sweep operate
on allocations returned to the system allocator. The sqlite work already showed
the weak spot (lookaside hides frees; the reuse-not-free `3r` row is missed by
every config).

**Step 0 is DONE (2026-07-31) and it REFUTES this section's original
hypothesis.** The plan assumed mruby's GC arena would make CHERI structurally
blind on most rows. The ASan evidence says otherwise
(`xlang/cheri/rows.tsv`): **all 9 temporal rows reach
the system allocator**, so revocation *can* see every one of them. The actual
distribution is

- **6 rows** (4, 5, 8, 10, 13, 15) — `realloc` of the **VM register stack**;
  the old block is genuinely freed (the triggers deliberately fragment the
  heap to force an out-of-place move, which is *why* the bug fires),
- **2 rows** (9, 14) — GC-driven, but the sweep path ends in a real `free()`
  (`obj_free`/`mrb_irep_free` → `mrb_free`, and `incremental_sweep_phase`
  releasing a 49,200-byte heap page),
- **1 row** (12) — an explicit `mrb_free` in gem code.

Two consequences, both to state in `RESULTS.md`:

1. **No `3r` analog exists in this corpus.** The sqlite table's cleanest
   "CHERI cannot" result — stale-but-allocated, missed by every config — has
   **no counterpart here**. Do not promise the paper one from xlang.
2. **The expected shape is the sqlite shape minus that row:** `eager` catches
   the temporal class, the realistic `async` default catches none of it at the
   contract point (the reproducers free and re-use promptly, with no sweep in
   between), and the two spatial rows fault under plain `spatial`.

Also note the corpus is **less diverse than 11 rows suggests** in
CHERI-verdict terms: 6 of 9 temporal rows are the same defect shape (a raw
pointer into the VM register stack held across a callback that reallocs it).
Report the shape count, not just the row count.

Two framing points, both **already settled** on 2026-07-27
(`history/27-07-2026_16-03-02_xlang-phase1-followups-items-1-3-closed-phase2-seam-proven.md`;
the stale "three decisions pending" section in `state/current-next-step.md`
predates that note):

- **Rows 6 and 11 are reclassified spatial** (heap-buffer-overflows), stated
  paper-facing in their READMEs; NVD itself assigns CWE-119 to CVE-2026-1979,
  so the spatial reading is on the record. Predicted **BLOCKED-SYNC** —
  genuine CHERI bounds catches, not null-deref technicalities as in the
  sqlite table. They favor CHERI's column *and* are precisely the rows
  revocation does not address. **The prediction holds only in the faulting
  parameter regime**: both rows read in bounds under some trigger parameters
  (row 6 at shallow recursion, row 11 under ~80 outer locals — see their
  "Tuning the trigger" sections), and since CHERI bounds are
  `malloc`-granular a shim landing in the in-bounds regime produces a
  spurious MISS. The faulting parameters are pinned in `rows.tsv` next to
  the prediction and reproduced by the shim's geometry, or the row measures
  the trigger, not CHERI.
- **Row 7 does not exist as specified** — independently settled by the NVD
  record (#6701 is row 6's bug; no bigint, no `mrb_bint_reduce`). It cannot
  appear in the measured table.

## 7. Infrastructure notes

- Reuse from `cheri-baseline/`: `cheri-run.py`, `oneshot.py`, `classify.py`,
  `run-in-guest.sh` (config sysctls + structured result lines),
  `cheri_status.c`. The same CheriBSD image/overlay flow applies; adding a
  second overlay directory does not disturb the sqlite rows.
- The `cheri-baseline` scripts *default* their workspace paths to one
  contributor's home directory, but every path is already env-overridable
  (`CHERIBUILD`, `OVERLAY`, `OUT`, `WORK`, `LOCAL`) — set those for the
  machine at hand; nothing blocks on it. A tidier de-hardcoding commit
  (`CHERI_ROOT`/`CHERI_WS` defaults) exists **locally only**, unpushed, on
  branch `capstone-bootstrap-cheri-real-sqlite`; land it separately if
  wanted — do not treat it as a prerequisite.
- Toolchain/vehicle identical to the sqlite run: cheribuild-built CHERI-LLVM,
  CheriBSD purecap (CHERI_CAPREVOKE kernel), `qemu-system-riscv64cheri`.
  Catch/no-catch only — QEMU is the right vehicle; no performance claims.

## 8. Steps

0. **Per-row lifecycle audit → `rows.tsv` with predictions.** Read each
   `xlang/<row>/target.md` + `asan.txt`; record defect class, the
   C-heap-vs-GC-arena bit (§6), the allocation geometry **and pinned faulting
   trigger parameters** for the spatial rows (§6), and the predicted verdict
   per config. Board-free, CHERI-free; can start immediately.
   **Gate: predictions committed before any CHERI run.**
1. **Build `mock-mruby/` + shims (§4).** One shim per in-scope row plus the
   `sanity_mock`-equivalent clean probe. **Gate: every shim passes the native
   ASan validation before it is compiled purecap.**
2. **Run + classify.** One boot, three configs, `classify.py` table →
   `RESULTS.md` with the same taxonomy and a "what the data says" section.
3. **(Separate, later) the Capstone column.** Not part of this plan, and it
   already has a designed path: **Phase 2's allocator seam**
   (`plans/xlang-phase2-seam-TODO.md`; `xlang/shim/` routes every mruby VM
   allocation through a three-function seam via `mrb_open_allocf`, proven
   byte-identical to stock on rows 4 and 11). The CHERI table produced here
   is the baseline that column gets compared against.

   **Scoping warning before step 3 (review finding, 2026-07-31, confirmed by
   R-12):** the seam measures `alloc=1668 free=1883` on a *trivial* mruby
   script, while the silicon rev-node pool has a 10-bit id — allocation
   #1025 **silently wraps onto live nodes** (no fault; `ref/ISSUES.md`
   R-12), and SQLite already had to merge string literals (1,059 → 179
   carves) just to fit its *entry glue*. A capability allocator carving one
   node per VM allocation therefore cannot run even mruby hello-world on
   silicon as the hardware stands. Before scoping the Capstone column,
   decide: **QEMU-only column**, or a **carve-reusing pooled allocator** at
   the seam (a small fixed set of carves recycled across allocations).

## 9. Open questions for discussion

1. Rows 1–3: exclude (precedent) or shim row 3's C side (§5)?
2. Is the empirical Capstone column (§8 step 3, via the Phase-2 seam) wanted
   for the same paper table, or do the `boundary.md` statements suffice for
   this submission? If empirical: QEMU-only or silicon (see the step-3
   scoping warning — as conceived it is silicon-infeasible without a pooled
   carve allocator)?
