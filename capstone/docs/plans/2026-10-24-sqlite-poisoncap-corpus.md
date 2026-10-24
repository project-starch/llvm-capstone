# The SQLite engine corpus on PoisonCap's published memsys5 (plan, 2026-10-24)

A side comparison, not an arm of the allocator study: the study excludes PoisonCap (decided
2026-10-07, because a port of it by us would be ours to defend). What differs here is that nothing
of PoisonCap's is ported. Their published SQLite and platform run unchanged; ours is the harness
around them. Whether the result enters the study is the project lead's call once it exists.

## Question

The 33 cases of `bug-corpora/sqlite/engine-repros` are real SQLite 3.22.0 defects, all served by
memsys5 (25 temporal, 8 spatial). Our Sublet memsys5 port catches 27 of them on the virtual
profile (`results/2026-10-11-virtual-request-bounds`, repeated on compact bounds in
`results/2026-10-20-q15-repair`). PoisonCap's paper ports the same allocator in the same SQLite
release (arXiv 2605.13210, §4.5 and §5.4) but runs no security test through it: its Juliet and
MSET results use the system allocator. Which of the 33 does their memsys5 catch?

## Fairness rules

1. Their mechanism unchanged. Their `sqlite/` tree, their QEMU, CheriBSD and LLVM, all from the
   artifact pinned in `ports/ffmpeg/buffer-pool/host/cheribsd/poisoncap/platform.json`. Any change
   the harness needs is a recorded patch, and a change to their platform is a separately labelled
   arm, never folded in silently.
2. The same cases as our arms: each case's `case.c` from the corpus (not the older copies in
   `ports/sqlite/cheribsd/cases/`, 16 of which differ), the same group flags (read from
   `corpus322.sh`, as `build-virtual.py` does), the same static memsys5 heap and 64-byte minimum
   block (`repro322_common.h`), `-O0`, and the same two source fixes (`fts5-azarg-patch.py`,
   `fts3-doclist-null-patch.py`; both apply to their amalgamation).
   Lookaside, SQLite's second nested allocator, is set the same in every arm of a row. Their port
   does not cover it and ours does, so there are two rows: lookaside on (SQLite's default, the
   setting of our recorded 27/33), and lookaside off in every arm (`REPRO322_NO_LOOKASIDE`; ours as
   `build-virtual.py --sublet --no-lookaside`), which compares memsys5 against memsys5 alone.
3. The same verdict rule: CAUGHT is a fault inside the case's declared fault site, read from the
   faulting PC, not any signal anywhere. A fault elsewhere is not a catch.
4. Matched arms. `published` (their `mem5.c` as is) and `spatial` (the same tree with its one
   `#define NESTED_TEMPORAL 1` removed, their own switch) differ in that line only.
5. Controls first: a planted memsys5 use-after-free and a planted memsys5 overflow must fault in
   both arms where their design says they should, or the run is void.
6. Predictions per case and arm committed before the run.
7. Their design properties are reported, not patched: bounds on the power-of-two block; the full
   quarantine (4,096 entries) frees without revoking; revocation only past 16 MiB in use, which a
   256 KiB case heap never reaches; the revoker's return value unchecked. Counters for each per case.

## Known risks, decided before the run

- **The platform's libc never removes poison** (`bug-corpora/cpython/pymalloc-repros/platform/`):
  a program that frees and reallocates a system-malloc block faults on the new owner's first write.
  The cases allocate SQLite's memory from the static heap, but libc and the harness still use
  malloc. Run the published platform first. Only if it dies before reaching cases, add the
  documented `mrs-poison-retire.patch` as a separate arm labelled `published+libc-fix`.
- **Heap size.** Their driver gives memsys5 256 MiB; the corpus gives it 256 KiB (1 MiB and
  8 MiB for two cases). Quarantined blocks stay out of the heap until a flush, so a small heap can
  run out (SQLITE_NOMEM) before a defect site. Keep the corpus sizes for the main row (the triggers
  depend on them; case 07 engineers its OOM) and report any NOMEM as NO-READING with its cause. A
  larger-heap row is a sensitivity check, not a substitute.
- **Fault attribution on CheriBSD.** A SIGPROT gives no PC in the exit status. Take it from a core
  file, symbolized against the case binary.

## Steps

| # | Step | State |
|---|---|---|
| 1 | `ports/sqlite/poisoncap/prepare-sqlite.py WORK`: artifact by its pin, `sqlite/` extracted, `mem5.c` checked against the audit hash, both amalgamations generated | done 2026-10-24, 65 s, record below |
| 2 | Platform: `prepare.py` + `platform.sh` llvm, qemu, cheribsd, image (pinned cheribuild) | not started; needs p12/p13 |
| 3 | `build-poisoncap.py`: the corpus `case.c` files for CheriBSD purecap against each amalgamation | not written |
| 4 | Runner: one boot, controls first, per-case exit, signal, core PC and fault site | not written |
| 5 | Predictions committed, then the run, twice (determinism) | — |

Step 1, run 2026-10-24 (WORK `/tmp/capstone/sqlite-poisoncap-corpus`): archive sha256
`c5e8b07d4f70ed8c…` equal to the pin, 1,703 files under `sqlite/`, `mem5.c` `d8cb4c0e…`;
amalgamation `published` `sqlite3.c` `23a56929…`, `spatial` `05799d28…`, `sqlite3.h` `b02de0ea…`
in both. The two `sqlite3.c` differ in the removed define and in the line number of SQLite's own
`__LINE__` self-check, which moves with it.
