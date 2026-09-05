# Compiler reproducers — index

One folder per compiler defect, each with its own README and a runnable `run.sh` where the
defect is compile-only (the scripts take the compiler from `llvm/cmake-build-debug/bin`). The
registry entry (`docs/ref/ISSUES.md`) carries the status; this index only says which folder is
which and what the 2026-09-05 sweep (`docs/plans/bug-sweep-2026-09.md`) found for each.

| folder | ID | what it reproduces | sweep 2026-09-05 |
|---|---|---|---|
| `C19-capinit-block-split-oob` | C-19 | the cap-init block split faulted OOB on silicon | RESOLVED 2026-08-26; the compiler half is pinned by lit, the simulation half re-measured PASS at 5097eb166 |
| `C20-cttz-i32-crashes-legalizer` | C-20 (= C-24) | `__builtin_ctz` on i32 crashed the legalizer | FIXED 2026-09-04 (`c20-cttz.ll`); `run.sh` PRESENT on the pre-c128 and 08-19 builds, ABSENT now |
| `C21-i128-select-of-constants` | C-21 | a select of two `__int128` constants could not be selected | GONE with the c128 carrier; `run.sh` PRESENT on the pre-c128 and 08-19 builds, ABSENT now |
| `C22-i128-or-sext-folds-to-constant` | C-22 | `c ? (__int128)-1 : k` dropped the condition | GONE with the c128 carrier; `run.sh` PRESENT on the pre-c128 build, ABSENT on 08-19 and now |
| `C23-i128-high-half-silently-dropped` | C-23 | an `__int128` computed on its low half only | NOT REPRODUCIBLE on any recorded compiler (the parent of its filing commit is clean, positive control fires); current compiler clean |
| `H01-beebs-matmult-float-host-headers` | H-01 | BEEBS matmult-float picked up host glibc headers | FIXED; `src/repro.sh` clean |

A folder stays after its defect is fixed: the `run.sh` is the instrument that shows the fix, and
a sweep re-runs it on the compiler the bug was filed against before calling anything GONE.
