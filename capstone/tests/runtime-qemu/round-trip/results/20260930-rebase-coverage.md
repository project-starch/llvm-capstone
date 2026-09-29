# Recovery after the intcap rebase

PR #94 is rebased onto dev `7b73c7a88a1e`. The Sema resolution keeps the
`__intcap` conversion exemption. The recovery pass and its safety regressions
are unchanged. [Result lines and hashes](20260930-rebase-coverage.json) identify
the tested compiler, source files and guest images.

## Measured reach

| Corpus | Integer-to-pointer occurrences, off/on | Recovered | Changed assembly units |
|---|---:|---:|---:|
| musl 1.2.5, 1355 compilable units, -O1 | 1102 / 1102 | 0 | 0 / 1355 |
| SQLite 3.53.3, deployed port, -O0 | 46 / 46 | 0 | 0 / 1 |
| SQLite 3.53.3, deployed port, -O2 | 64 / 64 | 0 | 0 / 1 |
| IR positive/negative fixture, -O0 | 54 / 30 | 24 | 1 / 1 |
| Same fixture, -O2 | 54 / 30 | 24 | 1 / 1 |

Each C file was compiled once to optimized IR. The same IR was passed through
llc with recovery enabled and disabled, retaining the IR immediately after the
pass and the complete final assembly. The denominator counts all `inttoptr`
occurrences, including conversions that are not recoverable round trips.
All 1357 real-source assembly pairs are byte-identical. Six of the 1361 musl
sources fail the existing mallocng pointer-size static assertion before code
generation; no unit fails in the backend. The survey keeps the musl positive
and negative build controls, and the recovery fixture changes in both modes.

These corpora show **no additional compatibility benefit** from the conservative
pass. Its known-object/explicit-DELIN mechanism works in the targeted fixtures
and the QEMU round-trip program; broad recovery in existing ports is unproven.
The ordinary argument version of `align_up` declines. The alloca version is a
positive control, and cannot support an argument-recovery claim.

An explicit DELIN result already provides the NONLIN proof, with validity
checked separately. A future parameter attribute would need an enforced
contract on every C and assembly entry path, preservation through optimization
and LTO, and tests for violations. A declaration alone would reintroduce the
unsafe argument assumption. No new exemption is added here.

## Validation

All 28 Clang Capstone tests and all 128 LLVM Capstone CodeGen/MC tests pass.
The LLVM suite uses Myers through a temporary Git wrapper. An initial manifest
failure came from the host's histogram setting; lit does not preserve
`GIT_CONFIG_*`. The manifest checker independently matches all 110 shared files.

QEMU round-trip passes 12/12 at both -O0 and -O2, including the runtime's atexit
handler. Its disabled-pass control faults before the first successful round
trip. Exit-hook returns 7 and 42 in its positive arms; the old-runtime and
unmodified-musl controls fault as required. The capability-safe atexit override
remains necessary. This is no new silicon result or full application gate.

## Reproduction

Build the rebased compiler and source `capstone/tests/capstone-test-env.sh`.
Prepare musl with the existing `prepare-musl-capstone.sh` and SQLite with the
existing `build-sqlite-capstone.sh`, using private output/cache directories.
The latter produces `sqlite3-capstone.c` beside its SQLite header inputs; if
using another output location, place the pinned `sqlite3.h` beside that file.
Then run from the repository root:

```sh
python3 capstone/tests/provenance-recovery-survey.py \
  --musl-dir "$MUSL_SRC_DIR" \
  --sqlite-source "$SQLITE_OUT/sqlite3-capstone.c" \
  --out "$CAPSTONE_TMP_ROOT/recovery-survey" --jobs 4
```

The survey imports musl's flag/file selection and reads SQLite's deployed
defines from the existing build recipe. It retains diagnostics, IR, assembly,
per-unit source hashes and `summary.json` in its output directory. It verifies
that the compiler binaries did not change during the run.

For the manifest check, propagate the intended Git setting explicitly:

```sh
GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=diff.algorithm GIT_CONFIG_VALUE_0=myers \
  python3 llvm/utils/capstone-shared-drift.py --repo .
```

For lit, put a temporary `git` wrapper that invokes the real Git with
`-c diff.algorithm=myers` on lit's `--path`; this preserves the user's Git
configuration. Run the Capstone CodeGen/MC suites and Clang's Capstone tests.
Run the round-trip probe under `capstone_with_qemu_lock`; exit-hook takes the
lock itself. Set `PYTHON` to an environment with `pexpect`.
