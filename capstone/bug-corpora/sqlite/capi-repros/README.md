# SQLite C-API binding defects

Nineteen pointer-lifetime defects in **host bindings over SQLite's C API** — CPython's
`sqlite3` module, rusqlite, diesel, PHP, sqlite3-ruby, go-sqlite3, node-sqlite3, expo-sqlite
and a Datasette plugin. The engine is the real SQLite amalgamation; what is reduced is the
binding's call sequence.

**These are not defects in SQLite.** The engine's own CVEs — CVE-2018-8740, CVE-2019-9937,
CVE-2019-19880, CVE-2020-9327, CVE-2020-13435, CVE-2020-35525 — are null-dereferences inside
the engine with no pointer lifetime crossing a boundary, and
[`row12_expo_unfinalized_close/PROVENANCE.md`](row12_expo_unfinalized_close/PROVENANCE.md)
records them as out of scope. Two advisories are cited across the nineteen rows:
RUSTSEC-2021-0128 (= CVE-2021-45713 = GHSA-q89g-4vhh-mvvm) and RUSTSEC-2021-0037.

**This directory was `cve-repros/` until 2026-09-28.** The name claimed a provenance class the
corpus does not have — nineteen rows, two advisories — where every other corpus is named after
the boundary its cases cross. Older `docs/history/` notes still use the old path, deliberately:
history is append-only here.

## What a row holds

    rowN_<slug>/
        README.md        upstream link, table row, class, essence, observed fault
        before.c         the host-side sequence, against the real amalgamation
        oracle           what ASan must report, or `none`
        case.json        the same claims, machine-readable, from the ledger below
        PROVENANCE.md    the upstream artifact, and what is literal or modelled

`row15_cpython_subinterp_gil` has a `NOTE.md` and no `before.c`: the upstream issue turned out
to concern interpreter concurrency rather than a C-API pointer lifetime, and the row is kept so
it is not silently dropped from the table.

## The three documents that make it readable

- [`PROVENANCE-LEDGER.md`](PROVENANCE-LEDGER.md) — one table collapsing all nineteen rows:
  upstream artifact, defect class, provenance verdict (11 literal, 4 modelled, 4 out of scope),
  observed host fault, Capstone primitive, and RTL status. This is the source `case.json` was
  generated from, and it stays the authority for the narrative.
- [`api-classification.md`](api-classification.md) — every `sqlite3.h` entry point by boundary
  direction and lifetime obligation, read off SQLite 3.53.3.
- [`stage2-mapping.md`](stage2-mapping.md) — which row maps to which Capstone primitive.

## Running it

    bash run-host-asan-repros.sh        # the host ASan arm, all rows, against the amalgamation

The Capstone column is measured on **7 of the 19** rows, by matched-pair scripts that live with
the port rather than here: `capstone/ports/sqlite/run-sqlite-row{2,3,3-b2,5,7,9,11,14}.sh`. Each
row's `case.json` names the ones that apply to it, and `corpus.json` lists them all. Unlike the
pymalloc and memory-context runners, `run-host-asan-repros.sh` has no control run before the
cases and does not use the exit-75 convention.

`live_in_pin` is `null` on every row, with the reason in each `case.json`: the defect is in the
binding, this corpus pins no binding release, and the host arm builds the 3.53.3 amalgamation.

## The virtual-malloc arm (and vp2-malloc on the board)

**The project lead's decisions, relayed by the runtime lane on 2026-10-22** ("capi и xlang как рекомендовано"):
- **(a) Version:** SQLite 3.53.3, the host arm's amalgamation, through ports/sqlite's six rewrites plus
  `-DHAVE_MALLOC_H=1 -DHAVE_MALLOC_USABLE_SIZE=1`.
- **(b) Controls:** the generic malloc controls suffice, with no per-row FIXED twin.
- **(c) Verdicts by oracle:**
  - heap-use-after-free, cause 25: CAUGHT;
  - bad-free, refused by the allocator (cause 2 in `cap_malloc_invalid_free`): CAUGHT;
  - null-deref (cause 24 or 12): its own class, never counted as a temporal catch;
  - none, exit 0: correct, no false alarm.

**How it runs** (`shared/run-virtual.py`, whose docstring is the contract):
- **The build:** the runner builds everything into its raw directory.
  - SQLite: `fetch-sqlite.sh`'s 3.53.3 amalgamation through the rewrite block of `build-sqlite-capstone.sh`. The block
    is read out of that script at run time, and a script without its six rewrites and six asserts stops the build.
    Compiled `-O1 -g -DSQLITE_THREADSAFE=0 -DSQLITE_OMIT_LOAD_EXTENSION` (the host arm's flags) plus the two
    usable-size defines.
  - Each row: `before.c` at `-O0 -g`, linked with `shared/virtual-main.c` through `-Wl,--wrap=main`.
- **The marker:** the wrapper prints `CAPI BEGIN` at main's entry and `CAPI RETURNED <rc>`.
  - BEGIN is entry, not the access.
  - A row that exits non-zero without a fault reads inconclusive, never MISSED.
- **The sites:** a row's `fault_sites` is host ASan's frame #0 for its before.c, read from
  `run-host-asan-repros.sh` on 2026-10-22 (18/18 oracles matched):
  - `main` for rows 1, 3-5 and 7-10;
  - `read_context` for row 2, `stale_udf` for row 6, `authorize` for row 16;
  - `cap_malloc_invalid_free` for the bad-free row 11, by decision (c).
  For the rows whose site is `main`, a fault anywhere in main is attributed. Main is the row's own short program, and
  this is the coarsest attribution in the corpus.
- **Not scored as temporal catches:** null-deref and none rows are recorded as NO-READING:out-of-denominator. The
  reading is in `notes` and in inputs.json's `classes`, where a fault on a none row is written FALSE ALARM. Row 15 has
  no program.
- **Lookaside** stays at SQLite's default, as on the host arm. A stale pointer into a lookaside slot sits inside a live
  malloc block, which this arm cannot see. The arm does not claim those.

### Pre-registration (2026-10-22, before its first run): qkit7-batch2, the batch SDK t3-freeze-nb-nomovc-cw-r60

The runner's build was run on its own, with no boot, before this was written:
- 18 row images build;
- the rewritten amalgamation equals, byte for byte (3b2007721ca3...), the one the exploratory run used
  (capstone/tests/corpus-cell-probes/README.md);
- the rewrite-block guard refuses a script with one rewrite removed;
- the observation logic was exercised on synthetic runs: catch at a site, catch elsewhere (unattributed), completed
  (MISSED), exit 1 (inconclusive), no BEGIN (setup-fault), the null-deref class, a clean none row, and a none-row false
  alarm.

**Predicted:**

| row | oracle | verdict | reading |
|---|---|---|---|
| 1, 3, 4, 5, 7, 8, 9, 10 | heap-use-after-free | CAUGHT | cause 25 in `main` |
| 2 / 6 / 16 | heap-use-after-free | CAUGHT | cause 25 in `read_context` / `stale_udf` / `authorize` |
| 11 | bad-free | CAUGHT | cause 2 in `cap_malloc_invalid_free` |
| 12, 14 | null-deref | out-of-denominator, class null-deref | cause 24 in `main` |
| 13 | null-deref | out-of-denominator, class null-deref | cause 12 at an unsymbolized pc (a call through a null function pointer; ASan's pc 0) |
| 17, 18, 19 | none | out-of-denominator, class none | correct, no false alarm |
| 15 | none | out-of-denominator | no program |

- Controls: bounds-malloc fault, uaf-malloc fault.
- The tally: CAUGHT 12, NO-READING:out-of-denominator 7.
- A row reading otherwise is reported as read, not re-sited.

### Result, run 1 (2026-10-22 18:30-18:31, qkit7-batch2 on QEMU domain A, at 9ec12779a18f): 18 of 19 rows as predicted

The bundle is `results/2026-10-22-virtual/virtual-malloc`, and the verdicts below are derived from it.
- **Controls:** both fault.
- **As predicted:**
  - 11 heap-use-after-free rows CAUGHT at their declared site with cause 25 (`main`; `read_context`, `stale_udf`,
    `authorize`);
  - null-deref rows 12 and 14 cause 24 in `main`, row 13 cause 12 at an unsymbolized pc;
  - none rows 17-19 correct, with no false alarm; row 15 has no program.
- **NOT as predicted: row 11** reads NO-READING:unattributed. The cause, 2, is as predicted, and so is the pc: the
  first instruction of `cap_malloc_invalid_free`. The miss is the symbol name.
  - `cap_malloc_invalid_free` is a zero-size global label inside the static function `invalid_free`
    (runtime/virtual/heap.c:350-352, whose whole body is the label and an `unimp`).
  - tools/verdicts.py's `Symbols` attributes a pc to the enclosing sized function, so this pc is named
    `invalid_free`. The exploratory run's symbolizer took the nearest label instead.
  - So the declared site was the right instruction under a name this judge never reports.

### The site alias for row 11, and run 2 (pre-registered 2026-10-22, after run 1, before run 2)

- **The change, made after run 1 read it and said so here:** row 11's `fault_sites` gains `invalid_free`, the name the
  judge gives the same instruction. No other row changes, and neither the runner nor the build does.
- **Run 2:** the same invocation, kit and SDK.
- **Prediction:** every row reads as in run 1, except row 11, which reads CAUGHT (cause 2 in `invalid_free`, by
  function).
- **Run 1's bundle stays as the record of the pre-registered reading.** `verdict_bundles` moves to run 2's only if run
  2 reads as predicted.

**Result, run 2** (2026-10-22 18:33-18:34, at b8f4f78fa9c7): **as predicted.** Every row reads as in run 1, except row
11, which now reads CAUGHT (cause 2 at `invalid_free`, by function).
- The tally: CAUGHT 12, NO-READING:out-of-denominator 7.
- Classes: null-deref 3 (rows 12, 14 cause 24 in `main`; row 13 cause 12), none 3 correct and 1 without a program.
- `verdict_bundles` now names `results/2026-10-22-virtual-run2/virtual-malloc`. Run 1's bundle stays beside it.

**For the board (vp2-malloc):** the same invocation with `--board-kit <kit>` in place of `--virtual-kit`. The bundle
records `vp2-malloc` under the same configuration. Its prediction is run 2's reading, row for row.

### Path-independent images, and run 3 (pre-registered 2026-10-22, before run 3)

- **The defect, found before the board ran anything:** run 1 and run 2 built each row with `-g` and no prefix map.
  - So an image recorded its raw directory and the compile's cwd. The same row built in two raw directories hashed
    a3ded783… and f865b93f….
  - The board rebuilds the rows in its own directories, so its images could never hash-equal the reference's. An
    image check against runs 1 and 2 cannot pass.
- **The fix:**
  - every compile, controls included, takes `-ffile-prefix-map=<raw>/bin=/capi-build` and `-ffile-prefix-map=<repo>=/repo`;
  - every compile runs with the build directory as its cwd.
  - Checked before this was written: rows 1 and 11, `sqlite3.o` and `controls.dom` hash identically across two raw
    directories, a cwd of /tmp, and a second repo root.
- **Run 3:** the same invocation as runs 1 and 2.
  - Predicted: every row reads as in run 2. Only the image hashes change, and from now on they are the same for any
    checkout and raw directory.
  - If run 3 holds, `verdict_bundles` moves to it, and it is the reference the board's image check uses.

**Result, run 3** (2026-10-22 18:45-18:46, at fd732cc48611): **as predicted.**
- Every row's verdict, reason, cause and symbol equals run 2's; all 18 image hashes changed.
- The hashes are reproducible: rows 1 and 11 equal the independent builds from two other raw directories and a second
  repo root (1328034ed2872e67, c027fedd724705fe).
- `verdict_bundles` now names `results/2026-10-22-virtual-run3/virtual-malloc`, the board's reference.

**What the reference image hashes still depend on:** the SDK's own include and resource directories are not prefix-mapped, so a board rebuild hash-equals the reference only from the same SDK path, `/mnt/hdd/<user>/capstone/t3-freeze-nb-nomovc-cw-r60/sdk` (sdk.json f5145cbb), with the same toolchain. The same checkout or raw directory is not required.
