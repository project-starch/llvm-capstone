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
