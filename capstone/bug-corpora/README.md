# Bug corpora

Reproduction material for defects in third-party software, one directory per program.
Deliberately outside `ports/`, which holds the ports themselves and their build.

**[INDEX.md](INDEX.md) lists every corpus, its case count, the version each one pins and
whether those defects are live in that version.** It is generated from each corpus's
`corpus.json`, so start there rather than here: the hand-written list this replaced had
memcached at two cases when it had five, and did not mention the Wireshark corpus at all.

    python3 tools/check-corpus.py --self-test    # the contract, over every corpus
    python3 tools/check-ports.py --self-test     # every port component's version pin
    python3 tools/build-index.py                 # regenerate INDEX.md and index.json

[`classification.json`](classification.json) gives every one of the 425 declared records a
normalized memory-safety axis and root/manifestation CWE classification.  It distinguishes the
418 unique active defects from two duplicate records and five withdrawn records.  The method,
acceptance rules and summary are in [CLASSIFICATION.md](CLASSIFICATION.md); regenerate it with
`tools/classify-cases-gemini.py` rather than editing model votes by hand.

[SCHEMA.md](SCHEMA.md) is the contract: what a case is, field by field, what a
`corpus.json` declares, and what a `port.json` declares. Where it and the checker
disagree, the checker is the authority.

## The other places bug material lives

INDEX.md also covers the project's own defect material:

- `../tests/fpga-repros/` — **our own** silicon and RTL defects, one self-contained folder
  per issue. No third-party defect belongs there and none of these belongs here.
- `../docs/ref/ISSUES.md` — **our own** compiler and runtime defects, as a registry rather
  than as reproduced cases. `ISSUES-ARCHIVE.md` holds the resolved ones.
- `../ports/*/security-tests/` — not bug material at all: those are this project's own
  protection fixtures and oracles, and INDEX.md counts them separately for that reason.

## How a case is laid out

Each case is its own directory recording what the defect is and where it came from. Two
schemas are in use, both stated in SCHEMA.md and both checked: most corpora carry
`case.c`, `case.json` and `PROVENANCE.md` per case, while the SQLite binding rows carry
`before.c`, `oracle`, `README.md`, `PROVENANCE.md` and a `case.json` holding the
consolidated ledger's own columns.

One sqlite row is a recorded non-reproduction rather than a case: `row15_cpython_subinterp_gil`
has a `NOTE.md` and no `before.c`, because the upstream issue turned out to concern interpreter
concurrency and not an SQLite C-API pointer lifetime. It is kept so the row is not silently
dropped from the table.

## Running a corpus

Cases are not run individually. Each corpus has its own runner, named in its `corpus.json`
and documented in its README; INDEX.md links both. For example:

    sqlite/capi-repros/run-host-asan-repros.sh
    postgres/mmgr-repros/run-host-repros.sh
    cpython/pymalloc-repros/runners/virtual/run-virtual.py

The postgres and pymalloc runners execute a control before any case — ASan must see a plain
malloc use-after-free; CheriBSD must pass its ABI and bounds probes — and exit 75 with NO
verdict if it fails, so an infrastructure failure can never be read as a measurement. The
sqlite runner has no such control and does not use that convention.
