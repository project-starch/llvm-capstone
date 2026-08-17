# MicroPython temporal-safety corpus

Tracking table for temporal memory-safety defects on MicroPython's own allocator,
the mark-and-sweep collector in `py/gc.c` plus the `m_malloc` layer above it.
Thirty cases in `temporal-allocator-corpus.csv`.

MicroPython is interesting here precisely because it does not use the system
allocator. It carves one heap at startup and manages it itself, so every case
below is a lifetime bug on memory the collector believes it owns. That is the
configuration a capability machine is supposed to change.

## What counts, and what does not

In scope: use-after-free, double free, dangling views and buffers, premature
collection, lifetime and shutdown ordering, re-entrancy that invalidates a
pointer already in flight, and allocator invariants that fail on their own input.

Deliberately excluded: purely spatial defects. Three of the six MicroPython CVEs
in NVD are heap buffer overflows (`CVE-2023-7158` in `slice_indices`,
`CVE-2024-8946` in `mp_vfs_umount`, `CVE-2024-8948` in `mpz_as_bytes`), as are
issues 12587 and 13006. They are real, they are just a different question, and
bounds checking already answers it.

Worth stating plainly: **NVD lists only six MicroPython CVEs in total**, so a
corpus of thirty cannot be CVEs alone. Three CVEs are in the table and the other
twenty-seven are issues from the upstream tracker, each one a dated, linked,
reproducible report. Two of the three CVEs have their originating issue in the
table as well (`CVE-2023-7152` with issue 12887, `CVE-2024-8947` with issue
13283), which is useful: those pairs give both a repro and an official patch.

## Provenance

Every factual column was copied from an API response, not typed:

- issue rows: GitHub GraphQL, one query over all thirty-four candidates
- CVE rows: NVD REST API, `keywordSearch=micropython`
- the builder is `build-corpus.py`; rerun it to refresh state and titles

`verify-corpus.py` re-reads the CSV and compares every title and state against
the stored API responses. It was negative-tested by corrupting two rows, and it
reported both.

## Columns

| column | meaning |
|---|---|
| `id` | stable local identifier, `MPY-T01` to `MPY-T30` |
| `source`, `ref`, `url` | where the case comes from, and the link |
| `title`, `state`, `first_seen` | copied verbatim from the source API |
| `class` | defect shape: `uaf`, `dangling-view`, `dangling-buffer`, `premature-free`, `lifetime-order`, `reentrancy`, `race-uaf`, `alloc-invariant` |
| `cwe` | CWE where the source assigns one, otherwise the closest fit |
| `component` | file or function, from the report |
| `scope` | `gc-core` (inside the collector), `gc-managed` (on collector memory), `port-heap` (port allocator interaction) |
| `trigger` | one line on what has to happen for it to fire |
| `capstone_hypothesis` | **a prediction, not a measurement** |
| `repro_status` | `none` for all thirty right now |
| `notes` | provenance and cross-references |

## The one column to be careful with

`capstone_hypothesis` currently reads `trapped` for twenty rows, `unclear` for
six and `not-trapped` for four. None of that has been run on hardware or in
simulation. It is a reading of each report, and reading a bug report is exactly
the step that has produced wrong conclusions on this project before. The four
`not-trapped` entries are the honest ones and the most useful: a deadlock in a
finaliser, a data race, a leak, and over-retention are not spatial or temporal
reach violations, so a capability machine has no reason to catch them, and a
corpus where every row conveniently confirms the thesis would not be worth
having.

Treat the column as a work list. It becomes evidence only when `repro_status`
moves off `none`.

## Next step

Nothing in this folder has been reproduced yet. The obvious first targets are
issues 18168 and 18171: both are open, both are pure Python triggers needing no
port hardware, and 18171 turns a stale view into a heap-corrupting write, which
is the sharpest primitive in the table.
