# What is open in the three-arm study

Generated against this tree, 2026-10-09, from `protection.json`. Every number
here is counted, not carried over: the figures this file used to hold were
taken on a tree with 162 declared cases and are not the ones that apply now.

**230 live cases.** Where a cell is open, the reason is in `protection.json`
and the table marks it; this file groups those reasons so the cost of closing
them can be read in one place.

| arm | caught | coincides | missed | of those explained | not run |
|---|---:|---:|---:|---:|---:|
| `cheribsd` | 37 | 0 | 91 | 0 | 102 |
| `capstone-sysalloc` | 62 | 0 | 74 | 0 | 94 |
| `capstone-sublet` | 116 | 51 | 12 | 0 | 51 |

`coincides` is not a weaker catch: the cell was measured by the run that
measured the arm beside it, because the object came straight from the system
allocator and there is no nested allocator for the two arms to differ by.

## The four steps that are prepared and not taken

Nine declaration commits deliberately held these back, each because it is a
CLAIM rather than a reading, and a claim belongs in a commit that argues for it.

| step | what it asserts | cells it moves |
|---|---|---:|
| dispositions | the measured reason each silence is silent -- including the one group whose silence was quarantine MEMBERSHIP, which the rule of 2026-10-08 scores as a catch | 49 explained, 26 re-scored |
| parking | a case no arm in this study can be asked about leaves every denominator | 17 cases |
| deletion | a case this study cannot be asked about at all, and the renumbering that follows | 26 cases |
| measurement | the cells no run has filled | 104 |

## Where the open cells are

**247 cells have no measurement.** Grouped by arm and group:

| group | arm | cells |
|---|---|---:|
| `sqlite/engine-repros` | `cheribsd` | 28 |
| `ffmpeg/plain-heap-repros` | `cheribsd` | 21 |
| `ffmpeg/plain-temporal-repros` | `cheribsd` | 13 |
| `wireshark/plain-heap-repros` | `cheribsd` | 10 |
| `wireshark/plain-temporal-repros` | `cheribsd` | 10 |
| `memcached/plain-heap-repros` | `cheribsd` | 7 |
| `postgres/sql-repros` | `cheribsd` | 7 |
| `memcached/plain-temporal-repros` | `cheribsd` | 3 |
| `mruby/release-differential` | `cheribsd` | 2 |
| `ffmpeg/plane-repros` | `cheribsd` | 1 |
| `ffmpeg/plain-heap-repros` | `capstone-sysalloc` | 21 |
| `ffmpeg/plain-temporal-repros` | `capstone-sysalloc` | 13 |
| `sqlite/engine-repros` | `capstone-sysalloc` | 11 |
| `wireshark/plain-heap-repros` | `capstone-sysalloc` | 10 |
| `wireshark/plain-temporal-repros` | `capstone-sysalloc` | 10 |
| `postgres/sql-repros` | `capstone-sysalloc` | 8 |
| `memcached/plain-heap-repros` | `capstone-sysalloc` | 7 |
| `memcached/allocator-repros` | `capstone-sysalloc` | 6 |
| `httpd/bucket-repros` | `capstone-sysalloc` | 4 |
| `memcached/plain-temporal-repros` | `capstone-sysalloc` | 3 |
| `mruby/release-differential` | `capstone-sysalloc` | 1 |
| `ffmpeg/plain-temporal-repros` | `capstone-sublet` | 13 |
| `sqlite/engine-repros` | `capstone-sublet` | 11 |
| `wireshark/plain-temporal-repros` | `capstone-sublet` | 10 |
| `postgres/sql-repros` | `capstone-sublet` | 9 |
| `wireshark/wmem-repros` | `capstone-sublet` | 4 |
| `memcached/plain-temporal-repros` | `capstone-sublet` | 3 |
| `ffmpeg/plane-repros` | `capstone-sublet` | 1 |

**177 misses carry no disposition.** The mechanism ran and said
nothing, and no run has established why. The largest blocks:

| group | arm | cells |
|---|---|---:|
| `wireshark/wmem-repros` | `cheribsd` | 22 |
| `wireshark/wmem-repros` | `capstone-sysalloc` | 22 |
| `cpython/pymalloc-repros` | `cheribsd` | 20 |
| `cpython/pymalloc-repros` | `capstone-sysalloc` | 20 |
| `ffmpeg/subobject-repros` | `cheribsd` | 10 |
| `ffmpeg/subobject-repros` | `capstone-sysalloc` | 10 |
| `ffmpeg/subobject-repros` | `capstone-sublet` | 10 |
| `httpd/bucket-repros` | `cheribsd` | 8 |
| `memcached/allocator-repros` | `cheribsd` | 8 |
| `mruby/release-differential` | `cheribsd` | 8 |
| `postgres/mmgr-repros` | `cheribsd` | 5 |
| `postgres/mmgr-repros` | `capstone-sysalloc` | 5 |
| `ffmpeg/pool-repros` | `cheribsd` | 4 |
| `ffmpeg/pool-repros` | `capstone-sysalloc` | 4 |

## Nineteen cases the table cannot see

`sqlite/capi-repros` declares nineteen cases and produces no row. A case is found
by its own directory, and that group keeps its cases as rows in a table
(`case_schema: sqlite-row`); reading one needs a reader `protection-matrix.py`
does not have. `xlang-row` is exempt by declaration and says so -- this was not
exempt, it was missed, and it was found by counting the tree against the table
rather than by reading either. PROTECTION.md now counts the shortfall in a
section of its own so the totals cannot be mistaken for the whole tree.

## The levers, largest first

**The SQL and SQLite silences.** `sqlite/engine-repros` and
`postgres/sql-repros` print only their query's result, so a completion there is
NOT a measured silence: without a marker after the defective access a clean exit
cannot be told from a case that never reached its defect. A `defect_marker` in
those cases, then one run per arm, is the single biggest lever in the study.

**The groups with no reader.** `ffmpeg`, `wireshark` and `memcached`'s
`plain-heap` and `plain-temporal` groups were measured on this base under the
older mechanism arms. Their cells say `not read yet` because feeding those
bundles to these three arms needs a reader for their shape. `RECONCILE.md`
counts what that would fill -- and records the one case where the read is NOT
valid: the older `spatial` arm is bounds-only, so on a temporal crossing it
misses where `capstone-sysalloc` would catch.

**Wireshark's wmem silences.** 22 cells whose disposition needs the quarantine
probe, and the probe needs the case to run on CheriBSD. There is no tshark
CheriBSD build. The largest block with no cheap path.

**The memcached port defect.** Every case that releases a slab chunk faults at
`slabs.c:533:10`, cause 24 untagged, ON THE UPSTREAM-FIXED SEQUENCE TOO. The
first hypothesis is refuted -- a silent image of the same campaign has 187
`shrink`->`stc` pairs against this one's 317 -- so the next step is the register
state at the trap, not more disassembly reading.

## Two things that will break quietly if left

**25 declarations are keyed on the case DIRECTORY.** A directory carries a
leading ordinal, and deleting any case in a group renumbers every directory
after it. `sqlite/engine-repros` already demonstrated the failure: its
`capstone-sublet` cell reported 5 catches where its bundle holds 22, because
five directories happened to keep their number and seventeen matched nothing and
fell to `not-run`. A `not-run` cell looks like a legitimate gap, the gate stays
green, and nothing says a word. That one is fixed by keying on the slug; the
other 25 want a checker rule, not vigilance.

**Nothing reads the staleness fields.** `tools/run-virtual-corpus.py` records
`platform_sha256` for a campaign and an `image_sha256` per row, and no tool asks
them. Three full campaigns ran in one night for at most 56 open cells because
the platform moved twice underneath. A query that refuses rows whose inputs have
not changed, and flags rows whose inputs have, comes before the 104.

## Not open, on purpose

- **PoisonCap.** Excluded by `arms.json`: it is our adapter for a competitor's
  platform, so a reader is entitled to discount it. No cell reads it.
- **The xlang row corpora.** Exempt by design -- their rows are not case
  directories and they declare no arms.
- **A silence whose object never reached the system allocator.** Once its
  disposition is measured it is answered, not open: no sweep policy reaches an
  object that never arrives.
