# SQLite 3.22.0 engine-internal defects, against a nested allocator

43 defects in SQLite's own engine, reduced to one domain program each, and run
against the arena allocator SQLite ships with. The sibling corpus
[`../capi-repros`](../capi-repros) covers pointer-lifetime defects in host
*bindings* over the C API and records engine-internal defects as out of scope;
these are those.

**Why memsys5 is the boundary.** memsys5 hands out `&mem5.zPool[i*szAtom]` — a
pointer derived from one 256 KiB arena. Every sub-allocation therefore inherits
the arena's bounds, so a use-after-free lands inside a live object and an
overflow that stays in the arena is in bounds. A capability machine is correct
to stay silent on both, and ASan is blind for the same reason: the arena is one
static array it never sees carved up. That is the measurement, not a caveat.

## Shapes

Every case declares one of these, and the table partitions the corpus.

| shape | cases | why it is not the same test |
|---|---|---|
| free / stale read | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 | the base case: a lifetime ends and a surviving pointer reads the block |
| read past the end of the record | 0, 25, 26, 27, 28, 29, 30, 31, 32 | the pointer is live; the decode or walk leaves the object |

## Arms

| arm | target | what it answers |
|---|---|---|
| `spatial` | Capstone domain, base | does the unprotected machine notice? |
| `sublet` | Capstone domain, nested-allocator discipline | does per-sub-allocation revocation notice? |
| `cheribsd-revocation` | CheriBSD riscv64-purecap, revocation on | does CHERI plus revocation notice? |

## On virtual Capstone (2026-10-11), under the shared judge

| arm | configuration (`tools/arms.json`) | caught | missed | no reading |
|---|---|---:|---:|---:|
| `virtual-malloc` | `virtual-sqlite-memsys5`: memsys5 stock over its static arena; virtual mallocng below sees one object | 5 | 25 | 3 |
| `virtual-nested-pools` | `virtual-sqlite-memsys5-pools`: memsys5 on its Sublet port, the pool lent linear by the virtual heap | 27 | 6 | 0 |

Those four counts are **transcribed from the bundle `corpus.json` declares**, not measured afresh:
`verdict_bundles` names `results/2026-10-11-virtual-request-bounds/<arm>`, and this table now
agrees with the derived record taken from it -- `results/verdicts.tsv`, written by
`tools/derive-verdicts.py`, and the `arms` block of every `case.json`. Until 2026-10-20 it carried
the superseded 2026-10-10 bundle's figures, 4 and 18, which contradicted every `case.json` beside
them. The bundle is the only original (`../../SCHEMA.md`, "Verdicts: one judge, one original"), so
the figures here follow it.

Built by `ports/sqlite/repro322/build-virtual.py` (`--memsys5`, `--sublet`) from the adapted 3.22.0
amalgamation plus `fts5-azarg-patch.py`, run by `shared/run-virtual.py` on a persistent virtual VM,
judged by `tools/verdicts.py`; the current bundles are `results/2026-10-11-virtual-request-bounds/<arm>`
(memsys5 and the lookaside on CDERIVE/CREVOKE, each memsys5 block bounded to its request), verdicts
derived into each `case.json`. Before any case each arm runs `controls/` -- a memsys5 block used after free and
written 1 KiB past its start -- which complete on the stock arm and fault on the port, as declared.
The adapted amalgamation the bundles record (`amalgamation_sha256` 7fc92e19…) is stock 3.22.0
(sha256 7991acd0…) through `ports/sqlite/adapt-sqlite-322.sh` as of 423d13f5b81f. Before that commit
the script lacked the sorter's 16-byte alignment line and produced b446615b…, so the bundles' input is
reproducible only from that commit on.

A fault counts only in a function the case justifies (`fault_sites`): where host ASan reported it,
or what the source and `PROVENANCE.md` name as the stale access.

**Does the case make its invalid access at all?** memsys5 makes ASan blind, so
`shared/run-host-asan.py` builds every case natively with SQLite on the system allocator, lookaside
off, no page-cache bulk pool and a 4 GiB quarantine, and both controls must report
(`results/2026-10-11-host-asan/results.json`). On 2026-10-11 it reports 26 of the 33 cases as an
invalid heap access (17 heap-use-after-free, 9 heap-buffer-overflow, case 00's stale read among
them) and case 02 as a SEGV. Six are silent, 01, 04, 08, 09, 11 and 13. ASan's silence is not the finding, because it only sees
`malloc` lifetimes: each of the six was then traced under gdb at its own free and use sites, and
each case's `note` records what the trigger does instead. In five the lifetime never ends before the
use (01 never reaches its `sqlite3ExprListDelete`; 08 and 09 restart only after the consumer has
finished; 11's pending hash is already empty because every INSERT auto-commits; 13's in-place
rewrite runs but no reallocation is established). **04 is different**: it reaches the condition
upstream's fix guards, a cursor at `CURSOR_REQUIRESEEK` whose page is unpinned in the cache, and the
read is answered from the cursor's cached cell info instead. The lifetime that ended there is the
cursor's REFERENCE to the page -- the page is neither freed nor reassigned, so no `malloc`-level
mechanism can see it. Each case's `note` also names what its upstream regression test does that the
trigger does not. (04's stale object is a cached B-tree page, which the page cache recycles without
a free, so ASan's silence is weaker evidence there.) The same check showed four corpus triggers had
fallen behind the corrected copies in `ports/sqlite/cheribsd/cases/` (10, 14, 15, 22); they were
replaced, each `PROVENANCE.md` says how. The harness
prints `<tag> BEGIN` before the case body, not at the access, so a MISSED here rests on the case
having begun and returned.

Two faults every arm hit were the platform, not a defect: FTS5's default tokenizer computes
`&azArg[1]` with `azArg` NULL (CINCOFFSET on a non-capability traps on Capstone) -- the physical
build had fixed that with `ports/sqlite/fts5-azarg-patch.py`, which the virtual builder now applies
too -- and `sqlite3Fts5IterTerm` returns `&z[1]` with `z` NULL (case 12, open, no reading).

**The physical `sublet` arm below is superseded for the per-bug table.** 25 of its 34 "detected"
were VM halts at one kernel pc that nothing ties to a defect.

## The result this tree can recompute

One table, and `tools/catch-tables.py --all --board` prints it. Every column names the arm, the
configuration and the bundle it comes from, so a reader can check each figure against the record
instead of taking it on trust. The tables used to be in the pull request body, where a reviewer
without `gh` cannot read them; they belong in the tree.

**The denominator is the 28 defects the review leaves counted.** Five of the 33 cases are out of it:
01, 04, 09 and 18 carry `withdrawn` with its reason, 25 carries `duplicate_of` 00, and catch-tables
leaves all five out of every table and lists them by name. Of the 28, **20 are temporal and 8
spatial** -- case 00 among the spatial, which is what its `class` now says and what upstream's own
"BUFFER OVERREAD" and the measured `cause=28` always said.

| axis | cases | (1) base | (2) Sublet | (3) CheriBSD |
|---|---:|---:|---:|---:|
| temporal | 20 | 5 / 17 (+3 no reading) | 17 / 20 | 5 / 20 |
| spatial | 8 | 0 / 8 | 8 / 8 | 0 / 8 |

* **(1) base** is the arm `virtual-malloc`, configuration `virtual-sqlite-memsys5`, bundle
  `results/2026-10-11-virtual-request-bounds/virtual-malloc`. The three cells with no reading are
  06, 12 and 20; catch-tables prints them as `(+3 ?)`.
* **(2) Sublet** is `virtual-nested-pools`, configuration `virtual-sqlite-memsys5-pools`, bundle
  `results/2026-10-11-virtual-request-bounds/virtual-nested-pools`. The three temporal misses are
  08, 11 and 13. Each one's `note` records that the trigger does not reach the defect it names; the
  replaced triggers the review wrote are not in this tree, so this column is 17 and not 20.
* **(3) CheriBSD** is `cheribsd-revocation`, hand-written from `results/matrix.tsv` and measured
  2026-10-06 in one boot whose positive-control gate was satisfied before any case ran. It is the
  one column with no bundle.

The retired physical arms `spatial` and `sublet` read differently over the same 28 -- 6/20 and 2/8,
19/20 and 7/8 -- and are superseded (see the note above). A column that mixes them with the virtual
arms is not a measurement of anything.

**Every one of (1)'s five temporal catches is `cause=24`**: an untagged value used where a
capability was required, which is the architecture's baseline tag check firing on the CONSEQUENCE of
an access it had already allowed, not a rejection of it. Scored by mechanism they are not
detections, and `catch_causes` is the key that says so. It is implemented and tested here and
declared by no configuration, because declaring it re-judges every stored Observation while
`derive-verdicts.py` refuses a bundle whose recorded verdict the current judge disagrees with (13
cells, measured 2026-10-28) and writes the recorded verdict into the cell regardless. So this column
reads 5; the mechanism-scored 0 lands with the declaration and a re-fold of the four bundles, not
before. (2) has one such cell, case 02; its other 16 temporal catches are `cause=25` or `28`.

**What this table does not carry: the non-nested configuration.** The same 28 defects with memsys5
out of the build, so SQLite allocates through the platform's `malloc`, is a second configuration of
this corpus and is not in it. The controls are here -- `controls/control_uaf_malloc.c` and
`control_bounds_malloc.c`, which the configuration `virtual-mallocng` requires to FAULT -- and
`build-virtual.py` builds it when `--memsys5` is left off, but no arm declares it, no case carries a
cell for it, and there is no bundle. No figure for it is recomputable from this tree, so none is
quoted here; it lands as its own arm with its own run. Two of the 28 will have no non-nested row:
both configure memsys5's heap themselves to force the OOM path their trigger needs, so without an
arena there is nothing to trigger.

**Case 02's cells are under correction.** The review found that its trigger never ended the cached
Btree's lifetime, so the temporal row above counts it on a reading of a trigger that did not reach
the defect. The corrected trigger needs a port fix and a re-run of all four bundle-derived arms, and
lands with them in a separate change.

## The case-by-case review of 2026-10-15/17

Every case's record was re-read against its own upstream commit, its `PROVENANCE.md` and the
bundle cells it cites. What that corrected is in each case's `note`; the four findings that
touch more than one case are here.

**Case 00 is an over-read of a live object, not a use-after-free.** Its own upstream commit
`adfb203a7d` is "Fix a buffer overread in fts5 ...", and `fts5SegIterReverse` keeps `pLast`
live across the decode -- only the pages it replaced are released. The case therefore moves
from the `free / stale read` row of the shape table to `read past the end of the record`, and
its `shape` field moves with it. ASan named it a use-after-free because it names a report by
the shadow state of the byte touched, and under one heap layout the over-read lands inside a
neighbour that had just been freed. The directory name keeps `stale_read`: the run bundles
refer to the case by that name and renaming it would orphan the measurements that cite it.
No arm's verdict changes -- only the axis.

**Two cases are duplicates and now say so in the schema rather than only in prose.** 09 records
the same upstream defect as 10, and 25 the same as 00; each declares `duplicate_of`, and
`tools/catch-tables.py` leaves both out of every table (`../../SCHEMA.md`, "Optional fields").
They are kept so numbering and citations stay valid.

**`catch_causes` exists but is NOT declared for these two configurations, deliberately.**
`tools/verdicts.py` now takes an optional `catch_causes` on a configuration and refuses to call
a fault outside that set a catch; it is documented in `../../SCHEMA.md` and covered by tests in
`../../tools/test_verdicts.py`. Declaring `[25, 28]` on `virtual-sqlite-memsys5` and
`virtual-sqlite-memsys5-pools` would re-score every `cause=24` catch as MISSED, which is the
reading the review argues for: a `cause=24` fault is the architecture's tag check firing on a
value the program had already read out of the dead object, one or two frames past an access
the mechanism allowed. It is not declared here because `tools/derive-verdicts.py` re-judges
every stored observation against the current `arms.json`, and a bundle is only ever written by
a runner: declaring the key without re-running the two arms would leave the gate failing on
every cell it re-scores. The mechanism lands first; the declaration waits on a run.

**The arena is part of the configuration, so a case that depends on it declares `arena_bytes`.**
3.22.0's `memsys5Init` does not require a power of two, and the two builders default
differently -- `ports/sqlite/repro322/build-virtual.py` to 1048576 and
`ports/sqlite/cheribsd/build-corpus-cheri.sh` to 262144. On an arena-granular arm the arena IS
the bound, so an undeclared case is not comparable across arms: the two runs would differ in
the size of the only object the mechanism can see. `arena_bytes` and `arena_bytes_why` record
it per case.

## What the three arms measured

`results/20261004/matrix.tsv`, one row per case per arm:

| arm | detected | silent | hang | no marker | not run |
|---|---:|---:|---:|---:|---:|
| `spatial` | 10 | 28 | 3 | 2 | 0 |
| `sublet` | **34** | 4 | 1 | 1 | 3 |
| `cheribsd-revocation` | 2 | 26 | 0 | 0 | 12 (3 not reached) |

**Sublet detects 34 of the 40 it ran; the base machine detects 10 of 42.** Same
sources, same build, same QEMU, one run apart: the two arms differ in the
allocator discipline and in nothing else a case can see. That difference is what
this corpus was built to measure.

**25 of Sublet's 34 stopped the VM rather than returning**, and all 25 are the
same instruction: `pc = 0x800239e4`, in kernel space. The delivered faults are
all in domain space (`0x101…`, `0x102…`). Cooperative fault recovery returns
through the domain's caller, so a fault taken in the monitor or the kernel has
no frame to return through and must halt. That is why the Sublet arm costs
seven boots on a group where the base arm costs one — a property of where the
fault lands, not a misconfiguration, and the run records it rather than hiding
it in a wall-clock number.

**A silent row does not mean the same thing on every arm.** The CheriBSD runs
carry a defect-site probe: a marker at the exact `sqlite3.c` line where the host
ASan oracle, running the same case source with the same flags, reported the
error, so each oracle string says whether the site was reached and whether the
defective access was witnessed. Three rows come back `not-reached`, and before
the probe existed those three were counted as "the mechanism missed it".

The `spatial` and `sublet` arms have no such probe. A marker cannot survive
there: the domain's `out_text()` writes a shared region the host reads only
after the domain returns, so a faulting domain prints nothing and the marker
goes with it. On Capstone the equivalent evidence is a symbolized fault pc.
Until that exists, a silent row on those two arms means the exit code was clean
and nothing more.

## Running it

The runner is `capstone/ports/sqlite/repro322/corpus322.sh`, by group:

    corpus322.sh build <group>
    corpus322.sh run   <group>

Groups exist because cases need different `SQLITE_ENABLE_*` flags, and the flag
is sometimes load-bearing: case 10 needs `SQLITE_ENABLE_FTS3_PARENTHESIS` or the
parentheses are ordinary characters, a flat query runs, and the case passes
having tested nothing. Cases 38, 39 and 42 need `SQLITE_ENABLE_DBSTAT_VTAB` or
the statement fails with "no such table: dbstat".

Two options make a run much cheaper, and they only work together:

* `SQLITE_DOMAIN_FAULT_RECOVERY=1` at build time, and
* `CAPSTONE_QEMU_BINARY` pointing at a trap-delivery QEMU
  (`capstone-qemu` branch `runtime/1-domain-trap-delivery`, commit `77d69353b7`).

With both, a capability fault returns through the domain's caller instead of
terminating QEMU, so one boot runs a whole group. Measured: a group of eleven
cases used to need six boots because three of them faulted; the base arm now
runs 43 cases with zero VM halts. Hangs still cost a boot — fault recovery
cannot contain a domain that never returns, which is why cases 40, 41 and 42 are
the expensive ones.

## What is not here

Five further R2 fuzz-diff images (`fz01`, `fz03`, `fz04`, `fz05`, `fz07`) have a
host-ASan signature but no derived trigger: fuzzcheck pairs every database with
every script and the provenance kept only the database id, so the SQL has to be
re-derived by replay. Eight more entries in the collection are instruments
rather than defects — five baseline probes and three allocator demonstrators —
and stay in the port.
