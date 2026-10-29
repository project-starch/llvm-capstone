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
| `virtual-platform` | virtual Capstone, memsys5 OUT of the build | with no inner allocator at all, does the platform notice? |
| `cheribsd-platform` | CheriBSD riscv64-purecap, memsys5 OUT of the build | the same question on the CHERI side -- the arm where revocation can act |

## On virtual Capstone (2026-10-29), under the shared judge

| arm | configuration (`tools/arms.json`) | caught | missed | no reading |
|---|---|---:|---:|---:|
| `virtual-malloc` | `virtual-sqlite-memsys5`: memsys5 stock over its static arena; virtual mallocng below sees one object | 3 | 26 | 4 |
| `virtual-nested-pools` | `virtual-sqlite-memsys5-pools`: memsys5 on its Sublet port, the pool lent linear by the virtual heap | 29 | 4 | 0 |
| `virtual-platform` | `virtual-mallocng`: no memsys5 at all, so every SQLite object is one of mallocng's | 25 | 6 | 2 |

Those counts are **transcribed from the bundles `corpus.json` declares**, not measured afresh:
`verdict_bundles` names `results/2026-10-29-virtual/<arm>`, and this table agrees with the derived
record taken from them -- `results/verdicts.tsv`, written by `tools/derive-verdicts.py`, and the
`arms` block of every `case.json`. The bundle is the only original (`../../SCHEMA.md`, "Verdicts:
one judge, one original"), so the figures here follow it. The superseded
`results/2026-10-11-virtual-request-bounds/<arm>` bundles stay in the tree as the record of what
the pre-review triggers measured; they are no longer what any table reads.

Built by `ports/sqlite/repro322/build-virtual.py` (`--memsys5`, `--sublet`, and with neither for
`virtual-platform`) from the adapted 3.22.0 amalgamation plus `fts5-azarg-patch.py` and
`fts3-doclist-null-patch.py`, run by `shared/run-virtual.py` on a persistent virtual VM, judged by
`tools/verdicts.py`. Before any case each arm runs the controls its configuration declares --
`controls/control_*_mem5.c` for the two memsys5 arms, `controls/control_*_malloc.c` for
`virtual-platform` -- a block used after free and written 1 KiB past its start, which must behave as
`tools/arms.json` says or the run is void. In the 2026-10-29 run every control behaved as declared.
The adapted amalgamation the bundles record (`amalgamation_sha256` 7fc92e19…) is stock 3.22.0
(sha256 7991acd0…) through `ports/sqlite/adapt-sqlite-322.sh` as of 423d13f5b81f; regenerating it
from that script on 2026-10-29 reproduced 7fc92e19… exactly, so the engine source is the same bytes
the 2026-10-11 arms ran. Before that commit the script lacked the sorter's 16-byte alignment line
and produced b446615b…, so the bundles' input is reproducible only from that commit on.

On CheriBSD the two arms are run by `shared/run-cheribsd.py` (`--mode nested`, `--mode
non-nested`), which writes one JSON record per run rather than a verdicts-v1 bundle:
`results/2026-10-29-cheribsd/{nested,non-nested}`. Each run's `poscontrol` gate -- a plain
system-allocator use-after-free -- FAULTed before any case ran, so both are readings. The cells are
written from those records, with the fault pc attributed by subtracting the load base (0x100000 for
the PIE case images; the one non-PIE image, `spellfixoom`, is loaded where it was linked).

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

**Two configurations, and `tools/catch-tables.py --all --board` prints the nested one.** The
non-nested table is recomputed the same way, from the same cells, by the arms named under it.

### Nested: memsys5 over the case's static arena

| axis | cases | (1) base | (2) Sublet | (3) CheriBSD |
|---|---:|---:|---:|---:|
| temporal | 20 | 3 / 16 (+4 no reading) | **19 / 20** | 3 / 18 (+2 no reading) |
| spatial | 8 | 0 / 8 | **8 / 8** | 0 / 8 |

### Non-nested: the same defects with memsys5 left out of the build

Without an arena there is no inner allocator, so base and Sublet are one measurement: every SQLite
object is its own object of the platform allocator, bounded to its request and retired on free.

| axis | cases | (1) = (2) base = Sublet | (3) CheriBSD |
|---|---:|---:|---:|
| temporal | 20 | **17 / 18** (+2 out of denominator) | 13 / 15 (+5 no reading) |
| spatial | 8 | **6 / 8** | **6 / 8** |

* **(1) base** is `virtual-malloc`, configuration `virtual-sqlite-memsys5`, bundle
  `results/2026-10-29-virtual/virtual-malloc`. The four cells with no reading are 02 and 20
  (a fault, unattributed), 06 (reached, then neither completed nor faulted) and 11 (the runner's
  bound; see below).
* **(2) Sublet** is `virtual-nested-pools`, configuration `virtual-sqlite-memsys5-pools`, bundle
  `results/2026-10-29-virtual/virtual-nested-pools`. The one temporal miss is 08.
* **(3) CheriBSD** is `cheribsd-revocation`, run `results/2026-10-29-cheribsd/nested`
  (`shared/run-cheribsd.py --mode nested`, poscontrol gate FAULTed before any case ran). The two
  cells with no reading are 02 and 20: both faulted, neither at the function host ASan reported.
* **The non-nested columns** are `virtual-platform` (configuration `virtual-mallocng`, bundle
  `results/2026-10-29-virtual/virtual-platform`) and `cheribsd-platform`
  (`run-cheribsd.py --mode non-nested`, `results/2026-10-29-cheribsd/non-nested`, same gate). The
  two cases out of the non-nested denominator are 07 and 22: both reach for `sqlite_heap` to
  engineer the OOM their trigger needs, and that array exists only when memsys5 is configured.

**Every catch in column (2) is a mechanism cause.** 19 of the 20 temporal catches are `cause=25`
INVALID_CAP -- the lifetime check refusing the access -- and all 8 spatial catches are `cause=28`
CAP_OOB, the bounds check. Column (1)'s three catches are all `cause=24` UNEXP_OP_TYPE: an untagged
value used where a capability was required, which is the architecture's baseline tag check firing on
the CONSEQUENCE of an access it had already allowed. That is the whole distance between the two
columns, and it is in the cells, not in prose: `verdict_note` carries the cause and the pc for every
catch. Before the 2026-10-29 run column (2)'s catches were themselves mostly `cause=24`; the
corpus-SDK rebuild and the corrected triggers are what moved them onto the mechanism.

**CheriBSD's three nested catches are not revocation, and the arm's own oracle says why.**
Revocation acts on a guest `malloc` free, and under memsys5 there is none: `sqlite3_free` returns
the block to memsys5's own pool inside one array the guest allocator sees as a single live object.
So nothing was revoked, and what cleared the tag is memsys5 writing its `Mem5Link` over the freed
block. The three (05, 19, 22) are recorded CAUGHT because the fault is at the function host ASan
reported -- the same rule the virtual arms use -- and each one's `verdict_note` states the
allocator-level reason it cannot be the mechanism. Scored by mechanism this column is 0, which is
what the non-nested column measuring 6/8 and 13/15 on the same guest demonstrates: there,
revocation can act, and it does.

**Five `cheribsd-platform` cells have a fault that cannot be tied to the defect** -- 11, 16, 17, 21
and 23, whose declared access is a libc primitive (`memcmp`, `memcpy`). The fault pc lies in a
shared object, the run record does not carry that object's load base, and the guest has ASLR enabled
(`kern.elf64.aslr.enable=1`) so the base cannot be reconstructed afterwards. They are
`NO-READING:unattributed`, which is why that column reads 13/15 and not 18/20. Recording the
faulting object's base in the run is the one change that closes them.

**Case 08 is in the denominator and is the one temporal miss on every arm.** Its `note` records that
the trigger does not reach the defect -- all three `fts3EvalRestart` calls come after both
`sqlite3Fts3Offsets` calls, so no saved poslist pointer is read across a restart -- and four shapes
were tried without reaching it, for a structural reason: phrases in a NEAR group advance together,
so whenever a sibling restarts the earlier phrase's saved pointer is nil. Upstream `51dd67080a`
ships no test and calls it a potential UAF. It is neither corrected nor withdrawn.

**Case 11 has no reading on the base arm.** Its trigger reads the freed pending-terms hash key and
then `fts3EvalGatherStats` rebuilds a doclist from it; on an arm that does not stop the read, what
that walk does next depends on what overwrote the freed bytes. It returns on the host, on CheriBSD
nested and on the non-nested arms, and on `virtual-malloc` -- memsys5 recycling inside a 1 MiB arena
-- it does not, so the runner's bound ends it. That is a property of the defect on an unprotected
allocator, not of the harness, and it costs one cell in column (1), whose reading is 0 by mechanism
either way.

**Provenance of the 2026-10-29 bundles.** The adapted amalgamation is stock 3.22.0 through
`ports/sqlite/adapt-sqlite-322.sh`, which reproduces the `amalgamation_sha256` 7fc92e19... the
2026-10-11 bundles record, so the engine source is byte-identical to that run and the only variables
are the corrected triggers and the new arm. The SDK is a corpus SDK with an effective
`-fno-builtin` (C-81), built from dev `92ebd4a62542`: that is the last revision whose runtime emits
a `.capstone_virtual` image the pinned platform kit's launcher accepts (`CPONVVM5`; dev
`13719fd944b2` moved it to `CPONVVM7`, which the kit's 2026-10-16 launcher rejects with
`ENOEXEC`), and the first that honours `CAPSTONE_SDK_EXTRA_CODEGEN`. The platform is the pinned kit
`kit-pinned` of 2026-10-17 -- its own QEMU, Image, firmware, rootfs, launcher and module together,
as `tools/virtualvm.py` requires. Every bundle's `inputs.json` records all of it.

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

**One case is a duplicate and now says so in the schema rather than only in prose.** 25 records
the same upstream defect as 00 and declares `duplicate_of`, and `tools/catch-tables.py` leaves it
out of every table (`../../SCHEMA.md`, "Optional fields"). It is kept so numbering and citations
stay valid. The duplicate is plausible, not cross-checked: 415540ddaa and adfb203a7d are two guards
on one read, and both pools cells are cause 28 in `sqlite3Fts5GetVarint` at the same in-image offset (pc ...224c,
differing only in load base), but neither
fix's corrupt image has been run against the other fix. (09 was earlier declared a duplicate of 10;
that is withdrawn, and 09 now carries `withdrawn` instead.)

**`catch_causes` exists and is still NOT declared, and the 2026-10-29 run narrowed why.**
`tools/verdicts.py` takes an optional `catch_causes` on a configuration and refuses to call a fault
outside that set a catch; it is documented in `../../SCHEMA.md` and covered by
`../../tools/test_verdicts.py`. Declaring `[25, 28]` on `virtual-sqlite-memsys5` would re-score that
arm's three `cause=24` catches as MISSED, which is the reading the review argues for and would make
column (1) read 0 by measurement instead of by the paragraph above. The run removed the old
obstacle -- there are fresh bundles for all three virtual arms, judged with controls that behaved as
declared -- and left a sharper one:

* `virtual-sqlite-memsys5` is named by TWO arms of this corpus, `virtual-malloc` on QEMU and
  `vp2-malloc` on the board. `tools/derive-verdicts.py` re-judges every stored observation against
  the current `arms.json`, so declaring the key re-scores the board bundle
  `results/2026-10-18-vp2-rdx0/vp2-malloc` as well, and that bundle records `cause=24` catches of
  its own (see the 2026-10-20 correction above). The gate then fails on every cell it re-scores.
  The key therefore waits on a board re-run, not a QEMU one.
* `virtual-mallocng`, which `virtual-platform` names, is shared with twelve other corpora
  (`ffmpeg`, `memcached`, `postgres`, `wireshark`, `sqlite/capi-repros`). Declaring the key there
  would re-judge their bundles too. It does not need declaring for this corpus in any case: every
  catch on that arm is already `cause=25` or `28`.

So the one arm where the declaration would change a figure is the one whose configuration the board
shares. That is a note for the silicon lane, not a change this tree can make alone.

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

## Side comparison: PoisonCap's published memsys5

The temporal cases were also run on PoisonCap's published artifact. That is a side comparison, not an
arm (`SCHEMA.md`, "Side comparisons"), and its bundles are separate from the verdicts above:
* `results/2026-10-24-poisoncap/`: the first comparison, with the corpus's 1 MiB heap. Its
  `PAPER-NOTES.md` records the headline then measured.
* `results/2026-10-26-poisoncap-heap256/`: PoisonCap's own configuration (256 MiB, minimum
  request 16), giving 13 of 17 at the site against Sublet's 17, and the Sublet block port with its
  ablation.
* `results/2026-10-27-poisoncap-libc/`: the C library's revocation switched on, stopped.

The harness is `ports/sqlite/poisoncap/` (platform, corpus, speedtest1, reuse), and a backported
lookaside use-after-free that is not a case of this corpus is under `ports/sqlite/poisoncap/backport/`.
