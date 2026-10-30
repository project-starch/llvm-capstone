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
| `virtual-malloc` | `virtual-sqlite-memsys5`: memsys5 stock over its static arena; virtual mallocng below sees one object | 0 | 29 | 4 |
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

**The denominator is the 27 defects the review leaves counted.** Six of the 33 cases are out of it:
01, 04, 08, 09 and 18 carry `withdrawn` with its reason, 25 carries `duplicate_of` 00, and
catch-tables leaves all six out of every table and lists them by name. Of the 27, **19 are temporal
and 8 spatial** -- case 00 among the spatial, which is what its `class` now says and what upstream's
own "BUFFER OVERREAD" and the measured `cause=28` always said. It was 28 and 20 until 2026-10-30,
when 08 was withdrawn for the reason 01 and 09 already carried.

**Two configurations, and `tools/catch-tables.py --all --board` prints the nested one.** The
non-nested table is recomputed the same way, from the same cells, by the arms named under it.

### Nested: memsys5 over the case's static arena

| axis | cases | (1) base | (2) Sublet | (3) CheriBSD |
|---|---:|---:|---:|---:|
| temporal | 19 | **0** / 15 (+4 no reading) | **19 / 19** | **0** / 17 (+2 no reading) |
| spatial | 8 | **0 / 8** | **8 / 8** | **0 / 8** |

### Non-nested: the same defects with memsys5 left out of the build

Without an arena there is no inner allocator, so base and Sublet are one measurement: every SQLite
object is its own object of the platform allocator, bounded to its request and retired on free.

| axis | cases | (1) = (2) base = Sublet | (3) CheriBSD |
|---|---:|---:|---:|
| temporal | 19 | **17 / 17** (+2 out of denominator) | **12 / 12** (+5 no reading, +2 out of denominator) |
| spatial | 8 | **6 / 8** | **6 / 8** |

* **(1) base** is `virtual-malloc`, configuration `virtual-sqlite-memsys5`, bundle
  `results/2026-10-29-virtual/virtual-malloc`. It catches nothing on either axis, and so does its
  silicon twin: `vp2-malloc` and `silicon-sqlite-stock` read 0 as well, over the same configuration
  and the mechanism rule below. The four cells with no reading are 02 and 20 (a fault, unattributed),
  06 (reached, then neither completed nor faulted) and 11 (the runner's bound; see below).
* **(2) Sublet** is `virtual-nested-pools`, configuration `virtual-sqlite-memsys5-pools`, bundle
  `results/2026-10-29-virtual/virtual-nested-pools`. It has no temporal miss and no cell without a
  reading.
* **(3) CheriBSD** is `cheribsd-revocation`, run `results/2026-10-29-cheribsd/nested`
  (`shared/run-cheribsd.py --mode nested`, poscontrol gate FAULTed before any case ran). The two
  cells with no reading are 02 and 20: both faulted, neither at the function host ASan reported.
* **The non-nested columns** are `virtual-platform` (configuration `virtual-mallocng`, bundle
  `results/2026-10-29-virtual/virtual-platform`) and `cheribsd-platform`
  (`run-cheribsd.py --mode non-nested`, `results/2026-10-29-cheribsd/non-nested`, same gate). The
  two cases out of the non-nested denominator are 07 and 22: both reach for `sqlite_heap` to
  engineer the OOM their trigger needs, and that array exists only when memsys5 is configured.

**A catch is the mechanism rejecting the access, and since 2026-10-30 that is a declaration and not
a paragraph.** `tools/arms.json` gives `virtual-sqlite-memsys5`, `virtual-sqlite-memsys5-pools` and
`silicon-sqlite-memsys5` a `catch_causes` of `[25, 28]`: `25` INVALID_CAP is the lifetime check
refusing the access and `28` CAP_OOB is the bounds check, and those two are the only causes a
Capstone mechanism reports. A fault with any other cause means the access was allowed and the
program crashed later on what it read, so the judge scores it MISSED and keeps the fault in the
cell's `verdict_note`. What that costs each column is the measurement:

| arm | bundle | cells re-scored |
|---|---|---:|
| `virtual-nested-pools` (Sublet) | `results/2026-10-29-virtual/virtual-nested-pools` | **0** |
| `virtual-platform` (non-nested) | `results/2026-10-29-virtual/virtual-platform` | **0** |
| `virtual-malloc` (base) | `results/2026-10-29-virtual/virtual-malloc` | 3 -- 05, 19, 22, all `cause=24` |
| `vp2-malloc` (base, silicon) | `results/2026-10-18-vp2-rdx0/vp2-malloc` | 5 -- 02, 05, 24 `cause=24`; 19 `cause=4`; 22 `cause=13` |
| `silicon-sqlite-stock` (stock, silicon) | `results/2026-10-16-silicon/silicon-sqlite-stock` | 2 -- 02, 05, both `cause=24` |

Sublet loses nothing, because every catch it records was already `25` or `28`. The base arm loses
all of them, because not one of its was. That is the whole distance between the two columns, and it
is in the cells rather than in prose: every catch's `verdict_note` carries its cause and pc.

The declaration re-judges each stored Observation, so it landed with a **re-fold** of the four
bundles: `tools/derive-verdicts.py --refold --why ...` re-scores the stored observations under the
current judge, writes the new verdict beside each one, and refuses to touch the Observation itself,
so what was measured is byte-identical before and after. Each bundle's `inputs.json` carries a
`refolds` entry naming every cell and why. Nothing was re-run and nothing needs re-running: the
board's own `cause=4` and `cause=13` readings are the same consequence faults its QEMU twin reports
as `24` (silicon raises a load access or page fault where QEMU raises an untagged dereference, at
the same pc), and neither is a mechanism. An earlier revision of this file said the key had to wait
for a board re-run. That was wrong -- the obstacle was a missing tool, not a missing measurement --
and it is withdrawn. **The silicon lane's two arms are re-scored by this and should be told.**

**CheriBSD's nested column is 0 by the same rule.** Revocation acts on a guest `malloc` free, and
under memsys5 there is none: `SQLITE_CONFIG_HEAP` puts every SQLite allocation inside `sqlite_heap[]`,
`sqlite3_free` returns the block to memsys5's own pool, and nothing ever reaches the guest allocator
for a sweep to act on. The arm faults on five cases (02, 05, 19, 20, 22) and three of those are at
the function host ASan reported; until 2026-10-30 those three were recorded CAUGHT. They are now
MISSED, with the fault kept in the note, because what cleared the tag is memsys5 writing its
`Mem5Link` free-list words over the block and not revocation. Two things make that a reading rather
than an assumption. The three are **the same three cases** the base Capstone arm fails on as
`cause=24` -- one event, two architectures' names for it. And the revoking path is demonstrably live
where it exists: the same harness under `SYSALLOC=1 NOMEM5=1` records `allocator=system
revoke=on-every-free` on 31 of the 33 cases, and that arm catches 12 of the 12 it can read. The two
unattributed cells (02, 20) stay `NO-READING`, as they do on the base arm: attribution comes before
the mechanism rule, there as here.

**Five `cheribsd-platform` cells have a fault that cannot be tied to the defect** -- 11, 16, 17, 21
and 23, whose declared access is a libc primitive (`memcmp`, `memcpy`). The fault pc lies in a
shared object, the run record does not carry that object's load base, and the guest has ASLR enabled
(`kern.elf64.aslr.enable=1`) so the base cannot be reconstructed afterwards. They are
`NO-READING:unattributed`, which is why that column reads 12/12 and not 17/17. Every cell it can
read is a catch, so 17/17 is what it would read if all five resolved that way, and 12/17 if none
did; the tree states 12/12 and neither. It is the only gap left between this corpus and the
figures the paper quotes.

**The instrument that closes them is in the tree, and this is its pre-registration (2026-10-30).**
A load base recorded afterwards was the wrong fix: with ASLR on there is nothing to record it from.
`repro322_common.h`'s fault handler now resolves the site where it CAN be resolved -- inside the
faulting process, with `dladdr` -- and prints `<<FAULT-WHERE>> obj= base= sym= symaddr= image=
imagebase=`, which `shared/run-cheribsd.py` records per case. The gate faults before every run, so
the gate's own FAULT-WHERE is the resolver's positive control and the record carries it as
`resolver`; a run whose gate faults and prints nothing has a broken resolver, and its unattributed
cells stay NO-READING rather than turning into misses. `dladdr` takes rtld's lock and is not
async-signal-safe, which is a deliberate trade: these faults are in the program's own code or in a
libc string primitive, never inside rtld, and if the lock were ever held the handler blocks and the
runner's timeout reports TIMEOUT -- a visible non-reading, not a wrong one.

Written before the run, including the arms that must NOT move:

* the gate prints a `<<FAULT-WHERE>>` line, and `resolver` is non-null in both records;
* 11, 16, 17, 21 and 23 resolve to `memcmp` or `memcpy` in the C library, which is each case's
  declared access, so all five become CAUGHT and `cheribsd-platform` reads **17 / 17** temporal;
* the 12 cells that already read stay exactly as they are, and so does the spatial 6 / 8;
* on the nested record the five faults (02, 05, 19, 20, 22) resolve inside the case image, 02 and
  20 to a function host ASan did not report, so `cheribsd-revocation` stays **0 / 17 (+2)** -- the
  mechanism rule, not the attribution, is what makes that column 0, and a better resolver must not
  move it.

Any other reading is a finding about the instrument, not about the arm.

**Two `cheribsd-platform` cells were not measurements of that arm, and are withdrawn (2026-10-30).**
07 and 22 ran under `--mode non-nested` and produced a silence and a fault, scored MISSED and CAUGHT.
Neither is a reading: `repro_init()`'s `NOMEM5` branch is what puts the case on the system allocator,
and both cases install memsys5 over `sqlite_heap` from their own `run_case()` without ever calling
`repro_init()`. What ran was the nested configuration under the stock quarantine. The run record
says so directly -- it carries `allocator=system` and `revoke=on-every-free` for 31 of the 33 cases
and for neither of these two -- and they are the same two the Capstone builder already reports not
applicable. Both are now `NO-READING:out-of-denominator`, which is what the Capstone side says, so
the two non-nested columns now count the same 17 cases.

**Case 08 is WITHDRAWN (2026-10-30), by the rule 01 and 09 were already withdrawn under.** Its
trigger does not reach the defect it names: `sqlite3Fts3Offsets` runs twice and all three
`fts3EvalRestart` calls come after both of them, so no saved poslist pointer is read across a
restart (gdb on the host -O0 build, 2026-10-11), host ASan is SILENT on it and it returns
(`results/2026-10-11-host-asan`), and in the 2026-10-29 run every one of the five arms reports
MISSED with its controls behaving -- including the two arms that catch 19 of 19 and 12 of 12 of the
cases around it. 02, 11 and 13 carried the same 2026-10-11 finding and were CORRECTED instead, each
with a replaced trigger that reaches its defect and a reading to show it; 08 is the one where no
correction was found, and the structural reason is that phrases in a NEAR group advance together, so
whenever a sibling restarts the earlier phrase's saved pointer is already nil. Upstream `51dd67080a`
ships no test and calls it a potential use-after-free, so there is no upstream shape to copy. The
defect is not disputed and the directory stays; a trigger that reaches it would bring the case back.
**With this the withdrawal rule is applied uniformly across the corpus** -- which is what an earlier
revision of this file recorded as the open question, with 01 and 09 withdrawn for the reason 08 and
11 were kept under.

**Case 11 has no reading on the base arm.** Its trigger reads the freed pending-terms hash key and
then `fts3EvalGatherStats` rebuilds a doclist from it; on an arm that does not stop the read, what
that walk does next depends on what overwrote the freed bytes. It returns on the host, on CheriBSD
nested and on the non-nested arms, and on `virtual-malloc` -- memsys5 recycling inside a 1 MiB arena
-- it does not, so the runner's bound ends it. That is a property of the defect on an unprotected
allocator, not of the harness, and it costs one cell in a column that reads 0 either way.

**What is still hand-written.** The two CheriBSD arms are the corpus's only cells not derived from a
verdicts-v1 bundle: `run-cheribsd.py` writes one JSON run record, and the cells are written from it
with the fault pc attributed by subtracting the load base. Folding those records into bundles would
put all seven arms under `derive-verdicts.py`, and it needs the guest binaries to re-symbolise, so
it waits on the next CheriBSD run rather than being done from the tree.

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

**`catch_causes` is DECLARED (2026-10-30), and what the review argued for is now measured.**
`tools/verdicts.py` takes an optional `catch_causes` on a configuration and refuses to call a fault
outside that set a catch; it is documented in `../../SCHEMA.md` and covered by
`../../tools/test_verdicts.py`. The three memsys5 configurations now declare `[25, 28]`, which is
what makes column (1) read 0 by measurement instead of by a paragraph. The table of what it
re-scores, arm by arm, is under "The result this tree can recompute" above.

**RETRACTED: that the declaration needed a board re-run.** This file argued, on 2026-10-29, that
because `virtual-sqlite-memsys5` is named by two arms -- `virtual-malloc` on QEMU and `vp2-malloc`
on the board -- and `derive-verdicts.py` re-judges every stored observation, the key had to wait for
the board to be re-run. The premise was right and the conclusion was wrong: re-judging a stored
observation is not re-measuring it, and the gate failed for want of a tool rather than for want of
data. `derive-verdicts.py --refold` is that tool. It re-scores the stored observations, writes the
new verdict beside each one, refuses to alter any Observation, and records in the bundle what it
did and why. The board bundle is re-folded, its five cells are MISSED, and nothing was re-run.

`virtual-mallocng`, which `virtual-platform` names, is still NOT declared: it is shared with twelve
other corpora (`ffmpeg`, `memcached`, `postgres`, `wireshark`, `sqlite/capi-repros`), so declaring
it is their decision as much as this corpus's. It would change nothing here -- every catch on that
arm is already `cause=25` or `28`, which the re-fold confirms by re-scoring none of them.

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
