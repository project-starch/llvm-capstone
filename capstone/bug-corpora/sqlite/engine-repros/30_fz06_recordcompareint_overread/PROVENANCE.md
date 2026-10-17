# fz06 -- fz06

Upstream fix `FZ06`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz06 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     vdbeRecordCompareInt:77204, heap-buffer-overflow
 *
 * PROVENANCE OF THE TRIGGER. This entry carried only the database and an ASan
 * log; the SQL was never recorded, because fuzzcheck pairs every database with
 * every script in the corpus and the provenance kept only the database id. The
 * statement below was re-derived by replaying candidate statements one process
 * at a time against a FRESH copy of the image -- a cumulative script does not
 * work here, since the earlier statements rewrite the very corruption the later
 * ones need. It reproduces the recorded crash at the SAME source line.
 *
 * The image is embedded and served by repro322_memfs.c, an in-domain in-memory
 * FILESYSTEM: the pager needs a rollback journal for any write, and memvfs.c
 * serves the main database only and takes its buffer as an integer in the URI,
 * which on a capability machine is a pointer forged from a scalar.
 *
 * REACHABILITY: every statement's rc is printed. SQLITE_CORRUPT (11) on the
 * statement that should reach the defect means SQLite rejected the image first
 * -- a real negative, not a silent pass.
```

## Review 2026-10-16 -- the only Sublet miss in this corpus is not a miss

### The four readings

| reading | result |
|---|---|
| host ASan | re-run: `heap-buffer-overflow`, 20 bytes after a 32-byte region, `vdbeRecordCompareInt <- sqlite3BtreeMovetoUnpacked <- sqlite3VdbeExec <- sqlite3Step` -- identical to the recorded bundle |
| virtual-malloc | **MISSED**: `open_rc=0`, `s0_rc=11`, `NOTRAP done`, `RETURNED` |
| virtual-nested-pools | **CAUGHT**: `cause=28` offset 536 in `vdbeRecordCompareInt`, fault address `0x3facf69394` |
| cheribsd-revocation | **MISSED**: `s0_rc=11`, `RETURNED rc=0`, probe site 65 reports `hits=3 witnesses=3`, every read `kind=OOB` |

### Why the recorded Sublet miss is wrong

Both physical rows explain their silence the same way, quoted from `results/matrix.tsv`:

> The case's own return code says why nothing was detected: `s0_rc=11`, SQLITE_CORRUPT -- the
> trigger's database image is rejected

**That reasoning does not hold**, and the two virtual arms show why on one image and one
statement:

```
virtual-malloc        fz06 open_rc=0 / fz06 s0_rc=11 / fz06 NOTRAP done / fz06 RETURNED
virtual-nested-pools  fz06 open_rc=0 / <fault> cause=28 pc=... in vdbeRecordCompareInt
```

Without Sublet the over-read **completes**: the decode reads past the record, returns garbage,
and SQLite's own corruption check rejects the result *afterwards*. With Sublet the read is
refused at the violation, before `s0_rc` is ever printed.

So `SQLITE_CORRUPT` is a **consequence** of the out-of-bounds read, not evidence that the read
never happened -- and the defect is reached on exactly the statement whose return code was taken
as proof that it was not. This was the only `not-detected` cell in Sublet's SQLite column.

It also settles the base cell: `virtual-malloc` is a genuine reached-and-silent, because the
Sublet arm establishes reachability on the same statement and the same image.

### The CheriBSD cell: a NO-READING that was this arm compiling a different program

Recorded on 2026-10-16 as a NO-READING -- with the image restored the arm gave `s0_rc=0`,
`s1_rc=11`, and probe site 65 reported `hits=0`, so `vdbeRecordCompareInt` was never
entered. The explanation offered, "a different statement sequence and a different build
reject the image earlier", was wrong, and so was reading it as a platform difference.

**This arm was not running this case.** It compiled its own copy,
`ports/sqlite/cheribsd/cases/case_fz06_r2.c`, which ran `PRAGMA integrity_check;` BEFORE the
trigger, as a "reachability probe". On a corrupt image that pragma makes SQLite reject the
following `SELECT` with `SQLITE_CORRUPT`, so the defect could not be reached and the run
scored as a clean completion. That copy's own header claimed "the trigger is identical, so
the arms differ by platform alone". It is the third such divergence this review has found,
after cases 03 and 21.

Fixed at the root rather than in the copy: `build-corpus-cheri.sh` now takes the case program
from the corpus directory `tags.tsv` names, adding `repro322_memfs.c` when the body serves its
image through the in-memory VFS -- the rule `build-virtual.py` already applies. The arm's
`cases/` copies remain only for tags with no corpus directory.

**Re-measured 2026-10-17** from the corpus `case.c`. Gates: `poscontrol` exit 162 with
`revocation=1`; both binaries verified in the guest by sha256, and by `strings` to carry the
embedded image (`repro_memfs`) rather than `/root/corpus/*.db`; the probe binary verified to
contain site 65's payload, so a `hits=0` could not have been the instrument's absence.

```
fz06 open_rc=0 / fz06 s0_rc=11 / fz06 NOTRAP done / fz06 RETURNED rc=0      exit=0
LB_WITNESS site=65 kind=OOB off=30685 n=8 req=20160 tag=1 cb=0x1d8c00+262144
LB_WITNESS site=65 kind=OOB off=30663 n=8 req=20160 tag=1 cb=0x1d8c00+262144
LB_WITNESS site=65 kind=OOB off=52    n=8 req=25    tag=1 cb=0x1d8c00+262144
LB_SUMMARY site=65 hits=3 witnesses=3 first_witness_at_hit=1
```

**The cell is a MISS, not a no-reading.** The defect site is entered three times and every
read is out of bounds. `s0_rc=11` is now identical to `virtual-malloc`'s, which is what one
shared source should give. Why the mechanism stays silent is in the witness line: the
capability behind the read has `tag=1` and bounds `cb=<arena>+262144` -- the whole 256 KiB
`sqlite_heap[]` -- and the over-read never leaves it. That is the signature case 00 records
for the same arm. The third witness is the sharpest of the three: eight bytes at offset 52 of
a block whose REQUESTED size was 25. memsys5 rounds that to a 64-byte atom, so the read sits
inside the atom and 27 bytes past the request; a `malloc`-granularity bound cannot see it,
and Sublet's request bound is exactly what catches it with `cause=28`.

### The defect

```c
const u8 *aKey = &((const u8*)pKey1)[*(const u8*)pKey1 & 0x3F];
int serial_type = ((const u8*)pKey1)[1];
...
switch( serial_type ){ case 1: lhs = ONE_BYTE_INT(aKey); ... case 6: /* 8 bytes */ ... }
```

The offset comes from the record's own header byte, up to 63 bytes in, and is never compared
against `nKey1`; then up to eight bytes are read at `aKey` according to `serial_type`. Probe
site 65 prints `pKey1`, `nKey1`, the header byte, `serial_type`, `aKey` and whether the widest
read already passes the end.

### One stale cross-reference corrected

`fidelity` pointed at "case 34", which does not exist -- the corpus went from 38 cases to 33 on
2026-10-06 and this reference was left behind, as cases 21 and 23's were. It is case 29.

## Re-measured 2026-10-17: one arena on every arm, every arm from the corpus source

Two things were wrong with the earlier spatial readings and both are fixed here. The arms did not
share an arena -- the virtual images were built at the 1 MiB header default, CheriBSD at 262144 --
and on an arena-granular arm the arena IS the bound, so the two were comparing arenas rather than
mechanisms. And this arm compiled its own copy of the case from `ports/sqlite/cheribsd/cases/`
rather than the corpus `case.c`. Both are closed: the case declares `arena_bytes` (1048576 here),
all three builders read it, and `build-corpus-cheri.sh` now takes the case program from the corpus.

| arm | verdict | evidence |
| --- | --- | --- |
| virtual-malloc | **MISSED** | both controls as declared; reached and silent, by the Sublet catch below on the same build pair |
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3f894e1760` in `vdbeRecordCompareInt`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 30 void: probe site 65 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=3 witnesses=3`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
