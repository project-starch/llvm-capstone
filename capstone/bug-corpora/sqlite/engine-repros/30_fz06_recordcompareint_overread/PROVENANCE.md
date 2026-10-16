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
| cheribsd-revocation | **NO-READING**: probe site 65 reports `hits=0` -- the function is never entered |

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

### CheriBSD is a NO-READING, and that is a different thing

With the corrupt image restored (the arm had been running these cases without their images; see
case 26 and the retraction in case 22's PROVENANCE), this platform gives `s0_rc=0`, `s1_rc=11`,
and probe site 65 reports `hits=0`: `vdbeRecordCompareInt` is never entered. A different
statement sequence and a different build reject the image earlier, which is the case template's
own criterion -- *SQLITE_CORRUPT on the statement that should reach the defect means SQLite
rejected the image first, a real negative, not a silent pass*. Recorded as a NO-READING rather
than a miss.

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
