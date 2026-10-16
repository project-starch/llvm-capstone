# fz10 -- fz10

Upstream fix `FZ10`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz10 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     statDecodePage:177956, SEGV / out-of-bounds read
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

## Review 2026-10-16 -- the CheriBSD cell becomes a detection

### The four readings

| reading | result |
|---|---|
| host ASan | re-run: `heap-buffer-overflow` in `statDecodePage` at the cell-pointer walk, via `statNext <- statFilter` -- identical to the recorded bundle |
| virtual-malloc | **MISSED**, `RETURNED` -- against the physical `spatial` row's `cause 5` |
| virtual-nested-pools | **CAUGHT**, `cause=28` in `statDecodePage` |
| cheribsd-revocation | **CAUGHT**: `SIGPROT(34) si_code=1 PROT_CHERI_BOUNDS`, pc resolving to `statDecodePage+828` |

The CheriBSD reading changes for the reason set out in case 29's PROVENANCE and retracted in
case 22's: the 2026-10-06 reading was taken without the corrupt image present, so it was a
NO-READING rather than a miss. `si_code=1` is a bounds violation on a **valid** capability, not
one of the `si_code=2` tag faults the false positives in this review turned out to be.

### Why this one is visible at malloc granularity

Probe site 64 hooks all three unchecked reads in `statDecodePage`. On this image:

```
ENTER     aData=0x210220 aHdr=0x210220 iPgno=2
CELLWALK  iOff=52992 reads=0x21d120
LB_WITNESS kind=CAP-OOB n=1 tag=1 cb=0x1d8c00+262144 ca=0x21d120
<<FAULT>> signal=34 si_code=1 (PROT_CHERI_BOUNDS/valid-cap-out-of-bounds)
```

`cb=0x1d8c00+262144` is the **whole 256 KiB arena**, and `ca=0x21d120` is past its end at
`0x218c00`. The cell-pointer walk, with `iOff` reaching 64768 in eight iterations, leaves the
arena -- and an over-read that leaves the arena is visible even where bounds are at `malloc`
granularity. Case 32's over-read stays inside the arena and that arm is silent on it; Sublet
catches both, because it bounds each sub-allocation. **So the two arms are separated here by the
distance of the over-read, which is a property of the corrupt input, and not by the class of the
defect.**

The free-block chain never runs on this image, which is what distinguishes this case from 32.

### The two base builds disagree

As in case 29: the physical `spatial` row reports `cause 5` and the virtual-malloc build
completes. Base Capstone holds the arena as one capability, so the fault depends on where the
over-read lands relative to the arena end, which differs between builds.
