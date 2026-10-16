# fz11 -- fz11

Upstream fix `FZ11`. Collected in round R2 (fuzz-corpus diff).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
fz11 -- R2 fuzz-corpus diff image, trigger re-derived 2026-10-03.
 *
 * Host ASan on stock 3.22.0 (x86):
 *     statDecodePage:177925, heap-buffer-overflow
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

## Review 2026-10-16 -- holds in full, and it is the control for what CheriBSD can see

### The four readings

| reading | result |
|---|---|
| host ASan | re-run: `heap-buffer-overflow` in `statDecodePage` at the free-block chain line, via `statNext <- statNext` -- identical to the recorded bundle |
| virtual-malloc | **MISSED**, `RETURNED` |
| virtual-nested-pools | **CAUGHT**, `cause=28` in `statDecodePage` |
| cheribsd-revocation | **MISSED**, and this time measured: the image is loaded, the defect is reached, and the arm is silent |

The earlier CheriBSD reading for this case was taken without the image present and was therefore
a NO-READING; this one is a miss that was actually measured. The run then ends in `SQLITE_NOMEM`
because the 256 KiB arena cannot complete the dbstat scan, which happens **after** the defect
site is reached.

### The defect, and what separates it from case 31

```c
iOff = get2byte(&aHdr[1]);
while( iOff ){
  nUnused += get2byte(&aData[iOff+2]);
  iOff = get2byte(&aData[iOff]);
}
```

No bound on `iOff` and no cycle detection. Probe site 64 hooks all three unchecked reads in
`statDecodePage`, and this image takes the **chain**: one call with `iOff=17234`, while its
cell-pointer walk stays within range at 506. Case 31 is the mirror -- the cell walk to 64768,
the chain never entered.

### This case is the control for the CheriBSD column

Both are over-reads in one function; what differs is how far they go. Case 31's walk leaves the
**whole 256 KiB arena**, so CheriBSD's `malloc`-granularity bounds catch it. This case's stays
inside the arena, and that arm is silent. Sublet catches both, because it bounds each
sub-allocation rather than the pool.

So the two arms are not separated here by the class of the defect but by the **distance** of the
over-read, which is a property of the corrupt input. That is the honest form of the claim the
spatial half of this corpus supports: CheriBSD sees a nested over-read only when it happens to
leave the arena, and nothing in the defect decides whether it does.
