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
| virtual-nested-pools | **CAUGHT** | both controls as declared; `cause=28 pc=0x3f94e1257c` in `statDecodePage`, a declared fault site |
| cheribsd-revocation | **MISSED** | `RETURNED rc=0`; `poscontrol` exit 162 with `revocation=1` in the same session |

Reachability on CheriBSD is not inferred from the return code, which is what made the first reading
of case 30 void: probe site 64 was built into this arm's binary -- verified by `strings` to carry the
site's own payload, not just the runtime's `LB_ARMED` line -- and reports `hits=14 witnesses=13`, every witnessed read `kind=OOB`.
The defect site is entered and the access happens; the mechanism is silent because its bound is the
whole arena and the over-read stays inside it.

Both binaries were verified before the run: built after the change, carrying the embedded image
(`repro_memfs`) rather than `/root/corpus/*.db`, at the declared 1048576-byte arena, and matching
by sha256 in the guest.
