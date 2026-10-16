# fts5inplace -- fb8ca7de0c

Upstream fix `fb8ca7de0c`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row10 / sqlite-fb8ca7de0c -- fts5StructureAddLevel() edits an Fts5Structure in
 * place; when the structure is shared (nRef>1) with a scanning fts5vocab cursor,
 * the in-place write reallocs/corrupts memory the cursor still reads
 * (fts5_index.c:925, nRef at 339). Fixed 3.37.0. Ext: FTS5.
 * CONTROL: open an fts5vocab scan cursor on ft (holds a ref to the structure),
 * then INSERT rows into ft mid-scan to drive segment merges / add-level on the
 * shared structure. On unprotected capstone completes -> NOTRAP.
 * NOTE: build uses -DSQLITE_DQS=0, so SQL string literals MUST be single-quoted.
 * Heap stays at the 256K default: raising it inflates the in-image sqlite_heap[]
 * and the domain then exceeds the buddy allocator limit (create_dom failed).
```

## 2026-10-16: re-measured; the object, the freeing function and the base arm's fault all corrected

The defect and the Sublet verdict stand. Three records do not.

**The freed object and the function that frees it.** The case named
`fts5StructureAddLevel` (ext/fts5/fts5_index.c:925), which reallocs the whole `Fts5Structure`.
What this trigger reaches is its sibling, `fts5StructureExtendLevel` (sqlite3.c:195165):

```c
nByte = (pLvl->nSeg + nExtra) * sizeof(Fts5StructureSegment);
aNew = sqlite3_realloc(pLvl->aSeg, nByte);
...
pLvl->aSeg = aNew;
```

so the object is **one level's `aSeg[]` array**, not the structure as a whole. Host ASan on this
platform:

```
READ of size 1, 8 bytes inside a freed 48-byte region
  use   fts5SegIterNextPage sqlite3.c:195736 <- fts5SegIterNext <- fts5MultiIterNext
          <- sqlite3Fts5IterNextScan <- fts5VocabNextMethod
  free  realloc <- sqlite3MemRealloc <- sqlite3_realloc <- fts5StructureExtendLevel
          <- fts5FlushOneHash <- fts5IndexFlush <- sqlite3Fts5IndexSync <- fts5SyncMethod
          <- sqlite3VtabSync <- vdbeCommit
  alloc sqlite3Fts5MallocZero <- fts5StructureDecode <- fts5StructureRead
          <- sqlite3Fts5IndexQuery <- fts5VocabFilterMethod
```

48 bytes is four `Fts5StructureSegment` entries of 12 bytes, the read is `pSeg->pgnoLast` at
offset 8 of one of them, and the allocation is the structure the vocab cursor read at xFilter
time. The defect class is still the one the case names -- an in-place edit of a structure whose
`nRef` is greater than one -- and `nRef` is still the struct's first field at fts5_index.c:339.

**The defect has two faces, which is why the arms disagree.** Probe site 48 prints the old and the
new `aSeg` pointer at every extend, and the read:

* when the realloc **moves** the array, the cursor's `pSeg` dangles -- a use-after-free. On native
  memsys5 that happens in 4 of 30 extend calls, and 10 of 40 reads land in an array a move had
  already freed;
* when the realloc **keeps** the block, the same function's `memmove(&aNew[nExtra], aNew, nMove)`
  and `memset(aNew, 0, ...)` shift the entries up and zero the first ones underneath the cursor.
  That is not an invalid access, but it is where the unprotected arms' downstream fault comes from.

**The base arm is not a detection.** virtual-malloc faults with `cause=24 address=0x0` in
`sqlite3Fts5IterTerm`, whose body ends

```c
const char *z = (const char*)fts5MultiIterTerm((Fts5Iter*)pIndexIter, &n);
*pn = n-1;
return &z[1];
```

With `z` NULL that last line is capability arithmetic over a non-capability: on Capstone it faults
before anything is loaded. So the fault is not a read of the defect at all, it is the second face
above working through to a NULL term. The judge already records NO-READING:unattributed for this
arm; the physical `spatial: detected` row is marked in `results/matrix.tsv`, and the physical
`sublet: detected` row is one of the 28 kernel-pc halts.

**The probe's `kind=OOB` is a misclassification, not a spatial defect.** On CheriBSD site 48
witnesses the stale read twice and reports

```
LB_WITNESS site=48 kind=OOB off=188 n=4 req=180 blk=0x243000 tag=1 cb=<arena>+262144 ca=0x2430bc
hits=40 witnesses=2
```

`off=188` in a block whose **requested** size is 180: memsys5 freed the old array and reissued the
block to a smaller object, so the stale pointer now sits past that new tenant's bytes and an
address oracle calls it an over-read. This reproduces the 2026-10-06 reading (kind=OOB, hits=40,
witnesses=2) and explains it. Host ASan, which quarantines rather than reissues, calls the same
access `heap-use-after-free`, so the axis stays **temporal**. This is the third form of the blind
spot case 07 documented: not a missing witness, but a witness with the wrong kind.

The same line measures why CheriBSD misses it: `tag=1` and `cb=<arena>+262144` -- the pointer is
tagged and bounded to the whole 256 KiB memsys5 arena, so the access is in bounds and the realloc's
free is nothing revocation can see.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | NO-READING:unattributed | both controls as declared; `cause=24 address=0x0` in `sqlite3Fts5IterTerm`, not the declared site |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3f89e2f350 in fts5SegIterNextPage`, attributed by function |
| cheribsd-revocation | MISSED | positive control traps; `rows=5`, RETURNED rc=0, with the probe witness above |

Every run reached the mid-scan state (`mid-scan inserts ok=24`). No verdict changes; the base
arm's cell in the study's tables does, from detected to reached-and-silent, which narrows the gap
the study reports.
