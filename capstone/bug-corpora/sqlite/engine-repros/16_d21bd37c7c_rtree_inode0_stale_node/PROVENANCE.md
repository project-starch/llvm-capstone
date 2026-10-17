# rtreeinode0 -- d21bd37c7c

Upstream fix `d21bd37c7c`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-16 / sqlite-d21bd37c7c -- an rtree node inserted into the node hash with
 * iNode==0 is freed but can never be unlinked.
 *
 * nodeWrite() in 3.22.0:
 *     sqlite3_step(p); rc = sqlite3_reset(p);
 *     if( pNode->iNode==0 && rc==SQLITE_OK ){
 *       pNode->iNode = sqlite3_last_insert_rowid(pRtree->db);
 *       nodeHashInsert(pRtree, pNode);          <-- no check for iNode==0
 *     }
 * If the %_node INSERT is suppressed (a BEFORE INSERT trigger raising IGNORE) and
 * no other rowid insert has happened on the connection, last_insert_rowid() is 0.
 * The node is then linked into aHash[0] still carrying iNode==0.  nodeHashDelete()
 * keys off iNode and treats 0 as "not in the hash", so nodeRelease() frees the block
 * and the hash keeps a dangling entry.  The next nodeAcquire() walks that bucket and
 * reads p->iNode out of freed memory.  Fixed on trunk 2026-09-24 (expected 3.54.0)
 * by returning SQLITE_CORRUPT_VTAB when last_insert_rowid() is 0.
 *
 * Host ASan oracle on 3.22.0, identical shape:
 *   READ of size 8, 8 bytes into a 1000-byte region   (RtreeNode.iNode)
 *     nodeHashLookup <- nodeAcquire <- rtreeFilter            (the use)
 *     nodeRelease <- SplitNode <- rtreeInsertCell             (the free)
 *     nodeNew <- SplitNode <- rtreeInsertCell                 (the alloc)
 *
 * Two requirements and how they are met in a freestanding :memory: domain:
 *  1) A connection whose node hash is EMPTY when the suppressed write happens.
 *     Upstream does `db close; sqlite3 db test.db`, which this domain cannot do --
 *     the VFS xOpen always returns SQLITE_CANTOPEN.  Instead this uses the row-12
 *     trick: shared cache plus a SECOND connection on file::memory:?cache=shared.
 *     Each connection builds its own Rtree object, so db2 starts with an empty
 *     aHash[].  (A :memory: database, shared-cache URI included, takes SQLite's
 *     memDb path and never calls xOpen.)  Needs -USQLITE_OMIT_SHARED_CACHE.
 *  2) A node filled exactly to capacity, so the next INSERT splits it.
 *     iNodeSize is page_size-64 and a 5-column cell is 24 bytes, so page_size=1024
 *     gives (1024-64-4)/24 = 39 cells.  Hence exactly 39 rows.  (At the default
 *     4096 the per-node cap RTREE_MAXCELLS=51 applies instead -- that is why
 *     upstream's test uses 51.)  An off-by-one row count does NOT trigger.
 *
 * `zero` is an empty INTEGER PRIMARY KEY table: it exists only so that nothing on
 * the connection has ever set a rowid, keeping last_insert_rowid() at 0.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```

## Review 2026-10-16 -- the defect is worse than recorded, and two of the three detections are not

### The defect, measured as a boolean

Probe site 55 puts a hook at each step of the mechanism the header describes. One run:

```
HASHINS p=0x20d000 iNode=0 bucket=0 <- iNode==0, linked anyway
HASHINS p=0x20d400 iNode=0 bucket=0 <- iNode==0, linked anyway
DELSKIP p=0x20d000 iNode=0 <- NOT unlinked (iNode==0)
FREE    p=0x20d000 iNode=0 still_in_bucket=0
DELSKIP p=0x20d400 iNode=0 <- NOT unlinked (iNode==0)
FREE    p=0x20d400 iNode=0 still_in_bucket=1
DELSKIP p=0x20cc00 iNode=1 unlinked
FREE    p=0x20cc00 iNode=1 still_in_bucket=0
```

`still_in_bucket=1` is the defect: the block returns to the pool while it is still the head of
`aHash[0]`. The third line is the control within the run -- a node with a real `iNode` is
unlinked normally.

### It is not only a stale read

```
before dangling-hash query
LB_WITNESS kind=UAF off=0 n=8 req=1024 blk=0x20d400 tag=1 cb=0x1e5c00+262144
LOOKUP  want=0 bucket=0 head=0x20d400 head_iNode=0
LB_WITNESS kind=UAF off=0 n=4 req=1024 blk=0x20d400 tag=1 cb=0x1e5c00+262144
HANDOUT p=0x20d400 iNode=0 nRef_read=0 <- returned as live
```

The bucket walk matches the dead node, because its `iNode` is 0 and 0 is what is being looked
up, and `nodeAcquire` then hands it back out as a live node -- which also **writes** `nRef++`
into the freed block. From there the program is operating on a freed object as if it were
alive. The record's "reads `p->iNode` out of freed memory" understates it.

### Why neither non-Sublet arm can see any of that

Under the 16-byte-pointer layout `iNode` is at offset 16 and `nRef` at offset 24, and
`Mem5Link` is 8 bytes at offset 0, so the free touches neither field. Both witnesses above
carry `tag=1` and `cb=0x1e5c00+262144`: a capability bounded to the whole 256 KiB
`sqlite_heap[]`. There is no tag to check and no bound to cross, on either arm.

### What they do fault on: case 15's artifact, reached by a longer route

| arm | verdict | cause | pc | fault address |
|---|---|---|---|---|
| virtual-malloc | NO-READING:unattributed | 24 | `nodeRelease+144` | **`0x17`** |
| virtual-nested-pools | CAUGHT | 25 | `nodeHashLookup+240` | `0x3fafb1fe20` |
| cheribsd-revocation | -- | SIGPROT(34) si_code=2 | `nodeRelease+0x2e` | -- |

The dead node, now held with `nRef=1`, is eventually released. `nodeRelease` does `nRef--` to
0; `iNode` is 0 so the `iNode==1` branch is skipped; then `if( pNode->pParent )` reads offset
0 -- the one field `Mem5Link` overwrote, tag cleared -- and recurses into it. Base Capstone's
fault address is `0x17`, which is `(-1) + 24`: the same value, the same instruction and the
same pc offset as case 15. CheriBSD's pc resolves to `nodeRelease+0x2e`, also the same
instruction. Site 55 does not print `pParent`; `0x17` admits no other reading, and site 54
printed the `0xffffffffffffffff` directly in case 15.

Two things make the contingency explicit:

* `nodeAcquire` would have removed even that fault had the terminator been 0. Its hash-hit
  path is `if( pParent && !pNode->pParent ){ ... pNode->pParent = pParent; }`, and `-1` is
  non-zero, so the assignment is skipped and the `-1` survives into the release. With a `0`
  terminator both arms would have completed silently.
* `nodeRelease` is deliberately **not** in this case's `fault_sites`, so the judge already
  calls the base arm `NO-READING:unattributed` -- and did so in the 2026-10-11 bundle. The
  `cheribsd-revocation` cell was never held to the same rule, and `results/matrix.tsv` has
  recorded it as `detected` ever since. That disagreement between the physical matrix and the
  virtual verdicts went unflagged.

### Records corrected

* `base`: **detected -> not a detection**, which only brings `matrix.tsv` into line with the
  verdict the pipeline already produced.
* `cheribsd-revocation`: **detected -> not a detection.**
* `sublet`: holds, on the virtual arm. The physical row is one of the 28 halts at kernel pc
  0x800239e4 and is marked.
* All preconditions were confirmed in every run: `inserted=39`, `nodes before=1`,
  `trigger rc=0`, `split insert rc=0`, `db2 nodes after=1`.

Host ASan could not be re-run (no x86 ASan runtime on this host); `case.c` is unchanged since
`7517a8c2fd5e` and the amalgamation is the same `sha256 7fc92e19`.

---

## Cross-case: the non-Sublet arms' false positives in this corpus

Recorded here because case 16 is the second instance of the same mechanism and the pattern is
now the thing a reader needs, not the individual row. Every entry has the same shape: **the
defective access itself succeeds on base Capstone and on CheriBSD**, because it is inside the
`memsys5` arena -- in bounds, and through a capability whose tag is intact -- and what trips is
a *later consequence* of having let it succeed. The cells were being credited for the
consequence.

| class | what actually faults | cases | signature |
|---|---|---|---|
| A. downstream NULL dereference | the stale read returns 0 or a zeroed structure and the program dereferences it | 02, 05 | fault at a NULL-ish address, in a function downstream of the defect |
| A'. capability arithmetic over a non-capability | nothing is even loaded: `return &z[1]` with `z` NULL | 12 | `cause 24 address=0x0` |
| B. the allocator's own in-band metadata read back as a pointer | `memsys5Link` writes `Mem5Link{next=-1,prev=-1}` over the first 8 bytes of the freed block; a pointer field living there is dereferenced | 19, 20 | `cause 24` / SIGPROT `si_code=2` at `nodeRelease`, fault **address `0x17`** = (-1)+24 |

CORRECTION, same day: an earlier version of this table carried a fourth class and put case 04
in it, as "the program's own assert aborts on base". That is wrong as a false positive -- case 04's base and cheribsd-revocation rows were recorded `not-detected` in the first place, so no
cell was removed there. What the review added for case 04 is why each arm is silent, and that
the two reasons differ: on base the program aborts on its own `assert()` and produces no
reading at all, while CheriBSD compiles the assert out under `-DNDEBUG` and completes. That
belongs next to this table, as the other way the arms fail to be comparable, not in it.

Class B is specific to a struct whose **first** field is a pointer that the freeing path later
dereferences, which is why it shows up twice in the rtree family and nowhere else so far. It is
also the phenomenon case 18 (withdrawn, campaign numbering) was withdrawn for demonstrating -- in-band metadata overwriting
caller data -- turning up inside real cases as a false positive for the two arms that are
supposed to be blind to this layer.

Counted from `results/matrix.tsv`, the physical campaign recorded **eight** base detections
(02 05 12 19 20 22 29 31) and **five** CheriBSD detections (02 05 19 20 22). Of those, every
one that has been reviewed has turned out not to be a detection: **5 of 5 on base, 4 of 4 on
CheriBSD**. The three still standing are the three not yet looked at -- **22**
(`fix-2026-07-26`, temporal, on both arms' lists, and the last `D D D` row), **29** (`fz02`,
spatial, base only) and **31** (`fz10`, spatial, base only). Case 22 is the first to check for
class B.

A standalone version of this record, with the per-case detail and the commit for each
correction, is kept outside the repository alongside the three-arm tables.
