# rtreecursor -- c8c9cdd9dd

Upstream fix `c8c9cdd9dd`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-5 / sqlite-c8c9cdd9dd -- writing an R-Tree while a read cursor is open frees
 * RtreeNodes the cursor still references.
 *
 * rtreeUpdate() -> rtreeDeleteRowid() -> nodeRelease()/removeNode() sqlite3_free()s
 * RtreeNode structures that an *already open* rtree read cursor still holds in its
 * RtreeSearchPoint stack. When that cursor is stepped again, rtreeNext() ->
 * rtreeSearchPointPop() -> nodeRelease() reads pNode->nRef out of the freed block.
 *
 * Fixed in 3.24.0 by refusing the write outright: the fix counts active cursors and
 * returns the (then new) SQLITE_LOCKED_VTAB.  In 3.22.0 there is no such guard, so
 * the nested DELETE returns SQLITE_OK -- that rc=0 is the decisive evidence that
 * this build is the vulnerable one.
 *
 * Host ASan oracle on 3.22.0, identical shape:
 *   READ of size 4, 16 bytes into a 488-byte region
 *     nodeRelease <- rtreeSearchPointPop <- rtreeNext            (the use)
 *     rtreeDeleteRowid <- rtreeUpdate                            (the free)
 *     nodeAcquire <- rtreeNodeOfFirstSearchPoint <- rtreeFilter  (the alloc)
 *
 * PRAGMA page_size=512 keeps iNodeSize (page_size-64) small so 30 rows already build
 * a 2-level tree (t1_node=3); a 1-level tree frees nothing and would PASS vacuously.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```

## Review 2026-10-16 -- the defect is real, and two of the three detections are not

### The lifetime ender, named

The header says a write to the same R-Tree frees nodes the cursor holds. The free is more
specific than that, and worth naming because it is what makes the case real. `removeNode`
does **not** free:

```c
  assert( pNode->nRef==1 );          /* compiled out: NDEBUG is on by default */
  ...
  nodeHashDelete(pRtree, pNode);
  pNode->iNode = iHeight;
  pNode->pNext = pRtree->pDeleted;
  pNode->nRef++;                     /* onto the deleted list, refcount UP */
  pRtree->pDeleted = pNode;
```

The free is at the end of `rtreeDeleteRowid`, in the loop that reinserts the contents of
underfull nodes, and it ignores `nRef` entirely:

```c
  for(pLeaf=pRtree->pDeleted; pLeaf; pLeaf=pRtree->pDeleted){
    if( rc==SQLITE_OK ) rc = reinsertNodeContent(pRtree, pLeaf);
    pRtree->pDeleted = pLeaf->pNext;
    sqlite3_free(pLeaf);
  }
```

Probe site 54 prints the refcount at that line: `nRef_at_free=3` and `nRef_at_free=1` on two
of the four frees in one run. The cursor's reference is still outstanding. This matches the
recorded host-ASan free stack exactly (`sqlite3_free <- rtreeDeleteRowid <- rtreeUpdate`,
not through `nodeRelease`), and the access stack (`nodeRelease <- rtreeSearchPointPop <-
rtreeNext`, 16 bytes into a 488-byte region) is `pNode->nRef--` on the dead node.

### All three arms fault in the same place, and they do not mean the same thing

| arm | cause | pc | instruction | fault ADDRESS |
|---|---|---|---|---|
| virtual-malloc | 24 | `nodeRelease+144` | `lw a1, 0x18(a2)` | **`0x17`** |
| virtual-nested-pools | 25 | `nodeRelease+144` | `lw a1, 0x18(a2)` | `0x3f8577dda8` |
| cheribsd-revocation | SIGPROT(34) si_code=2 | `nodeRelease+0x2e` | `lw a0, 24(ca1)` | -- |

Function, pc offset and instruction all agree, in both virtual images, and `nodeRelease` is
this case's declared `fault_sites` entry, so the judge attributes every one of them. The
offset is 24 rather than 16 because a pointer is a 16-byte capability on both targets, which
puts `nRef` at 24; the instruction is the `pNode->nRef--` load in both.

### Why base Capstone and CheriBSD fault at all

Three conditions coincide here:

1. `RtreeNode`'s **first** field is `RtreeNode *pParent`, at offset 0.
2. memsys5 has no block header -- `MEM5LINK(idx)` is `&mem5.zPool[idx*szAtom]`, the very
   pointer it handed the caller -- so `memsys5Link` writes `Mem5Link{int next; int prev;}`
   over bytes 0..7. On both targets that is the low half of a 16-byte capability, and the
   integer stores **clear its tag**.
3. `nodeRelease` dereferences exactly that field: `if( pNode->pParent ) nodeRelease(pRtree,
   pNode->pParent);`

Site 54 prints the chain, the same on native memsys5 and on CheriBSD purecap:

```
FREE    p=0x223000 nRef_at_free=1 pParent=0x0
rtreecursor nested delete rc=0
LB_WITNESS kind=UAF off=0 n=4 req=512 blk=0x223000 tag=1 cb=0x1e5800+262144
READ    p=0x223000 nRef_read=1 pParent=0xffffffffffffffff iNode=0
RECURSE pParent=0xffffffffffffffff (about to be dereferenced)
LB_WITNESS kind=CAP-OOB n=4 tag=0 cb=0x0+18446744073709551615 ca=0xffffffffffffffff
READ    p=0xffffffffffffffff nRef_read=
<<FAULT>> signal=34 si_code=2 (PROT_CHERI_TAG/untagged-deref) pc=0x1d2072
```

`pParent` is `0x0` at the free and `0xffffffffffffffff` at the read: that value is
`memsys5Link`'s own terminator, `next = -1` / `prev = -1`, written into the block it just
linked. On base Capstone the same chain ends in `cause=24 address=0x17`, and `0x17` is
`(-1) + 24` -- arithmetically it can be nothing else.

The CheriBSD witness line is the strongest single reading here: at the **stale** load the
capability has `tag=1` and bounds `cb=0x1e5800+262144`, the whole 256 KiB `sqlite_heap[]`.
The defective access therefore goes through a fully valid capability and **succeeds**; it
decrements a refcount in dead memory and drives the `nRef==0` branch. Only the next witness
is untagged, and that is the `-1`.

So what fires on these two arms is the baseline capability **tag** check, on a pointer the
allocator manufactured, two frames after the memory-safety violation was committed. It is
not a lifetime check, not a bounds check, and not allocator-aware. It is also luck: had the
block been reissued -- memsys5's own default, and the reason cases 05, 11 and 14 are silent
on these arms -- offset 0 would hold a live object's tagged pointer and neither arm would say
anything. CheriBSD cannot catch this case in any configuration: `sqlite_heap[]` is a static
array (`repro322_common.h:34`) and `sqlite3_free` never calls `free()`, so revocation has
nothing to revoke.

### What Sublet reads

`cause=25` is the lifetime check, it fires at the stale load itself, and its fault address
`0x3f8577dda8` is inside the arena rather than a small integer. That is a detection of this
defect. The probe cannot corroborate it from the inside -- the Sublet patch rewrites the
memsys5 functions the hooks sit in, so `build-virtual.py --sublet` refuses the probed source
by design -- but the base and CheriBSD witnesses establish that the access is stale, and the
cause code and the address establish what Sublet rejected.

### Records corrected

* `base`: **detected -> not a detection.** Marked in `results/matrix.tsv`.
* `cheribsd-revocation`: **detected -> not a detection.** The signal is real and reproduced
  (`SIGPROT(34)`, `si_code=2`, exit 162, revocation knobs 1/0, positive control trapping);
  what it means is not. Marked in `matrix.tsv`.
* `sublet`: holds, but the **physical** row reports `cause 24`, which is this same `-1`
  dereference and not the lifetime check. Marked in `matrix.tsv`; this review relies on the
  virtual arm's `cause 25`.
* The old `status` line said cheribsd-revocation had no verdict. That was stale: the
  2026-10-05 and 2026-10-06 runs both record `FAULT SIGPROT(34)`.
* `fault_sites` stays `nodeRelease` -- it is where host ASan reported it -- but
  `fault_sites_why` now says why a function-level attribution is not enough for this case.

Host ASan could not be re-run: this host currently has no x86 ASan runtime. `case.c` has not
changed since `7517a8c2fd5e` (2026-10-11) and the amalgamation is the same `sha256 7fc92e19`,
so the 2026-10-11 reading is a reading of this source.
