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
