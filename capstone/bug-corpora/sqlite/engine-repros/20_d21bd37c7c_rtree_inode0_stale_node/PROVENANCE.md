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
