# rtreestatic -- eab0e10304

Upstream fix `eab0e10304`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-3 (rtree arm) / sqlite-eab0e10304 -- rtree leaves a freed heap buffer bound
 * SQLITE_STATIC to a persistent cached statement.
 *
 * nodeNew() allocates the node and its payload as ONE block:
 *     pNode = sqlite3_malloc(sizeof(RtreeNode) + pRtree->iNodeSize);
 *     pNode->zData = (u8 *)&pNode[1];
 * and nodeWrite() binds that payload into the cached pWriteNode statement without
 * ever binding it back to NULL:
 *     sqlite3_bind_blob(p, 2, pNode->zData, pRtree->iNodeSize, SQLITE_STATIC);
 *     sqlite3_step(p);
 *     rc = sqlite3_reset(p);
 * sqlite3_reset() does not clear bindings, so when nodeRelease() does
 * sqlite3_free(pNode) the statement's parameter 2 dangles into the freed block.
 * Fixed 3.23.0 (16 days after 3.22.0 shipped) by adding sqlite3_bind_null(p, 2).
 *
 * This is the SECOND arm of the same defect; case_fts3_static_bind.c is the fts3 arm.
 * Both use the route the fix's own commit message names -- "obtain a pointer to the
 * persistent statement using sqlite3_next_stmt() and attempt to access the freed
 * buffer using sqlite3_expanded_sql() or similar" -- with the "or similar" variant,
 * re-STEPping the statement, because sqlite3_expanded_sql() is compiled out in this
 * port: -DSQLITE_OMIT_FLOATING_POINT=1 makes 3.22.0's sqliteInt.h also
 * `#define SQLITE_OMIT_TRACE 1`, so the function body is `return 0;`. See
 * case_fts3_static_bind.c for the full write-up of that.
 *
 * Host ASan oracle on 3.22.0 (page_size=1024, 120 rows), identical under
 * -DSQLITE_RTREE_INT_ONLY which this group uses:
 *   re-step  : READ of size 820, 40 bytes into a freed 864-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1   (upstream's route, for comparison)
 *     free  nodeRelease <- rtreeUpdate
 *     alloc nodeAcquire
 *   Offset 40 is exactly sizeof(RtreeNode) on the host, i.e. &pNode[1] == zData,
 *   which confirms the single-block layout is what is being read.
 *
 * WHY THIS ARM IS WORTH HAVING ALONGSIDE THE fts3 ONE: its consequence is far
 * sharper. The re-step writes the freed block's current contents over the node's
 * row in %_node, and a later full scan then simply LOSES the rows that lived in
 * that node -- quantitative, observable data loss with no memory-safety tool:
 *     full_scan_before=120 -> full_scan_after=71      (49 rows gone)
 *     node data: same length 820, sum changed, first4 0x00000031 -> 0x00000000
 * Note PRAGMA integrity_check returns ok here: it does not validate rtree shadow
 * tables, so the row count is the signal, not integrity_check.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```
