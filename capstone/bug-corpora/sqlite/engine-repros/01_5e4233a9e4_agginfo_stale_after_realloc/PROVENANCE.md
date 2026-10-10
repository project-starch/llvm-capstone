# agginfo -- 5e4233a9e4

Upstream fix `5e4233a9e4`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row17 / sqlite-5e4233a9e4 -- Use-after-free on AggInfo function expressions:
 * an aggregate inside a FROM sub-select is attributed to the wrong AggInfo (op2
 * miscount at resolve.c:598); countOfViewOptimization() (select.c) deletes the
 * ExprList owning those Expr nodes (5069 sqlite3ExprListDelete) while AggInfo
 * still points into them, and resetAccumulator() (4752) reads pF->pExpr after
 * the free. Needs -DSQLITE_COUNTOFVIEW_OPTIMIZATION for the free site. Fixed 3.45.0.
 *
 * CONTROL arm: run the SELECT that triggers countOfViewOptimization over a view
 * with a count() so the optimizer rewrites it. On unprotected Capstone the freed
 * Expr read is not caught, so the query returns rows / an error but the domain
 * RETURNS. Host ASan build flags the heap-use-after-free.
 *
 * Trigger shape (from the forum post c9970a37ed / chromium 41487453): a
 * SELECT count(*) FROM (SELECT ...) that the count-of-view optimization rewrites,
 * with an aggregate that the resolver mis-attributes across the sub-select level.
```
