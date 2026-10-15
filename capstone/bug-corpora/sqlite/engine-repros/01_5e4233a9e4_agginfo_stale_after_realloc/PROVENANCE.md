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

## The trigger does not reach the defect (reviewed 2026-10-15)

The header's argument has two steps, and this trigger takes neither.

**The free.** `select.c:5069` is the `sqlite3ExprListDelete(db, pSub->pEList)` in
`countOfViewOptimization()`. In 3.22.0 that function rewrites only a single-column
`count(*)` whose FROM is one UNION ALL sub-select (`if( pSub->pPrior==0 ) return 0;`).
This SELECT has three result columns at the outer level and a plain view in the inner one,
so every call returns before the rewrite and nothing is deleted.

**The read.** `select.c:4752` is the definition line of `resetAccumulator()`. Inside it,
`pFunc->pExpr` is read only under `if( pFunc->iDistinct>=0 )`, that is, for a DISTINCT
aggregate. This SELECT has none. (`resolve.c:598`, cited for the op2 miscount, is an
`assert` in 3.22.0; the counter upstream changes is `pExpr->op2` in `resolveExprStep`.)

**Upstream.** `5e4233a9e4` is described as fixing "a spurious "misuse of aggregate
function" error"; its regression tests are error-message tests. That the miscount can also
lead to a use-after-free is this corpus's own inference from reading the code, and no run
has shown one.

**Evidence** (result lines; probe sites 34-36 in `shared/probe/`):

- Host ASan, stock 3.22.0 on the system allocator: no report, `agginfo rows=3`, `RETURNED`
  (as `results/2026-10-11-host-asan`).
- Site 34 (entry of `countOfViewOptimization`, prints the conditions it tests) fires 8 times;
  none has a compound sub-select in FROM. Site 35 (the line after the checks, i.e. the rewrite
  and its delete) `hits=0`. Site 36 (the `pFunc->pExpr` read in `resetAccumulator`) `hits=0`.
  Identical on native memsys5 and on virtual Capstone (stock memsys5).
- Positive controls for the probe, built from this case with only the SELECT replaced:
  `SELECT count(*) FROM (SELECT a FROM t1 UNION ALL SELECT b FROM t1)` takes site 35 once;
  `SELECT count(DISTINCT a) FROM t1` takes site 36 once with no witness.
- The recorded verdicts reproduce on the review platform: virtual-malloc MISSED and
  virtual-nested-pools MISSED, with both arms' controls as declared. Neither arm had anything
  to catch, so the MISSED on both, and on cheribsd-revocation, is a not-reached rather than a
  miss.

The physical `sublet` row in `results/matrix.tsv` records this case as detected (a VM halt at
kernel pc 0x800239e4). Since nothing is freed, that halt cannot have been a detection of this
defect. Proposed for withdrawal, with any other case found not reached, in one narrowing when
the review of this corpus ends.
