# fts3snipor -- becd68ba0d

Upstream fix `becd68ba0d`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row6 / sqlite-becd68ba0d -- fts3EvalNextRow() nested-OR branch keeps evaluating
 * phrase nodes whose doclist was freed because bEof was not set on the exhausted
 * side (fts3.c:5134). Fixed 3.32.0. CONTROL: snippet() over an expression with
 * nested OR phrases; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy queried a flat OR and returned 3 rows; host ASan was silent on it. The CheriBSD copy runs upstream's fts3snippet2.test 2.1/2.2 nested-OR query, checks the snippet value and warns when no row comes back, and host ASan reports the heap-use-after-free in fts3SnippetAdvance. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).
