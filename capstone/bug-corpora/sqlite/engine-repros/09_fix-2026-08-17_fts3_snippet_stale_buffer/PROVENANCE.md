# fts3snip -- fix-2026-08-17

Upstream fix `2026-08-17`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row23 / sqlite-2026-08-17 (a87bd2471a9c) -- fts3BestSnippet() saves poslist
 * pointers (pHead/pTail) per phrase; a later incremental phrase under an OR restarts
 * the NEAR group (fts3EvalRestart, fts3.c:5540), freeing/reloading already-visited
 * doclists, so the saved pointers dangle while scoring (fts3_snippet.c:507/479).
 * Distinct from becd68ba0d. Trunk-only fix. CONTROL: snippet() over an OR of phrases
 * read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```
