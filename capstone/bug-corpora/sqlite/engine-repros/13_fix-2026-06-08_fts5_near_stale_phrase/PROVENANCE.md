# fts5near -- fix-2026-06-08

Upstream fix **`b677c5afd4`**, 2026-06-08. Collected in round R1 (NVD and upstream history).

"Avoid a potential use-after-free in fts5. Report [bugs:/info/2026-06-08T08:45:27Z]", shipped
with `ext/fts5/test/fts5near.test`. The directory's token is the fix's DATE, which could not be
looked up; `case.json` now carries the commit in `upstream_fix` and the date in `dir_token`.

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row22 / sqlite-2026-06-08 (b677c5afd4) -- fts5ExprNearIsMatch reads a lookahead
 * reader from each poslist buffer while WriterAppend rewrites the same buffer in
 * place; the append may realloc, so the reader reads freed memory (fts5_expr.c
 * 560/596/627). Fixed 3.53.3. CONTROL: run a NEAR/phrase query that drives
 * fts5ExprNearIsMatch. On unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts5 group.
```
