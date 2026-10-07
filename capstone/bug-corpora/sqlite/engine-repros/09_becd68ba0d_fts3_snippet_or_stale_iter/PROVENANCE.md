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
