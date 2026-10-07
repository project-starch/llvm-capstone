# fts3offsets -- 51dd67080a

Upstream fix `51dd67080a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row13 / sqlite-51dd67080a -- sqlite3Fts3Offsets() stores a poslist pointer per
 * phrase, then an incremental phrase triggers a NEAR-group restart (fts3EvalRestart,
 * fts3.c:5540) that frees/reloads already-visited doclists, dangling the saved
 * pointers (fts3_snippet.c:1544). Fixed 3.49.0. CONTROL: offsets() over an OR of
 * phrases read incrementally; on unprotected capstone completes -> NOTRAP.
 * NOTE: build in fts3 group.
```
