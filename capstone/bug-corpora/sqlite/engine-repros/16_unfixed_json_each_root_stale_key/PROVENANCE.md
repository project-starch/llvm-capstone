# jsoneachroot -- unfixed

Upstream fix `(unfixed upstream)`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
NEW-2 / json_each root/path column -- the SIBLING the 3.45.2 fix left behind,
 * STILL UNFIXED UPSTREAM.
 *
 * jsonEachColumn()'s default arm returns the cursor's own buffer as SQLITE_STATIC,
 * exactly like the JEACH_JSON arm did before 3.45.2:
 *
 *     default: {                                    (3.22.0 ext/misc/json1.c:2139)
 *       const char *zRoot = p->zRoot;
 *       if( zRoot==0 ) zRoot = "$";
 *       sqlite3_result_text(ctx, zRoot, -1, SQLITE_STATIC);
 *
 *     jsonEachFilter():      p->zRoot = sqlite3_malloc64(n+1)     (:2244)
 *     jsonEachCursorReset(): sqlite3_free(p->zRoot)               (:1963)
 *
 * The 3.45.2 fix (sqlite-28001204f4) converted ONLY JEACH_JSON to SQLITE_TRANSIENT;
 * current trunk still returns p->path.zBuf with SQLITE_STATIC here, so this arm is
 * live upstream today. zRoot is non-NULL only for the TWO-argument form
 * json_each(X, PATH), which is why the SQL below passes a path.
 *
 * Same trigger as case_json_each_static.c: cross TWO json_each calls with LITERAL
 * arguments so the INNER cursor is re-filtered once per outer row. A table-valued
 * function whose argument is a COLUMN reference yields zero rows in this domain
 * (see case_jsondiag.c), so the host's join-on-a-column shape cannot be used.
 *
 * RUNTIME-CONFIRMED on host SQLite 3.22.0 under ASan with this exact SQL:
 *   heap-use-after-free in memcmp <- binCollFunc <- vdbeCompareMemString <- minmaxStep.
 * max(j2.path) and the json_tree form fire identically.
 *
 * Expected on unprotected Capstone: the freed bytes are compared as a STRING (memcmp),
 * not dereferenced as a capability, so no tag check is tripped -> silent NOTRAP.
 * NOTE -DSQLITE_DQS=0: SQL string literals must be single-quoted (JSON keys keep their
 * double quotes INSIDE the single-quoted SQL literal).
 * Build in the json group (-DSQLITE_ENABLE_JSON1).
```
