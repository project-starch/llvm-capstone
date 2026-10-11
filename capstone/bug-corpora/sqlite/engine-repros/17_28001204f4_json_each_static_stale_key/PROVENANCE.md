# jsoneachstatic -- 28001204f4

Upstream fix `28001204f4`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
NEW-1 / sqlite-28001204f4 -- json_each/json_tree hand the SQL layer their .json
 * column as SQLITE_STATIC, pointing at the cursor's OWN malloc'd copy of the input:
 *
 *     case JEACH_JSON:                                     (json1.c, 3.22.0)
 *       sqlite3_result_text(ctx, p->sParse.zJson, -1, SQLITE_STATIC);
 *
 *     jsonEachFilter():      p->zJson = sqlite3_malloc64(n+1)
 *     jsonEachCursorReset(): sqlite3_free(p->zJson)
 *
 * SQLITE_STATIC means no copy is made, and sqlite3VdbeMemCopy() keeps MEM_Static
 * as-is, so a min()/max() accumulator retains the raw pointer. Driving json_each
 * from the right of a join re-filters it per outer row; each jsonEachFilter calls
 * jsonEachCursorReset and frees the PREVIOUS zJson while the accumulator still
 * points into it. minmaxStep() then compares against freed memory.
 * Fixed 3.45.2 (SQLITE_TRANSIENT). Ext: JSON1.
 *
 * RUNTIME-CONFIRMED on host SQLite 3.22.0 under ASan with exactly this SQL:
 *   heap-use-after-free in minmaxStep -> vdbeCompareMemString -> binCollFunc.
 * group_concat does NOT fire (it copies); min/max do.
 *
 * Expected on unprotected Capstone: the freed bytes are compared as a STRING
 * (memcmp), not dereferenced as a capability, so no tag check is tripped and the
 * read should be SILENT -> NOTRAP. That silence is the control-arm result.
 * NOTE -DSQLITE_DQS=0: SQL string literals must be single-quoted.
 * Build in the json group (-DSQLITE_ENABLE_JSON1).
```
