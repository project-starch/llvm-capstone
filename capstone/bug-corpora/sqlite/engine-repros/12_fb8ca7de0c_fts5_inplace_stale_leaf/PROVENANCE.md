# fts5inplace -- fb8ca7de0c

Upstream fix `fb8ca7de0c`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row10 / sqlite-fb8ca7de0c -- fts5StructureAddLevel() edits an Fts5Structure in
 * place; when the structure is shared (nRef>1) with a scanning fts5vocab cursor,
 * the in-place write reallocs/corrupts memory the cursor still reads
 * (fts5_index.c:925, nRef at 339). Fixed 3.37.0. Ext: FTS5.
 * CONTROL: open an fts5vocab scan cursor on ft (holds a ref to the structure),
 * then INSERT rows into ft mid-scan to drive segment merges / add-level on the
 * shared structure. On unprotected capstone completes -> NOTRAP.
 * NOTE: build uses -DSQLITE_DQS=0, so SQL string literals MUST be single-quoted.
 * Heap stays at the 256K default: raising it inflates the in-image sqlite_heap[]
 * and the domain then exceeds the buddy allocator limit (create_dom failed).
```
