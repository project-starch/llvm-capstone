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

## Correction, 2026-10-26

The header above names `fts5StructureAddLevel()` from inspection. The free that host ASan reports for
this case is `fts5StructureExtendLevel()`'s realloc of one level's `aSeg` array (fts5_index.c:962,
called from `fts5FlushOneHash`), and the stale read is in `fts5SegIterNextPage`, 8 bytes into the
48-byte array (`results/2026-10-11-host-asan/results.json`). Both functions edit the shared structure
in place. Under memsys5's power-of-two rounding the array grows inside its block, so the address stays
(`results/2026-10-24-poisoncap/RESULTS.md`, "Rounding: 12"). `case.json` names the measured object
since this date. The case source is unchanged, so earlier builds stay comparable.
