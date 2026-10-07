# fts3zterm -- dee0359ddb

Upstream fix `dee0359ddb`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row11 / sqlite-dee0359ddb -- fts3SegReaderNext() points pReader->zTerm straight
 * into the pending-terms hash key memory (fts3_write.c:1333); an optimize() during
 * the scan frees that hash, so zTerm dangles. Fixed 3.37.0. CONTROL: keep a
 * non-empty pending hash, open a SELECT scan, run optimize() mid-scan.
 * NOTE: build in fts3 group.
```
