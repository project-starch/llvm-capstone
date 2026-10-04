# fts5vocabeof -- 2639ddc474

Upstream fix `2639ddc474`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row1 / sqlite-2639ddc474 -- fts5VocabInstanceNext() keeps stepping the fts5
 * iterator after EOF, dereferencing an iterator whose backing data was released
 * (fts5_vocab.c:427). Fixed 3.26.0. CONTROL: full-scan an fts5vocab table so the
 * EOF transition runs; on unprotected capstone it completes -> NOTRAP.
 * Public, already-fixed bug; collected for the temporal-safety study.
 * NOTE: place beside repro322_common.h in ports/sqlite/repro322/ to build.
```
