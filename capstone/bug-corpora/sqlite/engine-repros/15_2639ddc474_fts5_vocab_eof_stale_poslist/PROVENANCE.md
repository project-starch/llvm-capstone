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

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy created fts5vocab('ft','row'), whose next-method never enters fts5VocabInstanceNext; host ASan was silent on it. The CheriBSD copy uses 'instance', and host ASan reports the heap-use-after-free in sqlite3Fts5PoslistNext64 <- fts5VocabInstanceNext. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).
