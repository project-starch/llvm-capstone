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

## 2026-10-16: re-measured, and the record held in full

Nothing needed correcting here. After cases 02 and 07 this is the third case in this review whose
object, lifetime ender, trigger, reachability contract, fault site and all three arm verdicts came
back as recorded.

Host ASan on this platform:

```
READ of size 1, 39 bytes inside a freed 64-byte region
  use   sqlite3Fts5PoslistNext64 sqlite3.c:189645 <- fts5VocabInstanceNext sqlite3.c:206968
          <- fts5VocabNextMethod
  free  fts5DataRelease <- fts5SegIterNextPage <- fts5SegIterNext <- fts5MultiIterNext
          <- sqlite3Fts5IterNextScan
```

`fts5VocabInstanceNext`'s loop condition is
`sqlite3Fts5PoslistNext64(pIter->pData, pIter->nData, po, pp)`, and the body's
`sqlite3Fts5IterNextScan` releases the leaf that data lives in at the EOF transition. The 3.26.0
fix adds the break after that step.

Probe site 51 marks three points: the pointer the loop condition reads, printed before the step
that may release its page; the release of the leaf inside `fts5SegIterNextPage`; and the pointer
again on the way out, where it is judged by address. They fall in this order on every platform:

```
STEP    pData=0x…efd57 nData=1 off=0 bEof=0
RELEASE leaf p=0x…efd30 nn=26
LB_WITNESS kind=UAF off=39 n=1 req=64 blk=0x…efd30 ca=0x…efd57
EXIT    pData=0x…efd57 bEof=1
```

`ca - blk = 39`: the pointer is inside the leaf the step had just released, which is host ASan's
offset and region size read off the pointers instead of the sanitiser.

| platform | witness |
| --- | --- |
| native memsys5 | `kind=UAF off=39 n=1 req=64` |
| virtual Capstone, lookaside off | `kind=UAF off=55 n=1 req=128` |
| CheriBSD purecap | `kind=UAF off=55 n=1 req=78 tag=1 cb=<arena>+262144` |

The offset differs because the page layout does; `hits=7 witnesses=1` on all three, which
reproduces the 2026-10-06 reading exactly. The CheriBSD line also measures why that arm is silent:
the pointer is tagged and bounded to the whole 256 KiB memsys5 arena.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; instance_rows=6, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3fb02374cc in sqlite3Fts5PoslistNext64`, attributed by function |
| cheribsd-revocation | MISSED | positive control traps; instance_rows=6, RETURNED rc=0 |

The case's own reachability contract held in every run: the instance vocab yields 6 rows, where
'row' and 'col' yield 4 and never enter the buggy function. No count in the study's tables moves;
the physical sublet row, one of the 28 kernel-pc halts, is marked in `results/matrix.tsv`.
