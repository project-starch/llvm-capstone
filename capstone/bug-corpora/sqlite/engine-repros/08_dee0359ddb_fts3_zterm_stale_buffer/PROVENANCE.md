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

## 2026-10-16: the trigger was replaced and the defect is now reached

The 2026-10-11 note had the first obstacle right and stopped one short of two more.

**What was right.** Each INSERT in the old trigger auto-commits, and every commit runs
`sqlite3Fts3PendingTermsFlush`, which calls `sqlite3Fts3PendingTermsClear`. So the pending-terms
hash was already empty when the SELECT opened, no `zTerm` ever aliased a hash key, and the mid-scan
`optimize()` cleared nothing anyone held.

**What was missing.** Putting the inserts inside an open transaction is necessary but not
sufficient. Fifteen host-ASan variants across three iterations were silent before the shape turned
up, and the instrumented runs say why:

1. With the inserts uncommitted the aliasing does happen -- probe site 47 prints
   `ALIAS zTerm -> hash key p=... term=5 bytes` -- and the mid-scan `optimize()` does free the hash
   while the scan's reader is alive. Still nothing reads the key, because a single-term lookup
   reader has one element, `fts3SegReaderNext` copied the doclist into its own buffer, and `zTerm`
   is read only by term comparisons, which all run before the first row.
2. `sqlite3Fts3MsrIncrNext`, the per-row advance, compares **docids** only
   (`fts3SegReaderDoclistCmp`), and for a pending reader it copies the position list into
   `pMsr->aBuffer` before handing it out. Upstream copies everything on this path except the term,
   which is exactly what `dee0359ddb` fixes.
3. So the trigger needs a **restart after the clear**: `fts3EvalRestart` ->
   `fts3EvalPhraseStart` -> `fts3EvalPhraseLoad` -> `fts3TermSelect` steps the EXISTING
   `Fts3MultiSegReader` again, and `fts3SegReaderStart` sorts its readers with `fts3SegReaderCmp`,
   which memcmps the dangling `zTerm`. The restart is reached the way case 06 reaches it: `offsets()`
   over a NEAR group under an OR whose AND operand to the left fails first, so `fts3EvalTestExpr`
   short-circuits before it can zero the group.
4. The comparison only happens when there are two readers to compare, so some rows must be
   **committed** (a segment reader) beside the uncommitted ones (the pending reader).

Hence `(xx AND (apple NEAR/3 qq)) OR pendingone`, two rows committed, two rows inserted inside
`BEGIN`, and `optimize()` after the first row. The query needs
`-DSQLITE_ENABLE_FTS3_PARENTHESIS`, so the case moves to the `fts3P` group.

Host ASan (prepared amalgamation, system allocator, lookaside off, 4 GiB quarantine):

```
READ of size 1, 0 bytes inside a freed 8-byte region
  use   memcmp interceptor <- fts3SegReaderCmp sqlite3.c:160670 <- fts3SegReaderSort
          <- fts3SegReaderStart <- fts3TermSelect <- fts3EvalPhraseLoad <- fts3EvalPhraseStart
          <- fts3EvalRestart <- fts3EvalRestart <- sqlite3Fts3EvalPhrasePoslist
          <- fts3ExprTermOffsetInit <- sqlite3Fts3Offsets
  free  fts3HashFree <- sqlite3Fts3HashClear <- sqlite3Fts3PendingTermsClear
          <- fts3DoOptimize <- fts3SpecialInsert
  alloc fts3HashMalloc <- sqlite3Fts3HashInsert <- fts3PendingTermsAddOne <- fts3InsertTerms
```

The freed region is the hash key copy itself, which is this case's declared object, and the free is
the mid-scan `optimize()`, which is its declared lifetime ender.

Probe site 47 prints the pointer at the alias, at the clear and at the comparison. The same key
pointer appears at all three on every platform:

| platform | trace |
| --- | --- |
| native memsys5 | `ALIAS p=0x…ee6b0` / `CLEAR frees p=0x…ee6b0` / `CMP lhs=0x…ee6b0` |
| virtual Capstone, lookaside off | `ALIAS p=0x3fa13ca010` / `CLEAR frees p=0x3fa13ca010` / `CMP lhs=0x3fa13ca010` |
| CheriBSD purecap | `ALIAS p=0x226b40` / `CLEAR frees p=0x226b40` / `CMP lhs=0x226b40` |

`witnesses=0` on all three: the `optimize()` allocates heavily right after the free and memsys5
reissues the block, so an address judgement at the comparison finds a live block -- the blind spot
case 05 documented. The pointer identity is the evidence, and it is identical across the three.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; reached, compared the freed key, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3f95e87544 in memcmp`, attributed by function to a declared site |
| cheribsd-revocation | MISSED | positive control traps; the case RETURNS rc=0, probe trace as above |

`fault_sites` declares both `fts3SegReaderCmp` and `memcmp`, the convention cases 13, 14, 17 and 19
already use for `binCollFunc`/`memcmp` and `sqlite3VdbeSerialPut`/`memcpy`: the stale load is the
comparison, which libc performs, and the runners resolve only the faulting pc's symbol.

**This changes a cell in the study's favour**, from not-detected to detected on the Sublet arm, so
it is the direction a reviewer should check hardest. It rests on three re-runnable readings: the
host ASan report, the cause-25 fault at a declared site with both controls as declared, and a probe
trace that is identical on three platforms.

Not a duplicate of case 06, which shares only the restart as a vehicle: the object here is the
pending-terms hash key rather than a merged poslist, the read is a term comparison rather than the
offsets loop, upstream fixed the two separately (3.37.0 against 3.49.0), and this access happens
inside the restart, so it strictly precedes case 06's.

**Tooling defect found on the way.** `cheribsd/build-corpus-cheri.sh` keys its cached probe engine
object on the defines and the amalgamation's FILE NAME, not its contents
(`key=$(printf %s "... $AMALG $cf" | sha256sum)`), so a second probe source at a different path
silently reuses the first object. The site-47 binary came back carrying site 46's hooks and printed
nothing; it was caught only because the binary held no `site=47` strings. Until the key includes the
source's hash, the cached object has to be removed by hand between probe sources.

The three physical rows in `results/matrix.tsv` measured the old trigger and are marked superseded.
