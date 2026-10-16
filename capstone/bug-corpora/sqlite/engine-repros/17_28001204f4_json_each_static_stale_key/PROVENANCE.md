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

## 2026-10-16: re-measured, and this is the corpus's sharpest case for identity over address

Every verdict holds. What the probe added is the clearest evidence in this corpus that neither an
address oracle nor a value check can see a stale access, and that identity still can.

**Verified in the pin.** The `JEACH_JSON` arm of `jsonEachColumn` is
`sqlite3_result_text(ctx, p->sParse.zJson, -1, SQLITE_STATIC)`; `jsonEachFilter` allocates
`p->zJson` with `sqlite3_malloc64` and passes it to `jsonParse`, which is what makes
`p->sParse.zJson` that same buffer; `jsonEachCursorReset` frees it. Host ASan:

```
READ of size 1, 0 bytes inside a freed 8-byte region
  use   memcmp <- binCollFunc sqlite3.c:144244 <- vdbeCompareMemString <- sqlite3MemCompare
          <- minmaxStep
  free  sqlite3_free <- jsonEachCursorReset sqlite3.c:185489 <- jsonEachFilter
  alloc sqlite3_malloc64 <- jsonEachFilter sqlite3.c:185749
```

Those last two line numbers are what separate this case from case 16 in the measurement rather
than in prose: case 16's buffer is freed at :185490 and allocated at :185771, because it is
`p->zRoot` from the default arm. One line apart in the free, twenty-two in the allocation, two
different fields, and upstream fixed this arm and not that one.

**The finding.** Probe site 53 prints the pointer at the `SQLITE_STATIC` hand-out, at the free, and
at every comparison in `binCollFunc`, judging both operands. The same sequence appears on native
memsys5, on virtual Capstone with lookaside off and on CheriBSD purecap:

```
RESET  frees zJson p=X
STATIC hands out zJson p=X [[3,4,5]]     the NEXT cursor's buffer, same address
CMP k1=X k2=X n=7      x3                the accumulator's dead pointer reads X
```

memsys5 reissues the freed block to the next cursor's `zJson`, at the **same address** and with the
**same content** -- the trigger re-filters the same inner table, so the new buffer holds the same
JSON text. Therefore:

* an address oracle cannot see the access: the block is live when it is read, which is why
  `witnesses=0` of 14 hits on all three platforms;
* a value or sanity check cannot see it either: the bytes are exactly the ones the comparison
  expected, and `max()` returns the right answer;
* host ASan sees it only because its quarantine refuses to reissue the block, which is a debugging
  configuration and not how the engine runs;
* Sublet rejects it with cause 25 anyway, because the accumulator holds a capability revoked at the
  free while the new cursor holds a different child covering the same bytes. Identity is the only
  distinction left when both the address and the content are "correct".

Case 16, the sibling, happened to have its freed block reissued elsewhere, so there the stale
operand was separable by address. That the two siblings differ in this respect is itself worth
keeping: whether an address-based oracle can see a stale access depends on allocator timing, not on
the defect.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; cross_rows=6, rows=1, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3f9da30018 in memcmp`, attributed by function and corroborated by site 53 |
| cheribsd-revocation | MISSED | positive control traps; cross_rows=6, rows=1, RETURNED rc=0 |

The physical sublet row reports `cause 24` at `pc=0x1021b801c` rather than the lifetime check.
Cause 24 is untagged-dereference and cannot distinguish a revoked capability from one that never
was, so that row does not establish that the lifetime check fired; it is marked in
`results/matrix.tsv`, and the virtual arm's cause 25 at the declared site is the reading this
review relies on. No count in the study's tables moves.
