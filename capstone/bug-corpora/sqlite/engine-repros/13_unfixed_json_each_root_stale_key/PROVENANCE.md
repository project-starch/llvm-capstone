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

## 2026-10-16: verified in the pin, attribution strengthened, two records corrected

This case needs the most careful reading of any so far, because it is the one the field found
itself rather than collected, and its `upstream_fix` is `unfixed`. Three things were checked.

**The defect is verified in the pin, line by line.** `jsonEachColumn`'s default arm -- the ROOT
column -- is

```c
    default: {
      const char *zRoot = p->zRoot;
      if( zRoot==0 ) zRoot = "$";
      sqlite3_result_text(ctx, zRoot, -1, SQLITE_STATIC);
      break;
    }
    case JEACH_JSON: {
      ...
      sqlite3_result_text(ctx, p->sParse.zJson, -1, SQLITE_STATIC);
```

`SQLITE_STATIC` means the SQL layer keeps the pointer without copying. `jsonEachFilter` allocates
that buffer (`p->zRoot = sqlite3_malloc64(n+1); memcpy(...)`) and `jsonEachCursorReset` frees it
(`sqlite3_free(p->zRoot)`), so re-filtering the cursor per outer row frees what an accumulator
still holds. Host ASan on this platform:

```
READ of size 1, 0 bytes inside a freed 8-byte region
  use   memcmp <- binCollFunc sqlite3.c:144244 <- vdbeCompareMemString <- sqlite3MemCompare
          <- minmaxStep
  free  sqlite3_free <- jsonEachCursorReset sqlite3.c:185490 <- jsonEachFilter
```

**What cannot be checked from here.** That the defect is live in 3.22.0 is settled above. That
upstream has *still* not fixed this sibling arm is the field's reading of 28001204f4 and cannot be
confirmed offline. The corpus measures the pin, so the `unfixed` label affects provenance, not any
verdict -- but a reviewer will ask, and the honest answer is that this one row rests on our own
reading rather than on an upstream commit.

**The attribution was weaker than the rest of the corpus, and is now pointer identity.**
`fault_sites` is `['binCollFunc', 'memcmp']` and the old `fault_sites_why` said so plainly: "Weaker
than a single function: any memcmp under this case would count", and it was declared on 2026-10-10
*after* the virtual runs resolved faults there. Probe site 52 replaces that reasoning with a
measurement. It prints the pointer at the `SQLITE_STATIC` hand-out, at the free in
`jsonEachCursorReset`, and at every comparison in `binCollFunc`, judging both operands:

```
STATIC hands out zRoot p=X [$.bb]
CMP k1=X k2=X                      while X is live
RESET  frees zRoot p=X
STATIC hands out zRoot p=Y [$.bb]  the next cursor's own buffer
CMP k1=X k2=Y   x3                 X is dead: the accumulator's kept MEM_Static pointer
```

Exactly **three** comparisons have the freed pointer as their first operand, and the count and the
shape are identical on native memsys5, on virtual Capstone with lookaside off and on CheriBSD
purecap (24 probe events, 3 stale comparisons). The verdict now rests on that identity rather than
on the function name. `witnesses=0` everywhere, because memsys5 reissues the freed block -- case 05's blind spot again.

**Two record corrections.** `fidelity` said this case has "the same shape and same reduction as
case 18 (withdrawn, campaign numbering)"; case 18 (withdrawn, campaign numbering) is the unrelated memsys5 freelist demonstrator, and the reference was to case 14.
The two json_each cases are kept apart on the *object*, not the instruction: 17 hands out
`p->zJson` from the `JEACH_JSON` arm, which upstream converted to `SQLITE_TRANSIENT`, and this one
hands out `p->zRoot` from the default arm, which it did not. Upstream fixing one arm and not the
other is itself the argument that they are separable -- and it is a different argument from the one
that made case 09 (withdrawn, campaign numbering) a duplicate of 10, where the read instruction, the saved pointer and the free
chain were all the same.

Arms, measured 2026-10-16 on this platform:

| arm | verdict | evidence |
| --- | --- | --- |
| host ASan | reports | as above |
| virtual-malloc | MISSED | both controls as declared; cross_rows=6, rows=1, RETURNED |
| virtual-nested-pools | CAUGHT | both controls as declared; `cause=25 pc=0x3fae630198 in memcmp`, attributed by function and corroborated by site 52 |
| cheribsd-revocation | MISSED | positive control traps; cross_rows=6, rows=1, RETURNED rc=0 |

Both of the case's own reachability probes held in every run: the cross join produces 6 rows, and
the root column reads `$.bb`, so the malloc'd-path branch was taken rather than the literal `"$"`
fallback. No count in the study's tables moves; the physical sublet row, one of the 28 kernel-pc
halts, is marked in `results/matrix.tsv`.
