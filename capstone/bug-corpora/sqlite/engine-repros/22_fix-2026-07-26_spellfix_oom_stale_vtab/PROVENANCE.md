# spellfixoom -- fix-2026-07-26

Upstream fix `2026-07-26`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row24 / sqlite-d9305a65c876 -- spellfix editDist3Install() registers editdist3()
 * three times sharing ONE heap EditDist3Config, and only the arity-1 registration
 * carries the editDist3ConfigDelete destructor. The structure is:
 *
 *     pConfig = sqlite3_malloc64(sizeof(*pConfig));                       // (1)
 *     rc = create_function_v2("editdist3", 2, ..., pConfig, ..., 0);       // (2)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 3, ..., pConfig, ..., 0); }  // (3)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 1, ..., pConfig, ...,
 *                                                  editDist3ConfigDelete); }          // (4)
 *     }else{ sqlite3_free(pConfig); }
 *
 * NOTE the else belongs to the THIRD if, so it fires when (2) or (3) failed:
 *   - (3) fails after (2) succeeded -> pConfig is FREED while editdist3/2 is still
 *     registered on it. Calling editdist3(a,b) then runs editDist3SqlFunc ->
 *     editDist3FindLang over the freed config. THIS is the use-after-free.
 *   - (2) fails -> nothing got registered, so no UAF.
 *   - (4) fails -> the else is NOT taken, so pConfig merely leaks (no destructor).
 * There is no refcount, hence the bug. Trunk-only fix. Ext: ext/misc/spellfix.c.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so the only OOM lever is
 * exhausting the memsys5 arena. editDist3Install is the LAST step of
 * spellfix1Register, so everything before it can be allowed to succeed.
 *
 * Rather than guess the reserve size across many QEMU boots, this domain SWEEPS it
 * itself in one boot: for each k it drains the arena to ~k*64 free bytes, runs the
 * init, then releases the arena and asks whether editdist3/2 is still callable.
 * rc==SQLITE_NOMEM together with a callable editdist3/2 IS the bug state, and the
 * domain then calls it to perform the freed-config read.
 * NOTE: -DSQLITE_DQS=0 -> SQL string literals must be single-quoted.
```

## 2026-10-11: the trigger is the CheriBSD copy

`case.c` is now byte-identical to `ports/sqlite/cheribsd/cases/` (the copy the CheriBSD probe runs used).
The source above is the earlier trigger. Why it was replaced: the corpus copy forced OOM by filling the heap with blocks; host ASan was silent on it. The CheriBSD copy sweeps an allocation-failure point through a wrapped allocator until spellfix init returns NOMEM with editdist3 still registered, then calls it; host ASan (the wrapper on the system allocator) reports the heap-use-after-free in editDist3FindLang. Evidence:
`results/2026-10-11-host-asan/results.json`, from `shared/run-host-asan.py` (every SQLite object its own
malloc, lookaside off).

One change against the CheriBSD copy, made the same day: `run_case` turns the lookaside off
(`SQLITE_CONFIG_LOOKASIDE, 0, 0`) before `sqlite3_initialize`. A lookaside slot is handed out
without calling the allocator, so the counting wrapper cannot fail it. With the lookaside on, 8 of
spellfix init's 16 allocations reached the wrapper on Capstone and the sweep printed `NO BUG STATE
FOUND`; host ASan, built with the lookaside off, found the bug state at the 14th.
