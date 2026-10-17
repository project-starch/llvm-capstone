# fts3destroyoom -- 129371553c

Upstream fix `129371553c`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
new-4 / sqlite-129371553c -- fts3DestroyMethod() dereferences a freed Fts3Table
 * after a nested OOM.
 *
 * fts3DestroyMethod() drops the five shadow tables through fts3DbExec():
 *
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_content'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segments'", zDb, p->zName);
 *     fts3DbExec(&rc, db, "DROP TABLE IF EXISTS %Q.'%q_segdir'",   zDb, p->zName);
 *     ...
 *
 * If one of those statements hits OOM, sqlite3VdbeHalt() -> sqlite3RollbackAll()
 * -> sqlite3ResetAllSchemasOfConnection() -> sqlite3VtabUnlockList() ->
 * fts3DisconnectMethod() sqlite3_free()s the Fts3Table -- i.e. a schema reset runs
 * underneath the object's own destructor. Control then returns to
 * fts3DestroyMethod, which goes on to the NEXT fts3DbExec. Its body returns early
 * because rc!=SQLITE_OK, but `p->zName` is still evaluated at the call site, and p
 * is freed. Fixed 3.27.0 by collapsing the five calls into one.
 *
 * Host ASan oracle on 3.22.0 (lookaside off, default page size, upstream's minimal
 * CREATE-then-DROP shape, counted OOM injection):
 *   READ of size 8, 40 bytes into a 592-byte region   (Fts3Table.zName)
 *     use   fts3DestroyMethod:149364 / :149365
 *     free  fts3DisconnectMethod <- sqlite3VtabUnlock <- sqlite3VtabUnlockList
 *             <- sqlite3ResetAllSchemasOfConnection <- sqlite3RollbackAll
 *             <- sqlite3VdbeHalt
 *     alloc fts3InitVtab
 *   40 of 300 injection points hit, in two contiguous runs: 167-184 and 259-280.
 *   The two runs are the two distinct fts3DbExec call sites that can be the use.
 *
 * WHY THIS DOMAIN CANNOT OBSERVE THE BUG ITSELF, and what it establishes instead.
 * The dangling read is Fts3Table.zName at offset 40. memsys5 overwrites only the
 * first 8 bytes of a freed block (its in-band Mem5Link freelist ints), so offset 40
 * keeps both its bytes and its capability tag: the read is necessarily SILENT on
 * base Capstone. It is also unobservable from inside the domain -- the value is
 * discarded (fts3DbExec returns before using it), and the post-DROP state is
 * identical whether or not the injection landed in the window: drop_rc=7 and the
 * table still present for EVERY injection point from 1 to 292 on the host. An
 * allocator-level witness does not separate them either, because distinguishing
 * in-window from out-of-window means knowing whether fts3DestroyMethod was still on
 * the stack when the free happened, which the allocator cannot see.
 *
 * So this domain does not claim a per-point correspondence. It establishes that its
 * allocation sequence is the SAME sequence the host oracle swept, by reporting two
 * numbers that are fixed by SQLite's code path:
 *     - the DROP's total allocation count   (host: 292)
 *     - the transition point where drop_rc stops being SQLITE_NOMEM and becomes
 *       SQLITE_OK because the injection point is past the end (host: last 7 at 292,
 *       first 0 at 293)
 * If those match, the host's window indices transfer. The domain then sweeps every
 * injection point and runs to NOTRAP, which is the control-arm result: the
 * use-after-free is silent on unprotected Capstone.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so OOM comes from a counting
 * wrapper installed over memsys5. SQLITE_CONFIG_HEAP sets sqlite3GlobalConfig.m to
 * memsys5's methods, SQLITE_CONFIG_GETMALLOC copies them out, and
 * SQLITE_CONFIG_MALLOC installs the wrapper -- so the order below matters and this
 * case does its own init instead of calling repro_init().
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted.
```

## Where an address-based oracle stops working (reviewed 2026-10-15)

The Sublet verdict stands, and establishing that took two probes rather than one, because this is
the case where judging a pointer by its address gives the wrong answer.

**The fault is this defect's.** Probe site 44, at the `sqlite3_free` in `fts3DisconnectMethod`,
records each `Fts3Table` the sweep frees. On `virtual-nested-pools` the last one before the fault
is `p=0x3fa6bcf000`, and the fault address is `0x3fa6bcf050` -- 80 bytes into that same 592-byte
object, with cause 25, the lifetime check, in `fts3DestroyMethod`. Host ASan reports the read at
offset 40 of the same object; the offsets differ because capability pointers change the layout.

**And an address check cannot see it.** Probe site 9 judges the same pointer against memsys5's own
free state, the way host ASan judges against its shadow map, and reports **no witness on either
virtual arm** in 70 reaches of the defect site. memsys5 reissues the freed block to the next
allocation before the stale read happens, so by the time the read occurs the address belongs to a
live object. That is not a gap in the probe: it is the same blindness that makes
`results/2026-10-11-host-asan` score this case only because `run-host-asan.py` replaces the
memsys5 configuration so every object is its own allocation, and the same blindness CheriBSD's
revocation has here. Sublet rejects the access because the reissued block carries a fresh colour
while the stale pointer keeps the revoked one.

So for this case the probe's `witnesses=0` is not evidence that nothing happened. Where a block is
reissued before the stale access, only an identity check can answer, and the site-44 address match
is what stands in for a witness.

## The arms do not reach the same occurrence

The trigger sweeps an OOM injection point over every allocation the DROP makes, so what it finds
depends on the allocation sequence -- and that differs per allocator:

| configuration | `drop_allocs` | what the sweep does |
|---|---:|---|
| host ASan variant (system allocator) | 296 | the recorded host oracle says 292, window 167-184 and 259-280 |
| `virtual-malloc` (stock memsys5) | 68 | full sweep of 76 points completes, 70 reaches, no fault |
| `virtual-nested-pools` (Sublet memsys5) | 70 | faults at about iteration 24 |

The Sublet port's memsys5 reports requested sizes rather than block sizes, which is the documented
reason its sequence differs from stock memsys5's. The consequence for this case is that
`virtual-malloc`'s MISSED is not a silent miss of the event Sublet caught: on that arm the sweep
ran to the end and the defect did not occur. A reader taking the two cells as the same defect seen
by two mechanisms would be reading more than the measurement supports.

## CheriBSD, and why the virtual arms are the outliers (2026-10-16)

Measured rather than inferred: on `cheribsd-revocation` this case completes, `RETURNED rc=0`, with
`drop_allocs=288`, a sweep of 296 points, and probe site 9 reporting no witness in 257 reaches.
The positive control faults under revocation in the same boot. So the address blindness described
above holds on that arm too, and the claim that CheriBSD's revocation cannot see this -- which was
an inference when first written -- is now a reading.

The four `drop_allocs` figures are 296 (host ASan variant), 288 (CheriBSD), 68 (`virtual-malloc`)
and 70 (`virtual-nested-pools`). The virtual arms are the outliers by a factor of four, and the
reason is not the allocator:

| configuration | lookaside | why |
|---|---|---|
| host ASan | OFF | `run-host-asan.py` sets `-DSQLITE_DEFAULT_LOOKASIDE=0,0` |
| CheriBSD | OFF | `cheribsd/build-corpus-cheri.sh:111` sets the same |
| both virtual arms | **ON** | `build-virtual.py` sets it nowhere; `REPRO322_NO_LOOKASIDE` exists but only for the `--no-lookaside` probe, which its own help calls "a probe, not an arm" |

With lookaside on, a connection's small allocations are served from its lookaside pool and never
reach the memsys5 wrapper this trigger counts, which is the whole of the four-fold difference.
That is not specific to this case: lookaside is SQLite's second nested allocator, the Sublet port
covers it and the stock arm does not, so on the virtual arms a small object's lifetime is a
lookaside slot's rather than a memsys5 block's. Recorded here because this case is where it became
visible; it bears on every case with small objects.
