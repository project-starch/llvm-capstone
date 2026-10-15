# backupattach -- b9e0f62c3a

Upstream fix `b9e0f62c3a`. Collected in round R1 (NVD and upstream history).

The reduction is argued in the case source's own header, reproduced here so the
claim travels with the case rather than living only in the build tree.

```
row15 / sqlite-b9e0f62c3a -- sqlite3_backup caches raw Btree* for src/dest
 * (backup.c:23/29, resolved by findBtree at 82). ATTACH reallocates db->aDb
 * (attach.c:113 sqlite3DbRealloc) and can invalidate the cached Btree*, so a
 * later backup step uses a freed/moved Btree. Fixed 3.53.3 (store db indices,
 * re-derive the Btree each step).
 *
 * CONTROL arm: start a backup, then ATTACH a database on the SAME connection to
 * force db->aDb realloc, then step the backup -- it dereferences the stale
 * Btree*. On unprotected Capstone the moved-Btree access is not caught, so the
 * step returns and the domain RETURNS. Host ASan flags heap-use-after-free.
```

## The trigger was replaced (2026-10-15)

The trigger this case carried until 2026-10-15 ATTACHed a database on the destination
connection and never detached one. ATTACH does not free a Btree in 3.22.0: `attach.c:108-113`
reallocates the `db->aDb` array and copies the `pBt` pointers into it unchanged. Nothing was
freed, so there was no use-after-free to find. What the arms reported instead was a NULL
dereference: the first `backup_step` opens a write transaction on the destination, the ATTACH
statement commits it, `pBt->pPage1` becomes NULL, and `p->bDestLocked` stays 1 so the next step
does not reopen one and `sqlite3BtreeUpdateMeta` reads through NULL. Host ASan on that trigger:
`SEGV on unknown address 0x50`, in `sqlite3BtreeUpdateMeta` from `sqlite3_backup_step` -- the
offset of `aData` within `MemPage`, not a heap address.

Upstream's own regression test for this fix (`test/backup5.test` 2.1) ATTACHes on the **source**
connection instead. Written out in C against stock 3.22.0, under host ASan, that is silent too.

**What does free the cached Btree is a DETACH of the destination database.** `detachFunc`
refuses a database that is in a read transaction or in a backup, but the backup check is
`sqlite3BtreeIsInBackup`, which reads `Btree.nBackup`, and `sqlite3_backup_init` increments that
on the source only (`backup.c:205`). The destination is never registered. Stepping the backup to
`SQLITE_DONE` first commits the destination's write transaction (`backup.c:567`), so the read
check passes as well, and `detachFunc` calls `sqlite3BtreeClose` on a Btree the backup object
still points at.

**Evidence** (result lines; probe sites 37 and 38 in `shared/probe/`):

- Host ASan, stock 3.22.0 on the system allocator: `heap-use-after-free`, 8 bytes inside a
  72-byte region, read in `sqlite3BtreeRollback` <- `sqlite3_backup_finish`, freed by
  `sqlite3BtreeClose` <- `detachFunc`, allocated by `sqlite3BtreeOpen` <- `attachFunc`.
- The same case with only the DETACH removed is silent and returns.
- Probe site 38 (the `sqlite3BtreeClose` in `detachFunc`) prints the destination Btree it is
  closing; site 37 (`BtShared *pBt = p->pBt` in `sqlite3BtreeRollback`) reports `kind=UAF off=0
  n=8` under memsys5. On the control, site 38 prints nothing and site 37 has no witness.
- Measured 2026-10-15 on the review platform: virtual-malloc MISSED, virtual-nested-pools CAUGHT
  with **cause 25** (`INVALID_CAP`, the lifetime check) in `sqlite3BtreeRollback`, both arms'
  controls as declared; CheriBSD purecap `RETURNED rc=0` with its positive control faulting under
  revocation. The arms separate as the corpus predicts for a nested temporal defect, which the
  old trigger's NULL dereference did not: it faulted on every arm, at one instruction, for a
  reason none of the mechanisms owns.

`ports/sqlite/repro322/case_backupattach.c` keeps the old trigger deliberately: it is the input
to the recorded silicon RTL smoke run of 2026-10-01
(`tests/rtl-smoke/repro322-silicon-2026-10-01/build-images.sh`), whose results and SHA256SUMS are
in the tree. That run's `mcause 24 at sqlite3BtreeUpdateMeta` is the NULL dereference described
above, not a detection. The corpus copy and `ports/sqlite/cheribsd/cases/case_backupattach.c`
carry the replacement.
