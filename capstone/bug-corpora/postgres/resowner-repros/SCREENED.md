# The sweep's other temporal findings, and why each is not a case here

A sweep of `REL_17_5..REL_18_6` (3781 non-merge master commits) plus
`REL_17_5..REL_17_11` (752 back-patches) on 2026-11-10 found **32** fixes for
temporal memory violations, of which **14** are live at the 17.5 pin. The code
diff of all 165 keyword candidates was read, not only the commit messages.

Five of the fourteen are already cases: `1f5b6a5e5d`, `3549ffb6af`,
`ed394c4bdf`, `a61592253e` and `9e0b4b1ab5` are `../mmgr-repros/00..04`. Two
became cases 00 and 01 here. This file is the other seven, so that the sweep's
count is not mistaken for a yield and so that none of them is re-examined from
scratch.

## Rejected because no allocator event happens at all

These four are genuine temporal defects at the pin and **no allocator-based
mechanism can see any of them, by construction.** That is a finding about the
limits of what this project measures, not a gap in the corpus, which is why
they are recorded here with their mechanism rather than dropped.

| 17-branch fix | fixed in | where | why no mechanism can see it |
|---|---|---|---|
| `ea834d747d` | 17.11 | `replication/logical/slotsync.c` | `drop_local_obsolete_slots` reads `local_slot->data` after `ReplicationSlotDropAcquired`. The slot is an element of one static shared-memory array that stays live; the stale pointer is **in bounds** of that allocation, and no free occurs |
| `080d61f07c` | 17.11 | `replication/slot.c` | `ReplicationSlotRelease` writes `effective_xmin` into an ephemeral slot already dropped. Same array, same reason |
| `d489c4439e` | 17.11 | `storage/lmgr/proc.c` | `ProcKill` pushes the leader's `PGPROC` onto the freelist twice. A double push onto an intrusive list in static shared memory: no allocator is involved on either push |
| `4071fe900e` | 17.8 | `storage/ipc/ipc.c` | `LWLockReleaseAll` runs after `dsm_backend_shutdown` unmapped the segment holding the lock. The memory is removed by `munmap`, so the fault is the OS's to deliver and an allocator has nothing to revoke |

A reduction of any of these would be a static array and an index, and every arm
would be silent for a reason that is already known before the run. If one is
ever built it belongs in a corpus whose boundary is shared memory, with that
silence as its declared expectation.

## Rejected because the boundary is another corpus's

| 17-branch fix | fixed in | where | which boundary |
|---|---|---|---|
| `1d0fc2499f` | 17.8 | `access/transam/parallel.c` | memory contexts -- `../mmgr-repros`, not this corpus. `ReinitializeParallelDSM` lets `shm_mq_attach` allocate in a context shorter-lived than the `ParallelContext` that points at the handle. Shape: stale pointer into a reset context. Upstream is explicit that it is latent: *"there is no way to reach a crash using only code that is present in core PostgreSQL"*, so it would be a case about an extension's reach, and `../mmgr-repros` is already `status: measured` with eight arms -- adding a sixth case re-opens that |
| `ac3afd1d00` | 17.6 | `utils/mmgr/alignedalloc.c` | allocator-internal. `AlignedAllocRealloc` pfrees the old chunk even when the inner allocation returned NULL. `../mmgr-repros` declares allocator-internal changes out of scope (*"a row is out of scope if it is allocator-internal"*), and rightly: its cases exist to show a CONSUMER defect the allocator hides. This one is the allocator. It is also latent twice over -- it needs `MCXT_ALLOC_NO_OOM` **and** a real OOM, and upstream says *"AFAICT no in-core callers use this function at all, let alone call it with MCXT_ALLOC_NO_OOM"* |

## Rejected because the reduction would be a model of the subject

| 17-branch fix | fixed in | where | why not |
|---|---|---|---|
| `af8f9248fb` | 17.10 | `contrib/postgres_fdw/connection.c` | An open cursor calls libpq on a `PGconn` already passed to `PQfinish` by `pgfdw_reject_incomplete_xact_state_change`. Temporal and non-nested, the same cell as cases 00 and 01 -- but the subject is libpq's connection lifetime, and reaching it needs libpq and a second server. A reduction would have to model `PGconn` itself, and a corpus whose subject is modelled measures the model. Case 01 reaches the same cell with real `resowner.c` doing the work, so this one buys nothing it does not already have |

## Not temporal, though the sweep's keywords matched

| fix | why not |
|---|---|
| `0c8b4e9cfc` (17.10) | the double-pfree DETECTOR, not a defect. Quoted in `../mmgr-repros` as the evidence that 17.x cannot see this class in production |
| `60887d34818` (17.11) | CVE-2026-16239, portal tuple-descriptor cross-check. Type confusion, not a lifetime defect |

## One correction this sweep forced

An earlier note called `2aa6be6e64` a **mixed** nested/non-nested defect, on the
reasoning that `od` is a memory-context chunk and `od->evp_ctx` is OpenSSL's.
Reading 17.5's `free_openssl_cipher` settles it the other way: the handle is
freed **before** the `ResourceOwnerForget` that raises, so `pfree(od)` is never
reached on either pass. `od` leaks; only the libc handle is freed twice. Case
01 is cleanly non-nested, and `01_.../PROVENANCE.md` records the order.
