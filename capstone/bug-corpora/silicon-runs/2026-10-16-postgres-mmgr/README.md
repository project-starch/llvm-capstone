# PostgreSQL's memory-context corpus on silicon: pre-registration (2026-10-16, before the boot)

The board campaign's wave 4 (`docs/plans/2026-10-16-board-campaign-all.md`, §3). Five use-after-free and double-free
defects, all live at 17.5. Each is reduced to a program that runs against PostgreSQL's own memory-context managers
(aset, mcxt, generation, slab, bump, alignedalloc, memdebug) from the pinned 17.0 release.

Two arms, one boot on `caplifive_sublet_d93a8d66b`: private image 89295c34884a3378, firmware 9dc96b2345000482, on the
exact-regions supervised-call line of sets 2-7.

| arm | managers (the port's `cmake/prepare-source.py` variants) | heap | predicted from |
|---|---|---|---|
| pgm-silicon-pg-stock | `spatial`: patches 0001 + 0002, the capability layout | lifetimes | `virtual-malloc` |
| pgm-silicon-pg-sublet | `sublet`: + patch 0003, every chunk a CDERIVE child of its block, CREVOKEd at pfree and when its context dies | lifetimes | `virtual-pg-pools` |

**The build** mirrors the port's `cmake/Replay.cmake`:
- `case.c` + the corpus's `shared/driver.c` (`PG_CORPUS_HOSTED`);
- the variant's `aset.c`, plus mcxt/generation/slab/bump from the variant on Sublet and from upstream on stock;
- upstream `alignedalloc.c` and `memdebug.c`;
- the port's `postgres-compat.c` and `printf.c`;
- `REPLAY_CHECK_DATA`;
- the variant's `pg_config.h` with `SIZEOF_VOID_P` 16.

The archive is postgresql-17.0.tar.bz2, sha256 7e276131…, checked by the script. The build is -O0, as the port's
capstone-application preset, which inherits Debug.

**The protocol** (kind `pg-mmgr`):
- `0 N`;
- completion is `PG_DEFECT case=N mode=0 completed`;
- a fault counts in `pg_probe` or a case.json fault site (case 00: `SubletRelease`, `GetMemoryChunkMethodID`);
- each case's `-DPG_NEGATIVE_CONTROL` build must complete and gates the arm;
- the corpus control 00 (one chunk pfree'd and read through its alias) must complete on stock and fault at the probe
  on Sublet;
- the tool's four controls run too.

## The silicon-mimic QEMU reading (capstone-qemu silicon/cderive-round ce1be0c642, `CAPSTONE_CDERIVE_ROUND=1`)

- **pgm-silicon-pg-stock:** all 5 complete, the stale access unseen. Every negative control completes, and the corpus
  control completes.
- **pgm-silicon-pg-sublet:**
  - 01-04 CAUGHT 24 at `pg_probe`;
  - 00 (the double free) CAUGHT 24 in `SubletRelease`, its declared site: the second pfree's CREVOKE of a revoked
    child;
  - every negative control completes;
  - the corpus control is CAUGHT at `pg_probe`.

## The board prediction

Every row as QEMU read it, by outcome and site, with each 24 reading 25.
- That includes 00's: the second pfree hands CREVOKE a revoked child, which the RTL keeps tagged (Q-11), as for
  wmem 22 in set 5.
- No 5 / 7 occurs in a case row.
