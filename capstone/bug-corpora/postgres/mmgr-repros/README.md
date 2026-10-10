# PostgreSQL memory-context defects

Five upstream use-after-free and double-free defects in PostgreSQL's memory
managers, all live at 17.5, reduced to programs that run against PostgreSQL's
**own** allocator: the memory-context managers from the pinned 17.0 release, compiled
unmodified but for the capability-ABI and Sublet patches the port applies. The
consumers are reduced, the allocator is not.

The layout is the contract in
[`../../SCHEMA.md`](../../SCHEMA.md):
one directory per case, `NN_<upstream-fix>_<slug>/`, holding a `case.c` that is
a complete translation unit, a `case.json` of machine-readable claims, and a
`PROVENANCE.md`. Case numbers are dense from zero. One program per case,
because a capability fault ends the run and a case that provokes one cannot
also report results beside it.

## The cases

| # | upstream fix | shape | layer | live in pin |
|---|---|---|---|---|
| 0 | `1f5b6a5e5d` | double free through a stale array entry | aset | **yes** |
| 1 | `3549ffb6af` | stale pointer to a recreated object | aset | **yes** |
| 2 | `ed394c4bdf` | alias freed through a sibling | aset | **yes** |
| 3 | `a61592253e` | stale pointer into a deleted ancestor | aset | **yes** |
| 4 | `9e0b4b1ab5` | same-address reuse from a fixed-size free list | slab | **yes** |

Three more (`83ce20d671`, `727bc6ac33f6`, `9d5ce4f1a00a`) are live at 17.0 but
already fixed in 17.5, so they are parked in
[`not-live-at-17-5/`](not-live-at-17-5/README.md) with their results.
`shared/run-defects.py` reads the case list from these directories, so it
labels and numbers runs the same way this table does.

## Shapes

`case.json`'s `shape` must be one of these, so that two cases sharing a
reduction class are visibly siblings rather than accidentally similar.

| shape | what makes it its own class |
|---|---|
| double free through a stale array entry | the stale access IS the second free, so there is no read probe |
| stale pointer to a recreated object | the object is destroyed and immediately recreated; the holder is never updated |
| alias freed through a sibling | one object, two pointers; freed through one and read through the other |
| stale pointer into a reset context | no free at all — a pool reset ends the lifetime |
| stale pointer into a deleted ancestor | an ancestor context dies, taking a grandchild's arena with it |
| same-address reuse from a fixed-size free list | reuse is deterministic, so the successor lands at the identical address |

## The systems

| system | what it is |
|---|---|
| **virtual Capstone** | this project's capability architecture on RISC-V. LLVM fork (`capstone64-unknown-elf`), QEMU fork (`virt-capstone`); each case runs as a Linux process under `capstone-vexec`, its heap musl mallocng |
| **Sublet** | two instructions on virtual Capstone: `CDERIVE` makes a child lifetime of a capability, `CREVOKE` ends a direct child and everything below it. The port's patch 0003 makes every chunk a child of its block and revokes it in `pfree` and `repalloc` |
| **CheriBSD** | CHERI-RISC-V purecap, CheriBSD under QEMU. Its own temporal safety is malloc quarantine plus a revoker sweep |

The same `case.c` builds for every target. That is what makes the comparison
one: it is the same source, not two reimplementations.

## Arms

| arm | target | what it establishes |
|---|---|---|
| `virtual-malloc` | virtual Capstone, the managers with the capability-layout patches | whether bounds and the system allocator see these defects |
| `virtual-pg-pools` | the same with patch 0003 (`PG_SUBLET=ON`) | fault at the labelled read probe; case 0 in `SubletRelease` |
| `cheribsd-revocation` | CheriBSD purecap, libc revocation on | whether the system allocator's revocation sees them |
| `native-detect` | host, `before.c` | written for cases 2 and 4 only |

What an arm must show is its configuration's, in `tools/arms.json`; the shared
judge (`tools/verdicts.py`) refuses to score a silence from an arm whose
controls did not behave.

## Building and running

The port builds the cases; case material does not live inside a port. Pass the
corpus root and the port builds one program per case for whichever target it
is configured for:

    -DPG_CORPUS_DIR=<repo>/capstone/bug-corpora/postgres/mmgr-repros

Programs are named as the contract names run artifacts --
`03-pgoutput-entry-cxt-teardown` -- so an archived result tree stays readable
away from the corpus.

**The virtual arms** run through `shared/run-defects.py`, against a VM started
with `capstone_vm --profile virtual`. Two builds per arm, both with the
`capstone-application` preset on a virtual SDK, the second from `controls/`
([controls/README.md](controls/README.md)); `virtual-pg-pools` adds
`-DPG_SUBLET=ON` to both:

    python3 shared/run-defects.py OUT --state VM_STATE --arm virtual-pg-pools \
      --hosted-build BUILD --controls-hosted-build BUILD_CONTROLS \
      --llvm-bin <toolchain>/bin --raw LOGS

The runner refuses a build whose `PG_SUBLET` or SDK does not match the arm. It
reports one Observation per run and the shared judge decides; it writes
`OUT/<arm>/` bundles, and `tools/derive-verdicts.py` turns the bundles named in
`corpus.json` `verdict_bundles` into the `case.json` verdicts and
`results/verdicts.tsv` (SCHEMA.md, "Verdicts").

**The CheriBSD arm** builds the same cases with the `cheribsd` preset and runs
them under the guest's own libc revocation; `results/20261008-cheribsd` is that
run.

## Why the system allocator's mechanisms miss these

PostgreSQL asks the system allocator for a block once and hands out chunks
from it itself, so between the `pfree` and the stale read there is nothing on
the layer that `free()`-time revocation watches. The same reason ASan is silent
here. Only a mechanism that listens for the moment the NESTED allocator takes
the chunk back can catch them.

## History

Until 2026-10-11 the corpus also ran a `spatial`/`sublet` pair in a physical
Capstone domain (`results/20260918-qemu`, `results/2026-10-10-qemu`) and a
PoisonCap pair on CheriBSD (2026-09-20). Both used the port's lifetime adapter,
which `CDERIVE`/`CREVOKE` made unnecessary, and `virtual-pg-pools` measured that
adapter over a lent linear arena (`results/2026-10-10-virtual`). Their targets
and runners are removed and their results stay as recorded. The 2026-09-21
comparison of eight cases (before the 17.5 re-pin) found the nested-allocator
mechanisms catching 8/8 and the system allocator's 0/8; case 0 faulted in
`GetMemoryChunkMethodID` under the adapter, which read the revoked chunk's
header first.

## What is NOT established

- **`native-detect` for cases 0, 1 and 3.** Only cases 2 and 4 have a `before.c`.

## What IS established, and how

Every case carries a `live_proof` in their `case.json`: an inspection at the
pin with file and line, cross-checked against the upstream commit's date, since
every one of these fixes postdates the 17.0 release of 2024-09-26.

Cases 1 and 2 are two DIFFERENT defects at the same site and were initially
attributed the wrong way round. `83ce20d671` (2024-12-04) fixes the callee's
stale local in `parallel_vacuum_reset_dead_items`; `3549ffb6af` (2025-10-03)
fixes the caller's unrefreshed field in `dead_items_reset`. Reading the pinned
tree alone could not separate them — it shows the pre-fix state but not which
fix addresses which half — and the inventory in
`docs/ref/postgres-nested-allocator-defects.md` calling them "same, 2024
instance" reinforced the error. Their `distinguishing` fields now say which is
which.
