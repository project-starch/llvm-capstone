# PostgreSQL

| directory | what it is |
|---|---|
| [`memory-contexts/`](memory-contexts/README.md) | the memory managers outside the server: replay, client examples, the mmgr corpus's build seam, and the Sublet patch (0003) |
| [`app/`](app/README.md) | the complete 17.5 single-user backend on virtual Capstone and CheriBSD; `PGSU_NESTED=sublet` applies the same patch 0003 |

The memory-context component is where allocator work starts. Its `upstream.json` is the single
**PostgreSQL 17.0** pin for the component, the scripts below and the native defect corpus.

## The host scripts

    bash build-mmgr-host.sh      # the host build and replay the native corpus prepares from
    bash census-capstone.sh      # what a freestanding capability compile of the seven files costs

`build-mmgr-host.sh` configures the pinned tree that
`../../bug-corpora/postgres/mmgr-repros/run-host-repros.sh` builds its native arm from, and
replays a recording against the unmodified manager (`tools/replay.c`, `a11trace.h`). The host
build is checked against the backend itself: the recording carries what the real manager asked
of `malloc` over the same workload, and the host replay must ask for exactly that.
`census-capstone.sh` compiles the seven files for capstone64 against `port/stubinc` and reports
what fails, rather than asserting, so a compiler or PostgreSQL change shows up as a different list.

## Two things a reader would otherwise get wrong

**Two lines, and the allocator names them itself.** `aset.c` asserts that its free-list link fits
in the smallest chunk, because that link lives inside the freed chunk. A capability is sixteen
bytes and the smallest chunk was eight, so the assertion fails. Raising the minimum to sixteen
doubles the ladder, so one size class comes off to keep its top at the 8 KiB chunk limit, which a
second assertion checks. `port/aset-capstone.patch` and the component's patch 0001 are those two
lines. Any machine with sixteen-byte pointers needs them, and the size classes on a capability
machine are 16 to 8192 and not 8 to 8192, so the host's exact block count does not carry over.

**The allocator had the bug the compiler exists to find.** It aligned its region by masking the
address and casting it back, which discards the capability and faults on the first dereference
(`-Wcapstone-pointer-roundtrip`). It aligns by moving the pointer now.

## History

Until 2026-10-11 this directory also held the original freestanding-domain path: domain builds
of the manager (`build-mmgr-domain.sh`, `domain-build.sh`), their gates (`run-pg-gate.sh`,
`run-pg-sublet-gate.sh`) and runners, the guest loader (`pg_host.c`), and an AllocSet-only
Sublet adapter (`port/aset-sublet.patch`, `port/pg_subpool.h`, `port/freestanding/`). The
memory-context component's patch 0003 replaced the adapter, and the domain targets are not
part of the virtual platform.
