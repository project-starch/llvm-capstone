# PostgreSQL 17.5 complete single-user backend

This port runs the PostgreSQL backend itself on Capstone/Linux and CheriBSD
purecap. `work.sql` creates a table, inserts 2,000 rows, builds an index, then
queries, updates, deletes and vacuums it. The checked final count is 1,500.
This is an application qualification workload, not pgbench and not a trace
replay. PostgreSQL sources are fetched outside the repository and pinned by
the archive SHA-256 in both build scripts.

## Builds

Source `capstone/tests/capstone-test-env.sh` first. `build-domain.sh` builds a
Capstone image with `PGSU_NESTED=none` or `sublet`; use separate `PG_SU_ROOT`
directories for those modes. The Sublet build applies the existing AllocSet,
Generation, Slab and Bump lifetime patches to 17.5 and links
`context-pools.o`. The application SDK's `build.py --app postgres --nested
postgres` supplies a separate 64 MiB inner arena and its counters.

`build-cheribsd.sh` uses `CHERI_SDK`, `CHERI_SYSROOT`, `PG_CHERI_MODE=spatial`
or `poisoncap`, and a distinct `PG_CHERI_ROOT` per mode. It applies the same
application ABI patches as the Capstone build where relevant, pins
`MAXIMUM_ALIGNOF=16`, and rebuilds `src/port/qsort.o` after the tag-preserving
swap patch. Study builds require a fresh root; `PG_CHERI_REUSE=1` permits an
explicit development rebuild, refreshes the adapter sources, and records root
reuse in `manifest.json`. Each build also records the archive, applied patches,
compiler, ABI settings and binary hashes. The PoisonCap variant links the
existing memory-context hook backend, with a quarantine policy selected for
the complete application.
Set `PG_POISONCAP_MODE=0` for its original-layout control or `1` for poison,
sweep and detox. The two modes use the same binary and context layout.

The batch policy poisons a freed chunk immediately, retains it on an external
queue, and sweeps before a queued chunk is issued again. Context block returns
still sweep synchronously. The adapter reports queued counts and bytes,
high-water marks, sweeps, poison/clear/zero spans, and mapped backing. Those
are selected allocator ledgers, not total RSS or a hardware-time estimate.
The older direct-link adapter remains available without the batch definition
for its existing defect checks; it sweeps on every free and must not be used
as a lower-bound cost for PoisonCap.

For a pristine Capstone input cluster, run
`python3 capstone/ports/postgres/single-user/build-native16-fixture.py`
after sourcing the test environment. It builds a native PostgreSQL 17.5
fixture generator at `-O1` with the domain's 16-byte `MAXIMUM_ALIGNOF` and
matching 16-byte chunk-header layout, then runs upstream `initdb` with C
locale, GMT and System V DSM. The fresh root and cluster remain under
`$CAPSTONE_TMP_ROOT`; its manifest records the archive, patch, binaries and
cluster-tree hashes. Copy the untouched cluster for each application attempt.
An ordinary 8-byte-MAXALIGN native cluster cannot be used by either domain.

## Running and checking

Use the common persistent Capstone application VM or the common CheriBSD
`Guest` runner. Give every attempt a fresh copy of a PostgreSQL cluster
initialized by a **16-byte MAXALIGN backend**. A conventional native cluster
uses a different on-disk alignment and is not interchangeable. Supply
`timezone=GMT`, `log_timezone=GMT`, `shared_buffers=4MB`,
`max_connections=10` to `--single`. The Capstone application SDK backend
uses `dynamic_shared_memory_type=sysv`, because its single-process System V
segment service is implemented and file-backed `mmap` is not. Pass the
`--single` arguments and `PG_SINGLE_INPUT=/mnt/host/.../work.sql` directly
through the shared application runner; set `PGSU_DOMAIN=1` for the domain's
synthetic uid 0. The CheriBSD backend uses `dynamic_shared_memory_type=posix`
with the same SQL input. CheriBSD needs the guest's ordinary
revocation default disabled while the adapter's explicit sweeps remain on.

Compare each result with an independent native 17.5 `work.sql` result using
`work-compare.py`. It requires all 22 result rows and the final count before
checking exact row equality. Also require successful process termination and
the mode-specific adapter report: `su` can return zero after a child SIGPROT,
so its status alone does not qualify a run. Do not use a row-equivalent but
faulted attempt as a paper measurement.

The full four-arm memory comparison additionally needs phase-level live,
retained, quarantined and metadata ledgers and repeated benchmark runs. The
SQL oracle alone does not establish any memory advantage.
