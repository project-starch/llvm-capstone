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
For inner reuse measurement, `PGSU_GAP_OBSERVER=1` compiles the fixed
integer histogram into either build. The original-layout build observes
memory-context chunk handouts, explicit frees and reallocations, plus live
chunks retired on context reset or deletion. Aligned-allocation wrappers
delegate to the ordinary chunk methods, so they are counted once. The Sublet
build observes its chunk handout/release path, including chunks retired by a
block or context reset. Both report `PG_REUSE_GAP` at exit. Link the complete
image through the application SDK with `--reuse-gap` so that the report and
the observer object survive section collection. Set `PGSU_ARGV0_OVERRIDE` to a dedicated share-local `.dom`
path when staging a measurement image beside its own PostgreSQL `share/`
directory; this avoids replacing another study's staged binary.

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
Set `PG_POISONCAP_MODE=0` for its matched adapter-layout spatial control or `1` for poison,
sweep and detox. The two modes use the same binary and context layout.
`PG_CHERI_GAP_OBSERVER=1` adds a fixed-capacity, integer-only observer at
the inner chunk handout/release boundary. Its `PG_REUSE_GAP` report contains
32 log-binned release-to-reissue distances, counts and an error code; reject
runs with a nonzero error or a histogram sum different from `reuses`.
The index counts successful handouts, so this report describes observed
reuses rather than the fixed-follow-up retirement metric.

The batch policy transfers the published SQLite MEMSYS5 thresholds: poison on
free, sweep after insertion when held spans reach 16 MiB and quarantined spans
reach one quarter of held spans, or before adding an entry to a full
4,096-entry queue. The full-queue path revokes before clearing, following the
study's documented correctness repair. Freed blocks also remain unavailable;
their spans replace overlapping queued chunks in the accounting. Held spans
are live rounded chunk spans plus quarantined chunk/block spans (including
used block prefixes). Reserved, unused block tails are not held spans.
Allocation skips poisoned free-list entries and never triggers an early sweep.
A managed reset that requires immediate reuse has a separately reported
exception counter; the published-policy measurement runner rejects nonzero
`managed_reset_sweeps`.

The complete backend defaults to 65,536 chunk and 8,192 block metadata entries,
configurable with `PG_CHERI_CHUNK_CAPACITY` and `PG_CHERI_BLOCK_CAPACITY`.
Both runtime modes use identical capacities. These are metadata limits, not
quarantine thresholds. The old 8,191 usable chunk entries were exhausted by
live and retained chunks; returning NULL caused recursive allocation during
PostgreSQL error formatting. Exhaustion now reports a direct failure and
bypasses allocation-using exit callbacks. Reused build roots reject mismatched
compiler flags and missing quarantine patches.

Reports include occupied/peak chunk records, static chunk-table bytes, queue
counts and spans, sweeps by cause, poison/clear/zero spans and mapped backing.
These are selected ledgers, not total RSS. The direct-link component without
the batch definition retains its synchronous policy for defect checks.

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
with the same SQL input. Disable the guest-wide revocation default for
setup services: the kernel can panic while SCP stages a cluster. Published-policy
points explicitly enable libc revocation in each benchmark process;
`PG_RUNTIME` reports the effective setting and the runner checks it. Both
process modes load the same corrected libc. Guest-service isolation is
recorded separately from the application's enabled outer policy.

Compare each result with an independent native 17.5 `work.sql` result using
`work-compare.py`. It requires all 22 result rows and the final count before
checking exact row equality. Also require successful process termination and
the mode-specific adapter report: `su` can return zero after a child SIGPROT,
so its status alone does not qualify a run. Do not use a row-equivalent but
faulted attempt as a paper measurement.

The full four-arm memory comparison additionally needs phase-level live,
retained, quarantined and metadata ledgers and repeated benchmark runs. The
SQL oracle alone does not establish any memory advantage.
