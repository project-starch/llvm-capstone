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
directories for those modes, and `ports/common/application/build-virtual.sh
postgres` to build either for the virtual profile. The Sublet build applies
the memory-context port's patch 0003
(`../memory-contexts/patches/postgresql-17.0-0003-memory-contexts-sublet-lifetimes.patch`)
to 17.5: every chunk a context hands out is a child lifetime of its block
(`CDERIVE`), and `pfree`/`repalloc` revoke it (`CREVOKE`). The 17.0 and 17.5
manager sources are identical. Nothing is linked beside the backend and no
separate arena is granted; the blocks come from the process's malloc.

For inner reuse measurement in the unprotected build, `PGSU_GAP_OBSERVER=1`
compiles the fixed integer histogram (patch 0017). It observes memory-context
chunk handouts, explicit frees and reallocations, plus live chunks retired on
context reset or deletion. Aligned-allocation wrappers delegate to the
ordinary chunk methods, so they are counted once. It reports `PG_REUSE_GAP`
at exit. Link the complete image through the application SDK with
`--reuse-gap` so that the report and the observer object survive section
collection. The Sublet build skips patch 0017, whose hooks sit where patch 0003
changes `mcxt.c`, and refuses the observer. Set `PGSU_ARGV0_OVERRIDE` to a
dedicated share-local `.dom` path when staging a measurement image beside its
own PostgreSQL `share/` directory; this avoids replacing another study's
staged binary.

`build-cheribsd.sh` uses `CHERI_SDK`, `CHERI_SYSROOT` and a fresh
`PG_CHERI_ROOT`. It builds the stock memory contexts, for the platform's own
libc revocation. It applies the same application ABI patches as the Capstone
build where relevant, pins `MAXIMUM_ALIGNOF=16`, and rebuilds
`src/port/qsort.o` after the tag-preserving swap patch. Study builds require a
fresh root; `PG_CHERI_REUSE=1` permits an explicit development rebuild and
records root reuse in `manifest.json`. Each build also records the archive,
applied patches, compiler, ABI settings and binary hashes. Until 2026-10-11 it
also built a PoisonCap variant (`PG_CHERI_MODE=poisoncap`, patch 0018 and the
memory-context port's lifetime adapter); its results stay as recorded.

For a pristine Capstone input cluster, run
`python3 capstone/ports/postgres/app/build-native16-fixture.py`
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
`--single` arguments through the [shared application runner](../../common/application/README.md),
using `--stdin work.sql` and `--user UID:GID` for an existing unprivileged guest
account. The cluster must be writable by that account. The old input-file and
root-bypass patches are removed, and so are the two that stood in for a process spawn
and for timers (the former 0014 and 0019): `setitimer` is delegated and SIGALRM is
delivered at the backend's next delegated call. So `statement_timeout` works: a
300 ms timeout cancelled `pg_sleep(2)` after 324 ms (the backend waits in `poll`),
and a counting query after about 600 ms (the signal waits for the query's next
delegated call). Statistics are scheduled with PostgreSQL's own idle SIGALRM again;
that path was not measured on its own, but it rides the same timer and signal.
`pg_import_system_collations` runs `locale -a` through the launcher's spawn; the
guest has no `locale`, so it imports none, with PostgreSQL's own warning, as the
patch did silently. Lock and idle-transaction timeouts are not measured. The CheriBSD backend uses `dynamic_shared_memory_type=posix`
with the same SQL input. Disable the guest-wide revocation default for
setup services: the kernel can panic while SCP stages a cluster. Published-policy
points explicitly enable libc revocation in each benchmark process;
`PG_RUNTIME` reports the effective setting and the runner checks it. Both
process modes load the same corrected libc. Guest-service isolation is
recorded separately from the application's enabled outer policy.

Compare each result with an independent native 17.5 `work.sql` result using
`work-compare.py`. It requires all 22 result rows and the final count before
checking exact row equality. Also require successful process termination:
`su` can return zero after a child SIGPROT, so its status alone does not
qualify a run. Do not use a row-equivalent but
faulted attempt as a paper measurement.

The full four-arm memory comparison additionally needs phase-level live,
retained, quarantined and metadata ledgers and repeated benchmark runs. The
SQL oracle alone does not establish any memory advantage.
