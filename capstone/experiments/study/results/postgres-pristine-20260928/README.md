# PostgreSQL 17.5 single-user qualification on a pristine cluster

The complete 2,000-row `work.sql` passes 6/6 Capstone attempts: three original
and three memory-context Sublet processes. Every run starts from a fresh copy
of the same 16-byte-MAXALIGN cluster in one persistent Linux VM boot, exits
normally, and exactly matches the independent native PostgreSQL 17.5 oracle:
22 SQL-result rows, including the final count of 1,500. The 1,081-byte
standard-output SHA-256 is
`fee37cc7adaf0c2c4e221ad07d150e5ed7f727b6d76903ee3285776f4e0f5329`.

The [fixture builder](../../../../ports/postgres/single-user/build-native16-fixture.py)
uses the pinned 17.5 archive, `-O1`, a 16-byte chunk header and a 16-byte
MAXALIGN, then runs upstream `initdb` with C locale. Its untouched output uses
GMT and System V dynamic shared memory. The builder and shared runner both
record the same cluster-tree SHA-256:
`ac378cd83c86129c8e843d7759bcc70bfd6ba1e893af5759dea3eb8c43fb9170`.
The native workload runs on a separate copy. The Capstone application SDK
images receive arguments and `PG_SINGLE_INPUT` directly from the shared
runner, and both compile the application at `-O1`.

The [summary](summary.json) and [raw archive index](archive.json) preserve
the six checked attempts, input, fixture and binary hashes, logs and native
oracle. The archive remains under `$CAPSTONE_TMP_ROOT`, not in Git. PostgreSQL
emits one `sync_file_range` warning per process because that optional
writeback hint is unserved; all six SQL executions finish and match the oracle.

This is a **functional two-arm qualification**, not a four-arm memory figure.
The reported outer-heap peak excludes Sublet's separate 64 MiB context region
and Capstone node storage, so its smaller number cannot be used as a total
footprint comparison. The protected PoisonCap `work.sql` run does not yet
complete; a threshold/quarantine policy and matched inner memory ledger are
still needed. `work.sql` is a whole-backend qualification workload, not
`pgbench`.
