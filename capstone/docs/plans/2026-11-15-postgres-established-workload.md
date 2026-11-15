# Reuse distance at PostgreSQL's memory contexts on pgbench's TPC-B-like transactions: pre-registration, 2026-11-15

Committed before any measured run. A reading that differs from what is written here is a finding;
this file is not edited afterwards.

## The question

The October study (`2026-10-11-nested-reuse-distance.md`) measured PostgreSQL 17.5 on `work.sql`, a
single-user session that we wrote. This run asks the same question on an established workload:
does the lifetime patch change when PostgreSQL's memory-context allocator reuses an address, while
the backend executes pgbench's built-in TPC-B-like transactions? It also measures how much two runs
of the same arm differ (the noise floor).

## Definition and boundary (October's)

Reuse distance counts allocations between the release of an address and its next hand-out, at one
allocator boundary, in power-of-two buckets by bit length (bucket k holds 2^(k-1) <= d < 2^k). The
observer is `experiments/study/reuse-gap-observer.h` through
`experiments/study/reuse-hooks/postgres/exp-reuse-gap.h`. The boundary is "contexts": every chunk
that `mcxt.c` gets from a manager's alloc method (aset, generation, slab, bump) is an issue at the
manager's chunk address; pfree and a moving repalloc release it; a reset or delete releases every
chunk still live in the context. The hooks and both arms' patches are unchanged from run-3.

## Platform

The trap route of run-3, kit `~/asr-work/eval/kit-run-3` on p13: QEMU c55fcdd2 (capstone-qemu
`address-space-root` 9289114e06), kernel Image 0ea6d1ef (Linux 6.1, `native-kernel-new-qemu` series),
firmware 975a2421, initramfs 2d846ebe, capstone-vexec 150d9692. Images: `reuse-hooks/postgres/build.sh`
for each arm from a tree export of `trap/eval` ed8cc375c2 (libc and lifetime patches of
`native-kernel-new-qemu` 7ba3f427f9), with the study patches `postgres-17.5-private-shmem.patch` and
`postgres-17.5-no-interval-timer.patch` in both arms, as in run-3. Image sha256: stock 43f0d5d3
(byte-identical to run-3's stock image), protected 98ca3e35.

## Why a single-user session and not pgbench against a server

pgbench is a client: it connects to a postmaster, which forks one backend per connection plus
auxiliary processes and shares its buffers and locks through shared memory. A protected program on
this platform cannot fork (fork, vfork, clone and clone3 return ENOSYS,
`runtime/virtual/pthread-contract.c`), cannot create System V or shared mappings
(`runtime/virtual/mapping.c`, and the kernel refuses shared mappings), and a signal to a handler ends
it. So no protected postmaster can serve pgbench. The run therefore replays pgbench's own
statements in one single-user backend, the mode run-3 used.

## Workload

- **Program**: PostgreSQL 17.5, `postgresql-17.5.tar.bz2` sha256
  fcb7ab38e23b264d1902cb25e6adafb4525a6ebcbd015434aeef9eda80f528d8
  (https://ftp.postgresql.org/pub/source/v17.5/postgresql-17.5.tar.bz2), and pgbench of the same release.
- **Initial database**: the run-3 fixture cluster (`prebuilt/reuse/pg-fixture/cluster`, initdb
  `--locale=C --encoding=UTF8`, 16-byte MAXALIGN), initialised by `pgbench -i -s 1` (default init
  steps `dtgvp`: drop, create tables, client-side COPY, vacuum, primary keys) against the native
  16-byte-MAXALIGN 17.5 server that built the fixture (`eval/tmp/pg-native16/install`, postgres
  0ec491f6, pgbench ab47ba9d). Scale 1: 100,000 accounts, 10 tellers, 1 branch.
- **Statements**: on a fresh copy of that cluster and the same native server,
  `pgbench -n -c 1 -j 1 -t T --random-seed=1 -M simple --debug postgres` with the built-in
  tpcb-like script. pgbench's debug log prints each statement as sent, after variable substitution;
  those lines in order are `tpcb.sql`, 7 T statements (BEGIN; UPDATE pgbench_accounts; SELECT
  abalance; UPDATE pgbench_tellers; UPDATE pgbench_branches; INSERT INTO pgbench_history; END). `-n`
  skips pgbench's pre-run vacuum and truncation, which a freshly initialised database does not need.
  The generation runs twice on separate copies and must give identical files.
  `reuse-hooks/postgres/make-pgbench-workload.sh` does all of this and writes `manifest.txt`.
- **T = 4000 transactions** (28,000 statements; workload `bench-postgres-perl/gen/s1-t4000`,
  `tpcb.sql` sha256 f095983b..., initialised cluster tree fa14e868..., `native.out` 602b65a4...). All
  four runs start from that one cluster. Calibration with the stock image on a T = 1000 workload
  (`bench-postgres-perl/cal/stock-t1000`, unmeasured; its statements are byte for byte the first
  7,000 of the T = 4000 file, its cluster a separate `pgbench -i`): exit 0, result rows equal to the native replay, observer error 0, 1,217,756 issues,
  1,211,821 reuses, 5,935 distinct starts, 312 s in the guest. T = 4000 is predicted at about
  21 minutes for stock and stays inside 60 minutes if the protected arm is up to 2.5 times slower.
- **Guest command** (run-3's, with `tpcb.sql` in place of `work.sql`): a fresh copy of the cluster
  in `/tmp/pgcluster`, then as user nobody `capstone-vexec postgres-ARM.dom --single -D /tmp/pgcluster
  -c shared_buffers=4MB -c max_connections=10 -c timezone=GMT -c log_timezone=GMT
  -c dynamic_shared_memory_type=mmap -c exit_on_error=true postgres < tpcb.sql`.
- **Output oracle**: the result rows (the pattern of run-ports.py's `pg_rows`) of every run must
  equal those of the native single-user backend reading the same `tpcb.sql` from the same cluster
  (`native.out`). This includes one `abalance` row per transaction.

## Arms and repetitions

**stock** and **protected** (the memory-contexts lifetime patch), the same observer in both. Two
repetitions per arm (stock-1, stock-2, protected-1, protected-2), one boot each, run by
`reuse-hooks/postgres/run-pgbench-reuse.py`. Results under `~/asr-work/eval/run-5/reuse/postgres-ARM-REP/`
(trap_platform.py's results.json, boot-1/, stage-1/, plus rep.json and provenance.txt), collected
by `reuse-hooks/cpython/collect-reuse.py`.

A run is rejected if it did not complete, exited nonzero, logged a fault, reports a nonzero observer
error, or its result rows differ from `native.out`.

## Measures

Per run: issues, releases, reuses, distinct start addresses, observer error, guest seconds. The
share of reuses in another bucket (`reusediff` of collect-reuse.py, sum_k |a_k/N_a - b_k/N_b| / 2)
for stock-1 against stock-2 (noise floor, stock), protected-1 against protected-2 (noise floor,
protected), and stock against protected (rep 1 against rep 1, and the range over the four pairs).

## Predictions

1. **Noise floor near zero.** One backend executes a fixed statement stream from a fixed cluster
   with no concurrency, so each arm repeats its allocation sequence. Predicted: within-arm reusediff
   at most 0.001 in both arms (identical bins expected; residual differences can only come from
   time-dependent text such as log-line timestamps).
2. **Stock against protected: small.** The protected arm's memory-context patch leaves the number
   of hand-outs and releases alone but changes a few placements (run-3 on work.sql: issues equal,
   reuses 44,472 against 44,464, reusediff 0.0039). Predicted: issues and releases equal between the
   arms, reuses within 0.1 %, median bucket equal, stock-against-protected reusediff below 0.01.
3. **Output**: every run's result rows equal the native replay, so the arms answer identically.
