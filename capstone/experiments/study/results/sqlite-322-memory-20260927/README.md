# SQLite 3.22.0 nested-allocator memory pilot (2026-09-27)

The full, official 32-phase `speedtest1 main --size 1` workload completes in
Capstone spatial, Capstone + Sublet, PoisonCap spatial, and PoisonCap temporal
with a corrected full-quarantine path. Every phase's SQL result count and
FNV-1a hash matches an independently compiled native SQLite 3.22.0 oracle;
the run produces 4,301 result rows. Lookaside is effectively disabled in all
arms. This is one exploratory execution per passing cell, not a statistical or
timing result.

| Arm | Heap/pool | App allocator tables/links | Static quarantine table | Total application-visible reservation |
|---|---:|---:|---:|---:|
| Capstone spatial | 1.25 MiB | 0 | 0 | 1.25 MiB |
| Capstone + Sublet | 1.25 MiB | 0.80 MiB | 0 | 2.05 MiB |
| PoisonCap spatial | 1.25 MiB | 0.16 MiB | 0.13 MiB | 1.53 MiB |
| PoisonCap temporal, corrected | 8.00 MiB | 0.99 MiB | 0.13 MiB | 9.11 MiB |

Thus the **observed successful configurations** add 0.80 MiB for Sublet
within Capstone and 7.58 MiB for corrected temporal PoisonCap within its
spatial platform. These are selected passing capacities, not proven minimum
capacities or total system memory. Kernel shadow, Capstone node metadata,
libc, stack, page tables and process RSS are excluded. The spatial and
temporal PoisonCap arms use the same instrumented allocator layout, including
the static quarantine table. The totals count page-rounded link mappings.

The corrected temporal arm reaches a peak 3.68 MiB quarantine and 4.09 MiB
live-plus-quarantine rounded heap capacity; it completes six full-queue drains
with six successful explicit revocations. The published full-queue policy
completes the same SQL work at an 8 MiB heap but performs **zero** revocations
for its six full-queue drains. It is shown only as a policy-path control; it
cannot stand in for the protected temporal comparator. The corrected 4.5 and
7 MiB trials panic in the published kernel with `Poison probe missing page`;
that fault prevents a minimum-heap conclusion. Smaller Capstone and spatial
PoisonCap trials also fail; every attempted configuration remains listed in
`data.json`.

The [backing figure](sqlite-backing.pdf) shows reservations and the two
within-platform increments. The [release/refill figure](sqlite-release-refill.pdf)
shows phase-end rounded live capacity and quarantine, with intra-phase peaks
in [data.json](data.json). The [policy-path figure](sqlite-poisoncap-policy.pdf)
reports completed drains/revocations and explicit payload rewriting. None of
these figures uses QEMU elapsed time. PNG copies are included for previews.

`data.json` includes source, binary and raw-log SHA-256 digests and all 32
phase records. Raw VM transcripts remain outside the repository, in accordance
with the project's log policy. The stock SQLite 3.22.0 amalgamation and official
speedtest1 driver were fetched under `/tmp/capstone` and are identified by
their digests in `data.json`; benchmark source is not vendored. The study's
source patches live under `capstone/ports/sqlite/study/`, and the memsys5-only
Sublet backport lives under `capstone/ports/sqlite/sublet/`.

To redraw from committed data (after installing `matplotlib` and `numpy`):

```sh
python3 capstone/experiments/study/plot-sqlite-memory.py \
  --data capstone/experiments/study/results/sqlite-322-memory-20260927/data.json \
  --out /tmp/sqlite-322-memory-figures
```

The source-log validation path uses `--inputs` with a manifest naming the
original transcripts and binaries. The checked figures were made through that
path. Capstone still used the legacy SQLite domain runner with a boot per
attempt because this port's old `create_dom` host is incompatible with the
persistent process VM; this affects experimental overhead, not the reported
memory counters. The next measurement pass needs a shared runner integration,
multiple matched workload sizes and repeated blocks before a paper claim.
