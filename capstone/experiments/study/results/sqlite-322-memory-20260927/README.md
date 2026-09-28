# SQLite 3.22.0 nested-allocator memory pilot (2026-09-27)

The full, official 32-phase `speedtest1 main --size 1` workload completes in
Capstone spatial, Capstone + Sublet, PoisonCap spatial, and PoisonCap temporal
with a corrected full-quarantine path. Every phase's SQL result count and
FNV-1a hash matches an independently compiled native SQLite 3.22.0 oracle;
the run produces 4,301 result rows. Lookaside is effectively disabled in all
arms. This is one exploratory execution per passing cell, not a statistical or
timing result.

**Build-comparability audit (2026-09-27): these four binaries are not yet a
normalized cross-platform build.** The Capstone translation unit starts from
the official 3.22.0 amalgamation (source ID `2018-01-22 ... 0c55d179`), while
the published PoisonCap port reports the same version number but a different
source ID (`2026-04-26 ... 637f6b03`). A direct source comparison shows 431
added and 56 removed lines in the PoisonCap port, including CHERI alignment,
`memsys5`, and quarantine changes. Both run the same 32 benchmark phases and
match the SQL-result oracle, but that proves output equivalence, not identical
allocation demand.

The Capstone build scripts use `-O0` for SQLite, `SQLITE_OS_OTHER=1`, a private
in-memory VFS, `SQLITE_TEMP_STORE=3`, `SQLITE_DEFAULT_LOOKASIDE=0,0`,
`SQLITE_ZERO_MALLOC=1`, and numerous `SQLITE_OMIT_*` switches. The recorded
PoisonCap binary exposes only `ENABLE_MEMSYS5`, `THREADSAFE=0`,
`OMIT_FLOATING_POINT`, and `OMIT_DEPRECATED` in its compiled-option strings;
its exact compile argv and optimization level were not preserved. The runs
report zero successful lookaside allocations, but that does not make their
complete feature configurations equal. The CheriBSD Unix VFS and Capstone
domain VFS also remain different by design. The four-arm backing bars and
phase curves are therefore exploratory observations of these specific builds,
not paper-ready protection overhead or fragmentation estimates. The
PoisonCap published/corrected policy-path control is a within-platform result.

A compile/link probe under `/tmp/capstone/poisoncap-plots` applied the complete
deployed Capstone SQLite define list to the instrumented PoisonCap spatial
source at `-O0`, omitting only `SQLITE_OS_OTHER`; its workload driver compiled
at `-O1`. The CheriBSD purecap executable linked successfully. Its first
guest attempt did **not** execute SQLite: the published guest kernel panicked
in `vm_map.c:6103` (`share->excl`) while `scp` copied the executable. The raw
serial log is `/tmp/capstone/poisoncap-plots/parity-guest-20260927/serial.log`.
The probe binary's SHA-256 is
`af9ae23b78f52b5e11b709c099f08420a8003f30da51e95b510cd8c1952f9668`;
the panic log's is
`38e6a2eb9fe2cfbce38f0ea7e7a0e9d141d523264ef1af2abea40516ee26132c`.
Thus feature parity is compile-feasible, but runtime parity and the 32-phase
oracle still require verification on that rebuilt binary.

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

The [backing figure](sqlite-backing.pdf) shows selected successful reservations
and their within-platform differences; these are not matched minimal budgets.
The [release/refill figure](sqlite-release-refill.pdf)
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
The [campaign build contract](../../../../docs/plans/application-memory-campaign.md)
and [build gate](../../check-build-comparability.py) describe what the next
four-arm rebuild must record and match before those figures can be promoted.
