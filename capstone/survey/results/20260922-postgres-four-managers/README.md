# The aggregated context level is whichever manager dominates the workload

Four W3 rows from one recording, on PostgreSQL 17.0, the revision
`capstone/ports/postgres/memory-contexts` pins. The aggregated `context`
level the survey has always reported is kept beside them, because what it
turns out to be is the finding.

## Two rungs, five levels, repetition 1

| rung | level | allocations | bulk | reuses | before release | distinct |
|---|---|---:|---:|---:|---:|---:|
| tpcb | `context` | 1,228,005 | 1,029,570 | 1,224,280 | **55.5 %** | 3,724 |
| tpcb | `aset` | 1,228,068 | 1,029,570 | 1,224,343 | **55.5 %** | 3,724 |
| tpcb | `generation` | 0 | | | never created | |
| tpcb | `slab` | 2 | 2 | 1 | 0.0 % | 1 |
| tpcb | `bump` | 5 | 5 | 2 | 0.0 % | 3 |
| sort | `context` | 20,119,297 | 20,098,964 | 19,921,248 | **0.5 %** | 198,048 |
| sort | `aset` | 119,270 | 98,964 | 115,709 | **30.6 %** | 3,560 |
| sort | `generation` | 0 | | | never created | |
| sort | `slab` | 2 | 2 | 0 | 0.0 % | 2 |
| sort | `bump` | 20,000,000 | 20,000,000 | 19,804,347 | **0.3 %** | 195,653 |

On tpcb the aggregate is AllocSet. The two differ by 63 allocations out of
1.23 million and read the same share to one decimal.

On sort the aggregate is Bump, 20 million of its 20.1 million allocations,
and it reads 0.5 per cent where Bump reads 0.3. AllocSet is doing something
quite different on the same run, 30.6 per cent over its own 119,270
allocations, and none of that survives in the aggregate.

So the aggregated figure moves from 55.5 to 0.5 between two rungs of one
program, and neither number belongs to a manager. It belongs to whichever
manager the workload happened to drive. Splitting the level is what makes
the number attributable, and this is that argument made with PostgreSQL's
own counters rather than by assertion.

## Generation and Slab

Generation is never created on either rung. Slab is created twice. That is
a statement about these two workloads and not about the managers, and it is
checked rather than assumed: the `generation` arm's report carries its
`libc` level and no manager level at all, and the counting core does not
print a level without activity. Both managers are ported and protected, and
neither recorded workload exercises them, so neither has a reuse share
here. `levels.json` marks them recorded and not exercised, with the reason,
and the check refuses a level that claims the one without the other.

## One build, five levels

The wrappers were already per manager, one `A1_WRAP` for each of the four
method-table entries, so all that was missing was a choice. `A1_PG_LEVEL`
names one manager at startup, and then only that manager's methods are
wrapped and `a1_select_level` renames the level so the report says whose
counts these are. Unset, all four are wrapped and the level stays
`context`, which is the row the survey has always produced. A name the
build does not know aborts rather than quietly measuring everything.

## Provenance

| | |
|---|---|
| PostgreSQL | 17.0, sha256 `7e2761…09de` |
| pin | matches `capstone/ports/postgres/memory-contexts/upstream.json` exactly, so it has a second independent record |
| workloads | pgbench `tpcb` and `sort`, scale 10, 1000 transactions, three repetitions |
| arms | `shipped` as the cross-check, `hooked` aggregated, and one per manager |
| instrument | `pghook.c` and `hook.py` in the manuscript's `experiments/a1/postgres/` |
| compiler | gcc 13.3.0 |
| raw | `raw/`, hashed in `raw/SHA256SUMS` |

## Limits

Two rungs. A build fix travels with this recording: in 17.0 the
`src submake-generated-headers` target does not produce
`utils/errcodes.h`, and the parallel build then fails with that header
missing in `src/common`, which is the symptom the Makefile's own comment
predicts. The backend's `generated-headers` does produce it, and asking for
both is harmless.
