# PostgreSQL 17.5: four-arm inner reuse

This campaign pairs six new CheriBSD processes with the six already qualified
Capstone processes. Each executes the same complete `work.sql`: insert 2,000
rows, index, query, update, delete and vacuum, ending at 1,500 rows. All 22
result rows must match the native 16-byte-MAXALIGN oracle. This is a complete
backend qualification workload, **not pgbench**. No trace replay is involved.

Run `python3 validate.py` to recheck every raw process, policy report and
32-bin histogram and compare the saved summary with its recomputed contents.
The two Capstone raw archives remain in their original adjacent directories;
their hashes and validators are part of this campaign's evidence. Three
repetitions assess repeatability of this input, not a population of workloads.

## Measured reuse

All 12 processes pass. Each row has three complete processes; ranges preserve
variation between them. Median gaps are bins, not interpolated point values.

| Arm | Issues/process | Observed reuses/process | Reuses/issues | Median observed gap |
|---|---:|---:|---:|---:|
| Capstone spatial | 54,032 | 45,002 | 83.29% | 4–7 |
| Capstone + Sublet | 54,032 | 43,705 | 80.89% | 4–7 |
| PoisonCap spatial | 54,004 | 44,974 | 83.28% | 4–7 |
| PoisonCap protected | 54,024–54,043 | 19,559–19,601 | 36.20–36.27% | 8,192–16,383 |

Every protected process performs five full-queue sweeps, zero percentage
sweeps and zero managed-reset sweeps. Peak occupied chunk records are
20,472–20,502, below the provisioned capacity. All inner observer errors and
tolerated-double-drop counters are zero. Sublet's reuse share is 2.40
percentage points below its own control; its median observed reuse gap stays
in the same bin. These results describe this SQL workload and allocation
boundary, not a general PostgreSQL performance or RSS advantage.

## Root cause and repair

The old protected adapter exhausted its fixed metadata tables. The original
core had all 8,191 usable chunk entries occupied, including 5,103 active
chunks, while both quarantine queues were empty. Sweeping could not free
records still needed for live or reusable chunks. Returning NULL entered
PostgreSQL's error machinery, which tried to allocate another error message
from the same exhausted table and recursed. Earlier attribution to
`flatten_grouping_sets` was incorrect: symbolization omitted the executable's
0x100000 PIE load bias. The corrected PC maps to `pg_subpool_carve`, followed
by allocation and error-formatting frames.

The repaired build provisions 65,536 chunk and 8,192 block records equally
for the spatial and protected modes, reports occupied/peak chunk records,
and terminates directly on capacity exhaustion. The block increase matters:
the published quarantine could outgrow the old 1,024-record block table
before either threshold required a sweep. These capacity failures are
adapter provisioning failures, not evidence of a PoisonCap memory advantage
or disadvantage. The enlarged tables' storage must be charged in any future
complete allocator memory comparison.

## Policy and measurement boundary

Both CheriBSD modes use the same freshly prepared `-O1` executable and
corrected libc. Mode 0 is the matched adapter-layout spatial control. The
Capstone control uses the original context layout. This difference is
explicit: compare each protected arm with its own control; allocation
counts across operating systems need not be identical for the same SQL.

The protected policy transfers the published SQLite thresholds with the
full-queue correctness repair: 4,096 queued entries, or at least 16 MiB held
and at least one quarter quarantined. A full queue revokes before clearing.
There is no allocation-pressure or immediate-reuse sweep. AllocSet searches
past quarantined entries for eligible free chunks. Released blocks remain
unavailable; their used spans replace overlapping individual chunks in the
quarantine ledger. Used block prefixes count in those spans; unused reserved
tails do not. Managed-reset exceptions are reported and must be zero for
admission to this campaign.

The application reports `malloc_revoke_enabled()` and the runner requires
outer libc revocation **enabled in both modes**, with asynchronous defaults
and no every-free override. Guest-wide default revocation is disabled only
for setup services: SCP otherwise exposed the separate `share->excl`
kernel fault before the benchmark started. Explicit process enablement and
the effective application setting are recorded in the raw evidence. This
is not a claim that the kernel locking bug has been fixed.

`PG_REUSE_GAP` observes real inner chunk lifetimes, including logical retirement
on context/block release. Delayed physical reclamation does not move the
logical release event. The distribution is conditional on observed same-start
reuse, indexed by successful new handouts. It is not elapsed time, a
fixed-follow-up probability, physical working set, or total memory. The CSV
retains issue/reuse counts so that a conditional violin cannot silently hide
its denominator. The violin pools all three complete process histograms;
individual counts and any repetition variation remain in `reuse-summary.json`. Full memory accounting and standard pgbench workloads are
separate outstanding milestones.

## Reproduction

Source `capstone/tests/capstone-test-env.sh`. Build with `CHERI_SDK`,
`CHERI_SYSROOT`, a fresh `PG_CHERI_ROOT`, `PG_CHERI_MODE=poisoncap` and
`PG_CHERI_GAP_OBSERVER=1`, using `ports/postgres/single-user/build-cheribsd.sh`.
The build manifest pins the archive, patches, compiler, flags, capacities,
ABI and executable. Restore the raw points and required inputs by their
manifest hashes; run the common `applications/cheribsd-run.py` with
`--disable-default-revocation --repeat 3 --timeout 300`. Its process policy
still explicitly enables libc revocation. The exact staging, pristine-cluster
extraction, runtime identities and invocations are in `cheribsd-raw.tar.gz`.
Private SSH keys and debug cores are excluded.
