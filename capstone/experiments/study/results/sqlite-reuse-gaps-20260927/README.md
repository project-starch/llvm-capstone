# SQLite 3.22.0: release-to-reuse gaps in the full application

The [paper-width reuse figure](reuse-gaps.pdf) compares the four nested
allocator arms while the complete official SQLite `speedtest1 main --size 1`
workload executes. The application itself issues all allocations and frees.
One process performs a warmup and 16 measured complete work units, opening
and closing a fresh database per unit while retaining allocator policy state.
Three independent processes per arm all complete: **12/12 long runs**, plus
**4/4 one-unit qualifications**. All **6,528** long-run SQL phase oracles
match the independent native reference. No allocation-demand replay occurs.

For each released block, the observer records the number of subsequent
successful memsys5 allocations before its **exact arena start** is allocated
again. Immediately reused storage has gap 1. The CDF divides reuses by
**all successful measured-unit allocations**, including allocations that never
reuse a start. Thus the curve's endpoint shows reuse frequency as well as gap.
The one allocation made during SQLite setup before the first work unit is
separately counted and excluded from the denominator. Across the full 17-unit
run, every arm has **550,137** measured allocations. All three repetitions
in every arm produce identical bins and counts.

| Arm | Same-start reuses / all allocations | Gap 1 / all | Gap ≤15 / all | Gap ≤4095 / all |
|---|---:|---:|---:|---:|
| Capstone original | 99.649% | 21.368% | 73.116% | 92.945% |
| Capstone + Sublet | 99.649% | 21.368% | 73.116% | 92.945% |
| CheriBSD original | 99.561% | 21.362% | 73.113% | 91.753% |
| PoisonCap corrected | 86.358% | 0.0075% | 0.052% | 18.011% |

Sublet and its own original have **identical counts in every gap bin**. With
PoisonCap, the gap-≤15 share drops by **73.061 percentage points** against
its own original, and same-start reuse by **13.203 percentage points** over
the complete observed run. The fractions are not throughput, cache hit rates,
physical working sets, or proof of asymptotic behavior. Same-start reuse does
not count partial byte overlap after splitting or coalescing.

## Comparability and measurement checks

The four application/driver builds use the same matched SQLite 3.22.0 source,
explicit feature switches, 64-byte memsys5 atoms, lookaside disabled,
129,055 allocatable atoms per arm, and `-O0`. Necessary VFS, compiler and ABI
differences remain documented in the [parent build comparison](../sqlite-normalized-memory-20260927/README.md).
The PoisonCap arm uses the separately labeled **corrected full-queue
revocation** path, not the unmodified published policy. The two original-layout
allocators are the within-platform denominators.

The new observer stores **integer block indices and allocation counts**, not
application capabilities. It uses 573,712 static bytes per arm, including the
earlier 49,152-byte address observer. Every new allocation total and reuse-bin
sum reconciles at each unit close; the latter also equals the independently
kept same-start reuse count. Every one of the **6,732** phase ledger rows is
identical to the earlier normalized application campaign in all fields except
the deliberately larger observer size. That check covers live, quarantined,
free, peak, metadata, address coverage, and allocation counters. It supports
observer invariance for this workload and configuration.

Both Capstone arms run the same QEMU and the legacy SQLite domain host with
`CAPSTONE_REV_NODES=4194304`; this finite node capacity is not a memory-cost
measurement or a long-run scalability result. CheriBSD uses the published
PoisonCap platform with outer automatic libc revocation disabled and explicit
nested revocation active in the corrected protected arm. No total kernel/node
memory or resident working set is reported here. The protected memory price
remains the separate [selected-byte result](../sqlite-normalized-memory-20260927/README.md):
Sublet uses more selected allocator metadata despite its preserved reuse.

Two initial Capstone transport attempts did not start SQLite because a 9p
guest share cannot follow the prepared host's absolute symlinks. They were
preserved in the [external raw archive](archive.json), then the same host and
measured binaries were copied as ordinary files into the share. The 4/4
qualifications and 12/12 measured processes above follow that correction.
There is no application failure in this campaign.

## Evidence and reproduction

- [summary.json](summary.json) reports fractions, ranges and paired percentage
  point effects; [bins.csv](bins.csv) retains every repetition and all 32 gap
  classes; [runs.json](runs.json) records binary and raw-output hashes.
- [source-delta.json](source-delta.json) and [patches](patches) identify the
  exact extra instrumentation relative to the earlier reconstructed SQLite
  campaign. The workload driver is byte-identical to that campaign.
- [provenance.json](provenance.json) identifies the plotter, input manifests and
  binaries. The external [archive](archive.json) retains raw output, builds,
  sources and the excluded setup attempts without guest private keys.

From the study worktree, with the experiment Python environment:

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-sqlite-reuse-gaps.py \
  /tmp/capstone/sqlite-reuse-gap-20260927 \
  capstone/experiments/study/results/sqlite-reuse-gaps-20260927
```

The plotter rejects incomplete processes, changed SQL oracles, mismatched
measured binaries, malformed histograms and any changed allocator phase ledger.
The vector PDF is 7.05 inches wide for a two-column manuscript figure.
