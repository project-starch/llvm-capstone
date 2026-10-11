# CPython inner-pymalloc reuse qualification (three arms)

The complete CPython 3.13.7 `objects.py 8 3 0` interpreter workload passed
the same `EXP-OK cpython 552` output and nine phase checks in three fresh
processes per available arm. The Capstone spatial and Sublet processes ran
in one persistent Linux guest with 262,144 provisioned nodes. The CheriBSD
PoisonCap-adapter spatial control ran in one CheriBSD guest with default
libc revocation disabled. All nine `PYM_REUSE_GAP` histograms have `error=0`
and reconcile exactly with their reported reuses.

| Arm | Passing processes | Inner issues per process | Reused starts per process |
|---|---:|---:|---:|
| Capstone spatial | 3/3 | 77,005 | 48,882–48,883 |
| Capstone + Sublet | 3/3 | 77,005 | 48,882 |
| CheriBSD PoisonCap-adapter spatial | 3/3 | 80,192 | 52,248 |

The histogram index advances on successful new-lifetime handouts. Bins are
logarithmic distances from a release to a later issue at the same start;
they describe the **observed reuses only**. They are not the fixed-follow-up
retirement fraction in `memory-metrics.md`. The two platforms make different
numbers of inner allocations, so cross-platform raw counts are not a matched
denominator. The Capstone spatial and Sublet distributions are nearly equal
for this workload. The protected PoisonCap interpreter has no completed
workload oracle and is not included; therefore this is not a four-arm plot or
a memory-cost ranking.

`capstone-raw.tar.gz` and `cheribsd-raw.tar.gz` contain exact runner points,
manifests, commands, stdout, stderr and per-process records, without VM
private keys. `reuse-gap-validator-at-run.py` matches the validator SHA-256
recorded in both runner manifests; the active parser adds duplicate-field
rejection after the run and accepts the same raw reports. The build manifests
are adjacent. The CheriBSD binary was
rebuilt from the observer-enabled source before the archived repetitions.
Run `python3 validate.py` here to recheck every raw transcript and the
reported counts. These are application runs, not trace replays.
