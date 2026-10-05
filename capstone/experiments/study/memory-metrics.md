# Memory metrics for the four-arm application campaign

Status: measurement specification, not a new result. This refines the
[campaign's A–D experiments](../../docs/plans/application-memory-campaign.md).
It applies to real application executions in Capstone spatial, Capstone +
Sublet, PoisonCap spatial and PoisonCap temporal. The observer runs at the
chosen inner allocation boundary; no allocation replay substitutes for an
application. There is no security score or QEMU timing comparison here.

## What a paper result estimates

The unit of comparison is one pinned application workload, input, policy,
build profile and completed-work horizon. The unit of replication is an
independent process execution, not an individual allocation or SQL phase.
The primary question is how much allocator storage is unavailable to new
requests while completing the same work, and whether that demand grows over
repeated units. Address reuse explains that behavior; it is not itself a
physical-memory saving.

Report both absolute four-arm values and protected-minus-spatial differences
within each platform. Cross-platform differences include ABI, port and policy
effects; even a difference between the two protection increments is not an
isolated hardware effect. A spatial control that already contains external
tables must retain those bytes in its absolute ledger. Subtracting that
control cannot establish the complete cost of adapting the original allocator.

Build parity is necessary. Matching explicit flags alone does not verify all
compiler defaults, ABI layouts or allocation requests. Preserve effective
feature settings, pointer/alignment sizes, the useful-output oracle, and
per-unit request counts/size histograms beside the build manifest. Investigate
within-platform demand differences before interpreting a protection delta.
Across platforms, disclose ABI-induced demand differences instead of silently
resizing the input to make requested bytes equal.

## A disjoint allocator byte ledger

At observation point `t`, all quantities below are integer bytes at one named
allocator boundary and refer to the same instant:

| Symbol / field | Definition |
|---|---|
| `L / live_requested_bytes` | Sum of original requested sizes of currently live allocations at this boundary. Record the request before allocator rounding. |
| `C / live_block_bytes` | Storage spans occupied by those live allocations, including their rounding and inline headers. |
| `Q / withheld_bytes` | Freed block spans still withheld from allocation by quarantine or pending reclamation. Freed blocks cease contributing to `C` immediately. |
| `F / reusable_bytes` | Free storage currently available to this allocator, subject to its size/alignment/arena rules. It need not satisfy every request. |
| `M / metadata_bytes` | Dedicated bookkeeping storage outside `C`, `Q`, `F` and `S`; include spatial tables as well as protection tables. |
| `S / snapshot_bytes` | Additional storage reserved for preserved payload copies. The original allocation stays in its own category. |
| `Z / structural_unused_bytes` | Known backing tails/alignment gaps unavailable for allocations and not in another category. |
| `B / backing_bytes` | Total storage capacity assigned to this ledger, including separately allocated tables and snapshots. |

The checked identities are `B = C + Q + F + M + S + Z`, `0 <= L <= C`, and
`H = B - F = C + Q + M + S + Z`. Label `H` **allocator bytes unavailable for
new allocations**, never RSS. Record `B` and `F` beside `H`: a fixed pool can
have constant backing while availability changes substantially. Free-list
links stored inside free blocks are part of `F`, not a second copy in `M`.
Count nested allocator pools once; do not add inner blocks to the parent reservation
that already contains them. Observers and platform metadata have separate ledgers.

`C-L` is live-block overhead, including headers. Only a separately observed
padding/size-class component may be called internal fragmentation. A residual
must not be silently assigned to `Z`: if any component or backing size is
unknown, record `null` and mark the ledger incomplete. A verified zero is `0`.
Validate grant/return accounting and the partition independently; deriving
`F` as a residual alone does not validate the allocator's free capacity.

This ledger measures storage capacity, not touched or resident pages. Store
virtual reservations, mapped/committed pages, RSS and attributable kernel
metadata in distinct fields with documented OS-specific semantics. Never add
them to `B` when they cover the same memory. Any claim about total system
memory requires the omitted layers, including Capstone nodes and PoisonCap
revoker state. A missing layer is not zero overhead.

## Four figure slots and exact endpoints

| Experiment | Primary endpoint and plot | Supporting interpretation |
|---|---|---|
| A: fixed-demand churn | `H`, `Q`, `F` and `B` at each release over 16 useful work units, plus exact intra-unit `max H` where observed. Plot cumulative allocated interval union separately. | Does the amount of storage tied up by the allocator stay bounded over the observed horizon at stable live demand? A flat history inside a fixed-size pool is not evidence of unlimited scalability. |
| B: live-set scaling | Plot `max H` and `max B` for each input against its observed `max L`; show metadata and quarantine contributions at the actual `H` peak. | This is allocator-capacity overhead versus live demand. It is not physical working-set scaling. Include the common input labels because ABI can change `L`. |
| C: burst/recovery | Show post-release `Q`, `H`, `F` and `B` before and after the burst. Report recovery in subsequent reference units for each quantity separately. | Reuse availability can recover without returning backing to the OS. A process that has not recovered by the observation end is censored, not a proven leak. |
| D: fixed budget | Exact completed units and status at every predeclared, identically charged budget. | Report the smallest **tested** successful budget. Neither a kernel panic nor an uncharged reservation establishes an OOM bound or a true minimum. |

Capture the simultaneous ledger vector when `H` reaches a new maximum.
`max(C+Q+M+S+Z)` is generally not `max C + max Q + max M + max S + max Z`.
Distinguish exact event-updated high-water counters from phase-end samples;
sampled maxima are lower bounds on intra-phase maxima. Never draw a stacked
peak bar from unrelated component maxima.

For C, freeze the tolerance before protected runs. For each quantity `X`, use
the maximum pre-burst reference post-release value plus
`max(one declared allocator quantum, 0.05 * pre-burst median X)` as its upper
recovery threshold. Report the first post-burst reference unit `k` for which
both `X[k]` and `X[k+1]` are within it. With eight follow-up units, `k` can be
1 through 7; otherwise report “no sustained recovery observed through unit 8”.
Also require live requested demand to return to its reference envelope.
Returning to or below the envelope is sufficient; do not force reclamation
or GC outside the common workload to produce recovery.

## Primary presentation: protected / original within each platform

For each application and declared memory metric `X`, use these two factors
as the primary comparison:

```
r_C = peak X(Capstone + Sublet) / peak X(Capstone spatial)
r_P = peak X(PoisonCap temporal) / peak X(PoisonCap spatial)
overhead_percent = 100 * (r - 1)
```

Both platform baselines are `1.0x`. A factor of `1.2x` means 20% additional
memory for the stated metric and workload. Main figures show the two protected
factors for each application with a common `1.0x` reference line; absolute
spatial/protected byte values accompany them. Compute each ratio within a
matched repetition block before reporting median/range, rather than dividing
separately aggregated medians. Both runs must complete the same planned
horizon, and the baseline must be nonzero.

Choose `X = H` for allocator capacity unavailable to new requests and report
`X = B` separately for assigned backing. Do not mix these in one ranking or
silently substitute RSS in one platform. Equal fixed reservations make the
`B` ratio exactly one by construction; they do not show equal usable capacity.
Do not divide quarantine by spatial quarantine, which is normally zero.

Before labeling this an original/protected comparison, record `baseline_kind`
and require the same definition on both platforms:

* `ported-original`: the minimally adapted spatial port before changes needed
  by the nested protection implementation. Its ratio includes required layout,
  table, snapshot and quarantine costs. This is the primary **full adaptation
  cost** question requested for the paper.
* `adapter-spatial`: the protected adapter's layout with temporal actions
  disabled, retaining the same tables and policy-independent configuration.
  Its ratio isolates the incremental temporal policy cost only when both
  platforms provide that kind of control.

The present Capstone spatial allocator and PoisonCap spatial adapter do not
have the same baseline kind: PoisonCap's denominator already contains external
tables. Their ratios may be described individually but must not be ranked as
equal-scope full adaptation overheads. Add an original-layout PoisonCap spatial
control for the full-cost question, retaining the existing adapter-spatial
control and the four-arm results. If only incremental temporal effects are
desired, a corresponding Sublet adapter-spatial control is needed instead.
Do not replace the missing control by subtracting an estimated table size:
changing the layout may change allocation behavior. Preserve the exact
baseline identity in the legend and show absolute ledgers alongside ratios.

With matching baseline kinds, `r_C < r_P` supports a lower **relative increase
for the measured metric** under these builds and policies. It does not alone prove lower total bytes,
physical memory, or an architecture-independent advantage. An optional
`r_C / r_P` summarizes the ratio of these factors; it must not replace the two
factors or be called the ratio of added overheads.

Also report `peak X_protected - peak X_spatial` in bytes. This difference of
peaks differs from the maximum pointwise difference at matched unit boundaries;
label the latter explicitly if used.
Do not normalize the two arms by their separate live-byte peaks and then
subtract those ratios: differing denominators can hide differing work.
For a zero denominator, report bytes and an undefined ratio.

## Reuse: two denominators that must remain separate

Maintain a monotonically increasing index `j` for positive-size allocation
and reallocation **attempts** at the inner boundary, including failures.
Zero-size calls and frees have separate API counters. A failed realloc leaves
the old lifetime live. An in-place realloc changes demand but is not a newly
reused start. A moving realloc retires the old lifetime and creates a new one;
record any transient overlap in capacity accounting. Explicit reset/GC releases
must retire the affected inner lifetimes, not merely emit one outer free.

1. **Allocation-side start reuse fraction.** Of all successful new-lifetime
   allocations, how many return a start belonging to an earlier retired
   lifetime? The denominator includes first-use starts. Report by useful unit
   and cumulatively. This measures placement decisions, not how quickly every
   retired allocation becomes reusable.
2. **Retirement-side start reuse within a fixed horizon.** For each retired
   lifetime `i`, let `f_i` be the attempt index at retirement and `d_i` the
   distance to the first later new lifetime with exactly the same start.
   Predeclare `Horizon = 4096` attempts for the main curve. Include only
   retirements with a complete 4096-attempt follow-up, regardless of whether
   reuse occurred early. On that fixed cohort, report
   `R(h) = count(d_i <= h) / count(eligible retirements)` for
   `h = 1, 8, 64, 512, 4096`. Retirements without observed reuse remain in the
   denominator. Separately report the cohort size, eligible unreused count,
   and late retirements excluded for insufficient follow-up. If the cohort
   is empty, this metric is unavailable; do not choose a smaller horizon
   after seeing an arm's result.

These curves need not reach one. A CDF computed only from reused blocks is
conditional on reuse and must not be substituted for `R(h)`. The fixed cohort
describes releases sufficiently early in that run, not all possible future
releases. An incomplete/crashed run is not admitted as a successful smaller
cohort comparison. Keep its observed prefix as diagnostic data.

Exact-start reuse can miss reuse after splitting/coalescing, and a smaller
allocation at an old start does not reuse the entire old block. Therefore
also report, per process, distinct starts `D(u)` and the byte length `V(u)`
of the union of all returned usable address intervals through unit `u`.
Compute the union, not `max address - min address`; do not union ASLR addresses
across processes. These are historical virtual-address measures. Neither
proves page residency, cache locality or lower physical fragmentation. Lack
of observed reuse can reflect allocation policy or request sizes even when
capacity is already reusable; use `Q`/`F` to distinguish availability from
actual selection.

## Fragmentation and working-set claims

A fragmentation diagnosis needs the failed request's size, alignment, target
arena, reusable free-block geometry and allocator policy at that point. Total
free bytes alone are insufficient: quarantine is unavailable capacity, and
free capacity in another arena may be inaccessible. For a fixed buddy arena,
record reusable block counts by order and the request's required order.
Failure with enough total reusable bytes but no policy-eligible block is a
request-relative external-fragmentation observation. Preserve size/arena
restrictions; `1 - largest_free_block / free_bytes` alone is not a universal
fragmentation score.

Allocation lifetimes measure the live allocation set. A memory-access working
set requires page or byte accesses in a defined window; we do not currently
observe that. Use “live requested bytes”, “allocator backing”, and “historical
address union” on the axes instead of the ambiguous “working set”.

## Recording, validation and reporting

Each record identifies run, paired block, app/case/input, arm, allocator
boundary, work unit, phase, attempt index, configuration/build/policy hashes,
observer version, sampling mode, and completion/oracle status. Emit raw integer
bytes, not rounded MiB. Record observer reserved bytes, capacity and overflow;
integer address history must not retain guest capabilities or enter the
measured allocator's allocation stream. Separately check observer-on/off
useful outputs and coarse allocation counters. Repeat the Capstone collector
invariance control for every new workload; matching earlier ports is insufficient.

Three process repetitions initially establish reproducibility. Show all run
points and median/range, stating `n`; do not treat thousands of allocations
as independent replicates or invent confidence from deterministic repeats.
If inferential statistics are needed, predeclare independent inputs/seeds and
enough paired runs for a justified uncertainty analysis. Use identical seeds
within a paired block, freeze exclusions beforehand, and retain failures in
the denominator. A retry is a new attempt with a reason; it does not replace
the failed attempt. Do not aggregate app rankings until complete per-app
results and coverage are visible.

Every figure identifies scope, allocator/policy, build profile, input, useful
work horizon, repetitions, exact versus sampled peaks, and missing metadata.
Publish the planned-cell/status table, compact records, source/log hashes and
plot command beside it. No smooth extrapolation past a failed run or the
measured horizon. The current SQLite pilot lacks build parity, a complete
ledger and inner reuse cohorts; it does not yet qualify these paper endpoints.

## Relationship to prior evaluation

Allocator evaluation must preserve real workload structure rather than
assume independent random requests; see Wilson et al.,
[Dynamic Storage Allocation: A Survey and Critical Review](https://www.cs.cmu.edu/afs/cs/academic/class/15213-f98/doc/dsa.pdf).
SQLite's [memory-allocation documentation](https://sqlite.org/malloc.html)
describes configured memsys5 heap/minimum-request parameters and conditional
allocation guarantees. Use the pinned 3.22 source for this study's actual
behavior; current documentation is background, not proof about our port.

[PoisonCap §5.5](https://arxiv.org/html/2605.13210v1#S5.SS5) reports SQLite
phase performance with incomplete nested-revocation coverage and uses SPEC's
libc allocation path for its other evaluation. Our proposed endpoints measure
memory availability at the internal boundary across repeated useful work.
They are a complementary experiment; prototype failures and differences in
policy do not establish an architectural lower bound for PoisonCap.
