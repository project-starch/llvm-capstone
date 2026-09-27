# Application memory behavior versus default CheriBSD

Compare the actual Capstone/Sublet ports with default CheriBSD purecap. Keep
application inputs and output oracles matched, preserve the guest's allocator
policy, and report every attempt. This study concerns memory behavior, not
security strength or emulator speed. The measurement and platform limits in
[comparison.md](comparison.md) apply throughout.

## What the papers establish

[Cornucopia (2020)](https://www.cl.cam.ac.uk/research/security/ctsrd/pdfs/2020oakland-cornucopia.pdf),
§VII and Figure 9, measures peak resident memory and discusses excess memory
from quarantine interacting with slab allocation. §VIII-A also evaluates early
physical-page release when whole pages are quarantined. It would be incorrect
to claim that this work ignores memory overhead, fragmentation, or physical
page reuse. Its results motivate investigating when freed storage becomes
available to subsequent application work.

[Cornucopia Reloaded (2024)](https://api.repository.cam.ac.uk/server/api/core/bitstreams/4c5357be-42dd-487d-93b2-48f68f195730/content),
§5, measures execution time, CPU time, bus traffic, peak RSS, and interactive
latencies. Figure 3 reports similar peak-memory behavior to Cornucopia; §5.5
reports mean allocated heaps and aggregate quarantine/revocation activity.
§7.2 explicitly identifies quarantine policy tuning, double-buffer pressure,
and detecting fully quarantined pages as further work. Its evaluation uses
snmalloc and a revocation shim on Morello. Our requested comparison uses
default jemalloc-based CheriBSD on RISC-V, so our results do not reproduce that
experimental configuration.

Neither evaluation presents the allocation-call-distance reuse distribution,
distinct allocation starts per repeated task, or the request-by-request
post-release ledger curves proposed below. These are additional questions,
not claims that the papers failed to evaluate their stated goals.

The checked [CheriBSD source](https://github.com/CTSRD-CHERI/cheribsd/blob/88f39900c32928d807dba245fba138808c666f34/lib/libc/stdlib/malloc/mrs/mrs.c#L765)
at `88f39900c32928d807dba245fba138808c666f34`, `quarantine_should_flush`, gates collection
on tracked allocated size reaching 8 MiB and then on the configured quarantine
fraction. Do not substitute the paper's differently described minimum-quarantine
rule for this source condition. Runs preserve defaults and record kernel
revocation state; no forced collection, allocator replacement, or policy sweep
is part of this comparison.

## Metrics that can be measured now

| Metric | Definition and useful question | Limit |
|---|---|---|
| Prompt address reuse | Fraction of successful allocation calls returning a previously freed start within 1, 8, 64, 512 or 4,096 allocation calls. Can new work reuse recently released addresses? | Allocation-call distance, not elapsed time or cache locality. In-place realloc is a continuation. |
| Distinct starts | Count of different starting addresses returned across a fixed number of completed tasks. Does repeated work keep using the same address set? | Not distinct pages, physical footprint or external fragmentation. |
| Distinct starts / peak live blocks | Normalize address history by the largest observed simultaneously live allocation count. | Describes historical address demand, not a storage lower bound or RSS. |
| Post-release allocator ledger | Sample occupied Sublet blocks or jemalloc allocated bytes after every task, beside common requested live bytes. Does a task release storage for subsequent work? | Different ledgers; jemalloc includes quarantine and private allocations, Sublet includes buddy rounding. Neither counts total platform memory. |
| Burst recovery | Plot the same ledgers before a fourfold burst and after subsequent ordinary tasks. Count completed tasks until returning to the prior level, if it occurs within observation. | Sampling is at task boundaries; absence of recovery within a run is not a permanent leak. |

Avoid ratios with a zero requested-byte denominator. Report absolute retained
bytes when a stream leaves no live observed allocations. A low live ledger
does not release Capstone's reserved pool to Linux. FFmpeg reserves an 8 MiB
logical pool from a 16 MiB grant; mruby reserves 16 MiB from 32 MiB. Static
allocator tables, the common 1.5 MiB observer, code/data/stack reservations,
node metadata and driver caching remain relevant to total-memory accounting.

## Independence from the temporary Capstone collector

The observer retains integer addresses and aggregate counts, not heap
capabilities or allocation events. The QEMU sweep recycles metadata identities;
these metrics observe application allocation decisions. Do not assume that
this makes every workload invariant: compare recorded phases across emulator
configurations.

`analyze-reuse.py` compares all passing primary repeats against each passing
control with the same workload. It requires identical application images,
output hashes, complete allocation samples and complete allocator samples.
The original 21 successful runs, twelve larger-node mruby controls and three
larger-node extension controls pass this check against all 36 primary Capstone
runs: 108 equal comparisons covering all twelve workloads. Failed controls
remain in the control status counts. This establishes equality for the recorded cases;
it does not prove equality for every application, a future reclaimer, or
time-dependent/concurrent workloads.

Timing, pause duration, cache/DRAM traffic, node-capacity scaling and whole-system
memory are excluded from the present advantage claims. In particular, long
execution made possible by the temporary collector is not evidence of the
intended hardware's sustainable metadata behavior.

## Workloads and extension across ports

Use real application operations and report common malloc-level metrics first.
Nested allocator storage is a separate ledger and must not be inferred from
outer malloc activity.

| Application | Repeated useful work | Release boundary |
|---|---|---|
| FFmpeg configured decoder | Decode identical independent streams, then vary stream length | Destroy the decoder and release its frames/buffers |
| mruby | Parse record batches while retaining an independent graph; include one fourfold burst | Drop the transient graph and run the workload's ordinary GC |
| Perl | Parse input batches with a retained hash and temporary records | Release transient references at each batch boundary |
| CPython | Parse JSON/object batches while retaining a dictionary | Drop each batch and collect as specified by the common workload |
| SQLite | Execute prepared query batches and churn temporary tables | Finalize/reset statements and release temporary objects |
| PostgreSQL | Repeated single-user queries with temporary contexts | Complete each query and release its transient contexts |

Only FFmpeg and mruby currently have matched default-CheriBSD application
measurements. The other rows are candidate workloads. PostgreSQL filesystem
coverage and the distinction between outer malloc and retained context pools
must be resolved before expanding its claim. The prepared tshark inputs still
need a complete dependency build.

The generic summary accepts the existing runners' output:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/applications/analyze-reuse.py \
  --capstone "$CAPSTONE_RUN/runs.jsonl" \
  --cheribsd "$CHERIBSD_RUN/runs.jsonl" \
  --control "$CAPSTONE_CONTROL/runs.jsonl" --out "$RESULTS"
python3 capstone/experiments/applications/plot-reuse.py "$RESULTS" \
  --out "$RESULTS/figures"
```

The plot script illustrates the checked FFmpeg/mruby workloads; the analyzer
groups by application, input size, batch count and retained-set size and can be
used with additional ports. Preserve failed attempts and observer errors.
