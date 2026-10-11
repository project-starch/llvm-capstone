# Reuse and retention in real application workloads

The paired matrix contains 12 workloads per platform, three repetitions each:
**72/72 pass**, with default CheriBSD allocator/revocation settings preserved.
Six new Capstone runs and nine new CheriBSD runs extend the existing data;
three earlier Capstone FFmpeg-64 runs are reused unchanged. These are actual
application operations, not allocation-event replay. The configured FFmpeg
Matroska/MPEG-4 decoder and mruby are the two matched applications.

The useful advantage is prompt reuse of allocation addresses and reduced
post-release allocator retention in many of these workloads. It is not a
hardware speedup, a proof of smaller total memory, or a universal fragmentation
win. [Paper analysis, metric definitions and remaining ports](../../memory-behavior.md)
explain the scope and what the two Cornucopia papers already evaluate.

## Address reuse

| Application workload | Capstone distinct starts | Default CheriBSD | Ratio | Capstone reuse within 8 calls | CheriBSD |
|---|---:|---:|---:|---:|---:|
| FFmpeg: 30 frames × 1 streams | 178 | 730 | 4.10× | 55.34% | 0.00% |
| FFmpeg: 30 frames × 4 streams | 178 | 2,170 | 12.19× | 56.06% | 0.00% |
| FFmpeg: 30 frames × 16 streams | 178 | 2,803 | 15.75× | 56.24% | 0.00% |
| FFmpeg: 30 frames × 64 streams | 178 | 2,982 | 16.75× | 56.29% | 0.00% |
| FFmpeg: 150 frames × 1 streams | 178 | 1,897 | 10.66× | 67.33% | 0.00% |
| FFmpeg: 600 frames × 1 streams | 181 | 2,334 | 12.90× | 70.68% | 0.00% |
| mruby: 32 records × 8 batches; retain 256 | 2,308 | 4,551 | 1.97× | 4.74% | 0.00% |
| mruby: 128 records × 8 batches; retain 256 | 4,503 | 10,909 | 2.42× | 2.76% | 0.00% |
| mruby: 128 records × 8 batches; retain 4096 | 21,150 | 33,988 | 1.61× | 0.94% | 0.00% |
| mruby: 128 records × 32 batches; retain 4096 | 21,150 | 52,660 | 2.49× | 1.06% | 0.00% |
| mruby: 512 records × 8 batches; retain 256 | 14,554 | 36,315 | 2.50× | 1.21% | 0.00% |
| mruby: 512 records × 16 batches; retain 256 | 14,554 | 44,552 | 3.06× | 1.19% | 0.00% |

All table entries are identical across the three repeats. Ratios divide
CheriBSD distinct starts by Capstone distinct starts; they are not byte or RSS
ratios. Exact start addresses are counted, not partial overlaps or pages.

At 64 FFmpeg streams, both platforms perform 46,720 allocation calls and peak
at 164 simultaneously live observed blocks. Capstone uses 178 starting
addresses across the whole run; CheriBSD uses 2,982. Distinct starts divided by
peak live blocks is 1.085 versus 18.183. After its first stream, Capstone needs
no additional starting addresses in this workload.

For mruby's 512-record, 16-batch workload, 69.97% of Capstone allocations reuse
an address within 4,096 allocation calls; CheriBSD's fraction is zero. Overall
reuse is 76.16% versus 27.01%. For 128 records, 32 batches and a retained graph
of 4,096, overall reuse is 59.84% versus zero. In-place realloc is excluded from
reuse. These are call distances and allocation fractions, not time or cache
locality measurements.

## Retention and its counterexample

All FFmpeg release boundaries have zero observed requested live bytes on both
platforms. Occupied Capstone buddy blocks also return to zero. In the 64-stream
CheriBSD run, the post-release jemalloc allocated ledger ranges from 4,269,768
to 8,789,576 bytes. This ledger includes quarantine and private allocations;
it must not be labeled as an exact quarantine counter.

For mruby with 512 records, 16 batches and 256 retained records, post-release
Capstone occupancy is 1,092,608–1,223,680 bytes; CheriBSD allocated is
2,282,440–11,917,272 bytes. After the fourfold burst, Capstone returns to its
pre-burst occupancy at the next ordinary batch. CheriBSD also collects during
this run: its drops remain visible in the complete curve.

The large retained graph is a counterexample to a blanket memory claim.
For 128 records, 8 batches and 4,096 retained records, Capstone ends at
10,521,600 occupied bytes, versus 9,214,680 jemalloc allocated bytes on CheriBSD.
Extending the same workload to 32 batches keeps Capstone occupancy constant;
CheriBSD's ledger grows to 11,600,184. The curve shows the crossover, rather than
selecting only a favorable endpoint. Rounding, retained objects and quarantine
contribute differently to these ledgers.

Capstone still reserves an 8 MiB logical pool / 16 MiB grant for FFmpeg, and a
16 MiB pool / 32 MiB grant for mruby. Its static allocator tables occupy
1,343,636 and 2,687,132 bytes respectively, inside the 32 MiB application-data
reservation. Both observers use 1,572,864 static bytes. These reservations,
code, stack, driver caches and node metadata prevent treating the ledger plots
as total-memory or RSS comparisons. FFmpeg's different input-path lengths add
34 requested bytes on Capstone; mruby has one additional observed allocation
and 1,465 additional peak requested bytes. Allocation counts and useful outputs
are otherwise consistent with the existing comparison contract.

## Does the temporary node sweep create these results?

The complete Capstone phase samples match controls using QEMU before the
in-process sweep change: application image, arguments, environment, output oracle,
output hash, allocation metrics and allocator metrics are checked. There are
36 passing control runs and **108/108 equal control/primary-repeat pairs**.
All 12 workload configurations have passing controls. This is a direct equality
check for the recorded workloads, not a general claim for every application or
for a future hardware implementation.

The original controls include 21 passing 65,536-node runs and twelve passing
262,144-node mruby runs. Six original capacity failures remain counted in the
control status summary. Three new controls run the extensions on the old QEMU
with 262,144 nodes: each completes without an in-process collector, and the
largest consumes 161,784 additional node allocations. The old collector still
runs at process teardown, after the application's measurement phases.

## Every attempt and instrumentation limits

- Main matrix: Capstone 36/36 and default CheriBSD 36/36 pass.
- Additional old-QEMU controls: 3/3 pass; prior controls are preserved separately.
- A 128-batch mruby extension filled the 65,536-entry address-history table.
  Its first attempt timed out with invalid counters; the next attempt was
  interrupted when this was diagnosed. Neither is an application failure or a
  usable comparison. The remaining repeats were not attempted.
- The bounded extensions use 16 batches for 512 ordinary/256 retained records,
  32 batches for 128 ordinary/4,096 retained records, and 64 FFmpeg streams.
  Their total allocation counts fit within the observer even without reuse.
  No observer size or allocator policy was changed to obtain the final matrix.
- The initial launch lacked the Python pexpect dependency and ran no application.
  The existing experiment virtual environment supplied it.
- Four analysis tests pass, checking the denominator, retained failure records,
  all-repeat control comparisons, and rejection of invalid observation phases.

## Figures, data and reproduction

- [Prompt reuse distributions](figures/reuse-window.pdf)
- [Distinct address history](figures/address-history.pdf)
- [Complete post-release curves](figures/release-retention.pdf)
- [Retained-graph tradeoff](figures/retained-graph-tradeoff.pdf)

PNG versions sit beside each PDF. [summary.json](summary.json) contains all
metric medians and ranges; [attempts.jsonl](attempts.jsonl) contains all main
attempts with phase aggregates, and [invariance.json](invariance.json) contains
all control checks. [provenance.json](provenance.json) and [inputs.json](inputs.json)
identify source data, policies and binaries. [archive.json](archive.json) identifies
the external raw evidence, including rejected attempts; guest keys are excluded.

Reproduction uses the existing application runners plus `analyze-reuse.py` and
`plot-reuse.py`. Commands and explicit point matrices are in the raw artifact.
Use the Python environment with pexpect for guest runs and matplotlib for plots.
No paper manuscript, guest allocator policy or application code was changed.
