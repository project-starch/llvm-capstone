# Native ARM64 survey, 2026-10-05

All 96 declared processes completed and passed their functional oracles:
eight applications, 16 input/configuration profiles, three fresh baseline
processes and three fresh observed processes per profile. These are native
Linux ARM64 executions under Docker Desktop. No Capstone or CHERI execution
is involved.

The observations establish within-backing address reuse in these workloads.
They do not establish vulnerability prevalence, protection effectiveness,
processor performance or application speed under temporal protection.

## Artifacts

- [Reuse fractions, PDF](reuse-fractions.pdf), [SVG](reuse-fractions.svg)
- [Reuse gaps, PDF](reuse-gaps.pdf), [SVG](reuse-gaps.svg)
- [Per-profile summary CSV](summary.csv), [JSON](summary.json)
- [Every baseline oracle and observed run](runs.json)
- [Native environment and build commands](environment.json)
- [Workload selection and exact parameters](../../workloads.json)
- [Instrumentation semantics and limitations](../../README.md)

`runs.json` contains the validated counters and hashes. It omits raw logs,
process identifiers and the incomplete byte diagnostics. Every raw result
record is retained in scratch and identified by its SHA-256. Sources, input
archives and upstream workload programs remain outside the repository.

## Reading the results

`I/A` is within-backing reuse divided by all successful object issues. `I/R`
uses only same-start reissues in the denominator. The figures show both.
A percentage without its allocator, workload and denominator is not a result
of this experiment.

Some useful examples, rounded means over three runs:

| Profile and allocator | I/A | I/R |
| --- | ---: | ---: |
| SQLite main, lookaside | 99.96% | 100.00% |
| PostgreSQL adapted pgbench, AllocSet | 55.26% | 55.51% |
| Python JSON loads, pymalloc | 99.09% | 100.00% |
| mruby AO, GC slots | 99.33% | 99.63% |
| Perl Binary Trees, SV heads | 98.93% | 100.00% |
| FFmpeg resolution change, AVBufferPool | 92.00% | 98.81% |
| tshark HTTP, wmem block_fast | 0.00% | undefined |
| tshark SIP/RTP, wmem block_fast | 98.48% | 100.00% |
| memcached memtier, slabs | 96.15% | 100.00% |

The small HTTP capture has no observed block_fast reissue. It remains in the
results. The other Wireshark family, wmem block, has about 39,000 to 42,000
issues in each whole-process run, mostly first uses. The initialization
contribution has not been isolated. Its I/A is below 0.5% even
though all of its observed reissues stay inside a live backing allocation.
This is why I/R alone would give an incomplete picture.

PostgreSQL also reuses addresses across allocator instances. The local-gap
distribution covers about 75.7% of its within-backing reissues. The remainder
has no same-instance allocation clock and is not assigned a synthetic gap.
Every other exercised family/profile with within-backing reuse has complete
local-gap coverage in this campaign.

Twelve of the 17 instrumented families issue objects in the selected profiles.
PostgreSQL Generation, Slab and Bump, and Wireshark simple and strict are not
exercised as object allocators. This is a coverage limitation, not a zero-reuse
finding about those mechanisms. PostgreSQL and Wireshark results must remain
attached to the allocators and configurations actually reached.

All observed families have zero unknown backing at allocation or reuse and
zero unmatched retirements. Every run satisfies `A = F + live` and
`R = inside + outside + unknown`. The exact source-hook audit passes for all
eight baseline and all eight observed trees. Recorder controls cover a backing
release/reacquire at the same address, bulk retirement, successful in-place
resize, failed resize, unknown backing, cross-instance gaps, concurrent
metadata updates and rejection of a live-object overlap.

## Rebuild the figures

Run the exporter using a Python environment with matplotlib 3.10.8. The
committed counters are sufficient, so the figures do not depend on local logs:

```sh
python3 capstone/experiments/native-lifetime-survey/analyze.py \
  --from-export capstone/experiments/native-lifetime-survey/results/native-arm64-20261005 \
  --output /tmp/capstone/native-survey/replotted \
  --plots
```

The exporter rejects missing or duplicate cells, incomplete observations,
inconsistent lifetime counters and oracle mismatches. Whiskers show the
observed range over three runs, not a confidence interval. Gap curves pool
events from those three runs and report CDF values at logarithmic bin upper
bounds. The Wireshark gap panel contains one curve per input with reuse,
grouped by allocator color and line style.

For a stronger paper claim, the next measurement should separate steady-state
packet dissection from Wireshark startup and add independently selected SQL
workloads that reach the currently unexercised PostgreSQL families. These
extensions are not prerequisites for using the current observations with
their stated scope.
