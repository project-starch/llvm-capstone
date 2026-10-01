# Three complete-application reuse panels

The [paper-width CDF preview](reuse-cdf.pdf) renders three previously
qualified four-arm campaigns through one visual contract. Every line comes
from three independent complete application processes with the exact
application-output oracle and identical release-gap histogram within its
arm. The denominator is **all successful allocation or lease issues at the
named inner boundary**, including starts never reissued. X counts subsequent
issues from release to reissue. The panels use separate X limits because
FFmpeg's observed gaps end much earlier; all use the same 0–100% Y scale.

| Application workload | Inner boundary | Issues per process | Gap ≤15, original → protected on Capstone | Gap ≤15, spatial → PoisonCap on CheriBSD |
|---|---|---:|---:|---:|
| SQLite 3.22.0 `speedtest1 main --size 1`, 17 units | memsys5 | 550,137 | 73.116% → 73.116% | 73.113% → 0.052% |
| mruby 4.0.0-rc2 AO width 16 | GC slots | 915,981 | 1.217% → 1.224% | 1.224% → 0.000% |
| FFmpeg 9.0.1, 16 × 30-frame streams | pool leases | 6,112 | 90.461% → 90.461% | 90.461% → 90.461% |

The source campaigns, with raw archives, build comparisons and boundary
definitions, are [SQLite](../sqlite-reuse-gaps-20260927/README.md),
[mruby](../mruby-gc-memory-20260927/README.md) and
[FFmpeg](../ffmpeg-reuse-gaps-20260927/README.md). The
[plot generator](../../plot-cross-application-reuse.py) checks all four arms,
three process repetitions, histogram reconciliation and repetition equality
before writing [derived data](data.json) and vector PDF. This is a visual
preview of existing data, not a new campaign or a selected six-application
paper result.

The allocator boundaries and useful work differ by panel. In SQLite and
mruby, the tested PoisonCap policies postpone reissue relative to their own
spatial controls; the FFmpeg adapter sweeps synchronously and shows identical
lease gaps in all four arms. Address reuse is neither physical working set
nor total occupied memory. The SQLite and mruby selected-metadata costs and
FFmpeg's adapter-specific poison/clear/copy spans remain separate results.
