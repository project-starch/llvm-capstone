# First reuse figure: five complete applications

Review [the PNG](by-metric/00-reuse-distributions.png) or
[the PDF](reuse-distributions.pdf). CPython joins SQLite, mruby, FFmpeg and
PostgreSQL; the prior four applications retain their observations. There are
five applications and seven workloads. Perl still needs an inner-allocator
comparison and is absent.

For each workload the left split violin compares Capstone spatial/Sublet;
the right compares CheriBSD spatial/PoisonCap. The beige half is each pair's
own spatial control. Width is conditional mass in the measured logarithmic
reuse-gap bin. All three complete process histograms contribute to each shape;
there is no fabricated KDE or interpolated sample. The CSV retains counts and
all-issues denominators, including arms with no observed reuse.

CPython's Sublet and control median observed gaps both lie in [2,3]. Its
PoisonCap median lies in [4096,8191], versus [2,3] for its own control. Observed
reuse shares are 63.48% in the Capstone pair and 55.77% versus 65.13% in the
CheriBSD pair. This supports prompt reuse for this workload, not a total-memory
or physical working-set ranking. CPython JSON/GC and PostgreSQL SQL remain
qualification inputs, not pyperformance or pgbench scores.

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-application-memory-paper.py \
  --reuse-review capstone/experiments/study/results/published-policy-20260928 \
  --reuse-campaign capstone/experiments/study/results/postgres-reuse-four-arm-20260928 \
  --reuse-campaign capstone/experiments/study/results/cpython-reuse-four-arm-20260928 \
  --out capstone/experiments/study/results/reuse-five-applications-20260928
```

The new application's source archive, kernel repair, transferred quarantine
policy and raw validator are in the [CPython campaign](../cpython-reuse-four-arm-20260928/README.md).
The figure's provenance binds all input summaries; every application retains
its raw processes and scoped method. Rotated labels are included in the export
bounds so the added application name is not clipped.
