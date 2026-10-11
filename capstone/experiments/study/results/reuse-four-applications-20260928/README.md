# First reuse figure: four complete applications

Review [the PNG](by-metric/00-reuse-distributions.png) or
[the PDF](reuse-distributions.pdf). This extends the approved first figure
with PostgreSQL; SQLite, mruby and FFmpeg retain their existing observations.
There are four applications and six workloads, not six applications.

Each workload has two split violins: its Capstone spatial/Sublet pair and its
CheriBSD spatial/PoisonCap pair. Width is the conditional mass of **observed**
same-start reuse in each logarithmic gap bin; the vertical coordinate counts
successful new handouts between release and reuse. Each shape pools three
complete process histograms. Three existing application campaigns have
identical per-arm repetitions; PostgreSQL protected runs vary slightly.
The CSV reports pooled counts and denominators; raw per-process values remain
in the source archives. There is no interpolation or fabricated KDE sample.

The new PostgreSQL input is complete-backend `work.sql`, a qualification
workload rather than pgbench. Its within-platform controls differ in layout
across platforms; compare each protected half against its own adjacent control.
Protected median observed gaps are [4,7] with Sublet and [8192,16383] with
PoisonCap, versus [4,7] in both controls. Reuse shares are 80.89% versus 83.29%
for the Sublet pair, and 36.20–36.27% versus 83.28% for the PoisonCap pair.
These are workload-scoped reuse results, not total memory or working-set claims.
CPython and Perl are still missing complete protected measurements and do not
appear in this figure.

Reproduce from the study checkout after sourcing the test environment:

```sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-application-memory-paper.py \
  --reuse-review capstone/experiments/study/results/published-policy-20260928 \
  --reuse-campaign capstone/experiments/study/results/postgres-reuse-four-arm-20260928 \
  --out capstone/experiments/study/results/reuse-four-applications-20260928
```

The PostgreSQL validator rechecks its six new and six archived processes before
rendering. `reuse-provenance.json` binds the input summaries and archive; the
original three-application validators and raw archives remain in their source
campaign directories. No guest execution or trace replay occurs during plotting.
