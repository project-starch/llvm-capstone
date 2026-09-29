# Reuse figure: six complete applications

Review [the PNG](by-metric/00-reuse-distributions.png) or
[the PDF](reuse-distributions.pdf). Perl joins SQLite, mruby, FFmpeg,
PostgreSQL and CPython; the other five applications retain their observations,
and regenerating the [five-application figure](../reuse-five-applications-20260928/README.md)
with the same script still reproduces its CSV and PNG byte for byte. There are
six applications and eight workloads.

For each workload the left split violin compares Capstone spatial/Sublet;
the right compares CheriBSD spatial/PoisonCap. The beige half is each pair's
own spatial control. Width is conditional mass in the measured logarithmic
reuse-gap bin. All three complete process histograms contribute to each shape;
there is no fabricated KDE or interpolated sample. The CSV retains counts and
all-issues denominators, including arms with no observed reuse.

Perl's boundary is its SV-head allocator, the 48-byte head of every value.
Its Sublet and control median observed gaps both lie in [2,3], as does its
PoisonCap spatial control; its PoisonCap median lies in [4096,8191]. Observed
reuse shares are 63.74% in the Capstone pair and 57.27% versus 63.74% in the
CheriBSD pair. Both spatial controls reissue upstream's slots in upstream's
order, so the short gaps are Perl's own LIFO free list. The workload is the
study's `records.pl 512 3 0` run by the complete interpreter; it is not a
standard benchmark score, and the result is not a total-memory or physical
working-set ranking.

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-application-memory-paper.py \
  --reuse-review capstone/experiments/study/results/published-policy-20260928 \
  --reuse-campaign capstone/experiments/study/results/postgres-reuse-four-arm-20260928 \
  --reuse-campaign capstone/experiments/study/results/cpython-reuse-four-arm-20260928 \
  --reuse-campaign capstone/experiments/study/results/perl-reuse-four-arm-20260928 \
  --out capstone/experiments/study/results/reuse-six-applications-20260928
```

The new application's adapter, builds, raw processes and validator are in the
[Perl campaign](../perl-reuse-four-arm-20260928/README.md). The figure's
provenance binds all input summaries; every application retains its raw
processes and scoped method.
