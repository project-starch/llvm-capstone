# SQLite paper-layout preview

These figures reformat the existing complete SQLite 3.22.0 `speedtest1 main`
repeated-work campaign. No new application execution is represented. They
illustrate the [application figure contract](../../../../../docs/plans/application-memory-figures.md),
not a completed multi-application evaluation. Each vector figure is 7.05 inches
wide. The failed burst is outside these repeated-work figures and remains in
the [original experiment record](../README.md#burst-result).
The separately measured [four-arm reuse-gap figure](../../sqlite-reuse-gaps-20260927/reuse-gaps.pdf)
now fills the release-to-reissue slot for SQLite memsys5.

[All three figures as one PDF](sqlite-layout-preview.pdf).

## Captions

1. [Paired memory cost](01-paired-memory-cost.pdf). SQLite `main --size 1`,
   lookaside disabled. Each protection mechanism is divided by its own
   original-layout allocator on the same platform. Left: cumulative allocated
   address coverage after one warmup and 16 measured units. Right: peak
   selected allocator bytes over the 16 measured units. Points and whiskers
   denote median and full range of three paired repetitions; all coincide.
   Original absolute denominators are 1,280,960 / 1,283,008 address bytes and
   1,404,831 / 1,406,879 peak selected bytes for Capstone / CheriBSD respectively.
   Sublet preserves the baseline's address coverage; its selected footprint
   ratio exceeds PoisonCap's. Selected bytes exclude node/kernel storage.
2. [Allocated address coverage](02-address-coverage.pdf). Cumulative union of
   allocated arena intervals, sampled after each complete workload unit.
   Unit 1 (shaded) is the warmup. Each unit opens and closes a fresh database;
   allocator state survives. Dashed/open markers indicate original layout,
   solid/filled markers protection. Sublet and original coincide. Both platform
   pairs reach a plateau within this observed run. Lines join checkpoints;
   three repetitions coincide. This is address coverage, not resident memory
   or a reuse-gap distribution.
3. [Repeated-work footprint](03-repeated-work.pdf). Selected allocator bytes
   at the exact per-unit event peak (left) and after database close (right).
   The quantity includes rounded live spans, quarantined spans and the
   specified allocator tables, simultaneously accounted. Unit 1 is warmup;
   three repetitions coincide. Sublet has no quarantined spans at close but
   keeps its larger tables; PoisonCap retains a changing quarantine. Nodes,
   kernel state and other excluded storage are specified in the parent methods.
   The two original curves nearly coincide. These curves do not establish
   total-memory scalability.

PoisonCap uses the documented corrected full-queue revocation path throughout.
Both pairs have 129,055 allocatable 64-byte atoms. All 12 processes complete;
every complete unit matches the native SQL phase oracles. The same input and
build-comparability checks as the parent campaign apply. Measured optimization
is `-O0`; qualification at the intended final optimization remains necessary.

## Reproduce

From the study worktree, with matplotlib available in the Python environment
(the local measurement environment is `/tmp/capstone/application-memory/venv`):

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/experiments/study/plot-sqlite-normalized.py \
  capstone/experiments/study/results/sqlite-normalized-memory-20260927 \
  --paper-layout
```

The existing parser revalidates raw ledgers and completed-unit result oracles
before rendering. The preview also requires all three complete churn repetitions
in all four arms. [provenance.json](provenance.json) records input and generator
hashes. Original exploratory figures and measurements remain separate.
