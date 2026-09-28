# Memory figures from complete applications

Status: proposed publication layout and measurement contract. The SQLite layout
preview uses previously collected data; it is not a new experiment or a
preregistered confirmatory result. This refines the presentation of the
[application campaign](application-memory-campaign.md).

## Experimental unit

Run the actual application and its benchmark body in the target environment.
The application's own control flow performs allocation, access, release, and
ordinary GC. Observe these events in place. Do not drive allocators from a
recorded request stream. A data file consumed by a plotter is a measurement
record, not an executable workload.

Use a benchmark's complete work unit: a SQL workload, transaction, decoded
stream, or interpreter benchmark body. Repeated units execute in one process
so allocator state, caches and quarantine survive. Preserve the benchmark's
natural resets; do not reset the allocator to manufacture recovery. For a
server, keep the server process alive across client work units. Pin input,
seed, application version, features, optimisation, correctness oracle, and
normal GC/cleanup policy. Report unavoidable ABI and platform differences.

The comparison has two pairs: Capstone original / Sublet and CheriBSD original /
PoisonCap. Each original retains necessary platform compatibility changes but
excludes protection-induced allocator restructuring. Retain an adapter-only
control separately if needed to explain that restructuring. Default CheriBSD
heap revocation does not substitute for PoisonCap's nested allocator path.

## Three main figures

Each figure answers one question across applications. Do not give every
application an unrelated dashboard. Keep a fixed application/workload order.
Where an application has several allocation boundaries, name the boundary
and give it its own panel; do not silently pool inner and outer events.

### 1. What memory price does protection impose?

Two aligned horizontal dot plots across the full text width, one application
and benchmark per row. Left: allocated address coverage expansion. Right:
peak occupied-memory expansion. Each row has Sublet and PoisonCap points,
normalised to their own original on that platform. The vertical reference is
1x. Dots and line segments show the median and full observed range of paired
repetitions. Baseline absolute bytes appear in an adjacent compact table.

For a scalar memory measure X, compute X_protected / X_original within each
matched repetition before aggregating. Peak footprint means
max_t H_protected(t) / max_t H_original(t), with the peaks taken over the same
work interval; it is not a ratio of sums of independent component peaks.
Keep per-application rows rather than a headline geometric mean over different
memory questions. A zero baseline uses a byte difference, never an epsilon
denominator. No confidence interval is inferred from deterministic reruns.

The intended right panel includes all attributable allocator/protection storage,
including nodes and revocation metadata. Until that ledger exists, a preview
must say **selected allocator bytes** and must not be described as total
memory overhead. A lower address-coverage factor does not establish a lower
total memory cost. Both results remain visible, including a tradeoff.

### 2. Does protection change address reuse?

Small multiples, one panel per application/allocator workload, normally three
columns at full text width. Each panel has the same four styles: blue for the
Capstone pair, orange for the CheriBSD pair; dashed/open markers for original,
solid/filled markers for protected. If four curves are too dense, use two rows
by platform, sharing axes. Colours never change meaning between figures.

X is the number of allocation events from release to reissue, logarithmic;
Y is cumulative qualifying reissues divided by **all successful allocation
events at that boundary**, from 0 to 100%. Use identical bins and limits.
The endpoint reports reuse frequency as well as the distribution of gaps.
Do not renormalise away never-reused allocations. Keep total event counts in
the companion table. A representative main-paper subset is chosen by allocator
type before the confirmatory results; the artifact retains every admitted case.

An address is an arena identity plus byte offset, not an across-process virtual
address. For same-start reuse, record the latest release of the prior object
at that start. If A allocations have completed when it is released and its
next allocation has index B, the gap is B-A; immediate reuse has gap one.
Treat bulk release as the end of the objects it releases. Define arena
destruction/recreation, moved realloc and in-place resizing explicitly;
in-place resizing is not a free/reuse event. Same-start reuse misses partial
overlap after splitting/coalescing, so pair it with interval-union address
coverage, not an assertion that it describes every reused byte.

### 3. Does memory stabilise, and does it recover after a burst?

Use the paper's compact 2x2 structure: columns are the two platform pairs;
rows are sustained work and burst/recovery. Each panel has its own original
and protected curves, common vertical units, and common work checkpoints.
The horizontal axis is completed useful work, never emulator wall time.
Mark baseline, burst, release and recovery phases unobtrusively.

Use a predeclared workload with bounded live demand for sustained work and a
natural application demand knob for the burst. First qualify configurations
where all four arms complete; then freeze the input and record new runs.
Rerun all four arms after changing configuration. Do not replace only the
unsuccessful arm with an easier input. Quarantined bytes, reusable cached
storage and protection metadata have a companion component figure. A cached
block becoming available is different from returning pages to the OS.

Plot enough complete units to expose settling and periodic reclamation;
choose the observation horizon in a qualification study rather than declaring
a fixed short plateau to be asymptotic. Report only the observed interval.
Use separate rows/figures for different memory quantities, never dual Y axes.
Failure during qualification is a diagnostic to resolve, not the positive
headline. Preserve its record. A capacity-failure result requires its own
controlled experiment with total budgets and policy sensitivity.

## Measurement semantics

At one boundary, distinguish requested live payload L, rounded live spans C,
withheld free spans Q, reusable free spans F, and allocator/protection metadata
M. For a fixed pool P, check C+Q+F=P. Internal rounding is C-L. Occupied
allocator storage is C+Q+M when all the listed components are disjoint and M
is fully counted; retained/reserved backing is a separate quantity. Do not
count a parent pool and its children's suballocations twice in the total.
Physical pages, logical node slots, and reserved capacity require separate
columns and an explicit conversion before combination.

Instrument event peaks and sample components at the same work checkpoints.
Do not sum independently attained maxima. Record observer storage outside the
allocator under test; use numeric IDs/offsets rather than capabilities that
could retain objects. Check observer capacity and output oracles. Validate
that observation does not change useful work or the allocation policy. Moving
work into the harness or forcing collection changes the experiment.

Cumulative address coverage is the union of allocated byte intervals up to a
checkpoint. It is monotone by definition and is not a leak detector. A separate
rolling union over a predeclared number of work units can show active address
coverage; compute it from a union, never by summing per-unit distinct counts.
Neither quantity is a physical working set. To claim working-set behaviour,
measure memory accesses or attributable resident/committed pages with compatible
semantics on both platforms. Post-close bytes alone are also not fragmentation:
measure rounding, available size classes, and allocability separately.

## Rendering and captions

- Export vector PDF at the final manuscript width, approximately 7.05 inches
  for a two-column figure; check labels at 100% size. Use 8-9 point labels,
  colour plus redundant dash/marker styles, and one legend per figure.
- Share limits for comparable small multiples, start byte/count axes at zero,
  and mark logarithmic or ratio axes explicitly. No 3D bars or broken axes.
- Use captions for version, input, repetitions, normalisation, accounting
  boundary, and the factual interpretation. Avoid long diagnostic footers
  inside panels. Configuration failures stay in the experiment ledger.
- Display repetition ranges as observed variation, not statistical confidence.
  With identical deterministic runs, say that they coincide. Do not treat
  allocation events within one process as independent experimental repetitions.
- Keep source hashes, raw observations, derived CSV, formula definitions and
  plot generator together. An omitted metric remains explicitly unavailable.

## Current SQLite preview and remaining measurements

The [three-application figure set](../../experiments/study/results/application-memory-paper-20260928/README.md)
now applies this layout to SQLite, mruby and FFmpeg, with a common four-arm
reuse figure, signed changes from each platform's control, three memory
companions and an eight-workload supplement. It reproduces all 96 existing
reuse-process records through their raw campaign validators and revalidates
the 12 committed SQLite memory transcripts. The review PDF includes captions;
vector figures and LaTeX snippets are separate. This is a checked reanalysis,
not a new application campaign. It preserves the SQLite metadata tradeoff,
mruby's post-render retention and FFmpeg's coincident reuse curves.

The [SQLite layout preview](../../experiments/study/results/sqlite-normalized-memory-20260927/paper-layout/README.md)
renders only the complete 12-run repeated-work experiment, at paper width.
It shows paired costs, absolute address coverage, and selected allocator
occupancy across complete workloads. It preserves the metadata tradeoff.
It is derived from the same checked logs as the earlier exploratory figures.

The [measured SQLite release-gap CDF](../../experiments/study/results/sqlite-reuse-gaps-20260927/README.md)
fills its memsys5 reuse slot, and the [FFmpeg decoder lease-gap
CDF](../../experiments/study/results/ffmpeg-reuse-gaps-20260927/README.md)
fills a second inner-boundary slot from complete application executions. The
FFmpeg study also counts explicit per-granule poison/clear and copy operations in the selective
PoisonCap adapter despite identical reuse bins in all four arms. The complete
mruby GC-slot and adapted FFmpeg FATE observations are included in the new
figure set. The remaining three applications still need complete four-arm
inner-boundary observations. Requested payload,
complete protection-metadata accounting, a qualified common burst, and a
sustainable node policy remain
before the other planned memory figures can make total-cost claims. Existing
address counts cannot be relabelled as a reuse CDF, full memory, fragmentation,
or working set. SQLite with lookaside disabled says nothing about its lookaside
allocator.
