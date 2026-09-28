# Application memory with the published PoisonCap thresholds

**Review starts here: [visual index](index.html).** The primary view groups
plots by metric; the secondary view groups the same data by application.

The [distribution view](reuse-distributions.pdf) places benchmarks on the
X-axis and release-to-reuse gaps on the logarithmic Y-axis. Each benchmark
has two split histogram violins: Sublet in teal and PoisonCap in violet,
each with its own spatial control as the sand-colored left half. Shapes are
conditional on observed reuses, with identical area and a common width
scale per log2 bucket; there is no KDE, interpolation or reconstructed event
sample. Reuse-frequency columns have been removed from this compact view;
the CSV retains reuses divided by **all** allocation issues. The distribution
alone does not show how frequent reuse is. The PoisonCap Xvid distribution
is undefined (zero reuses), rather
than a spike at zero or infinity. [CSV](reuse-distributions.csv) preserves
the bin counts, conditional mass and total allocation denominator. Three
repeats coincide; each distribution is displayed once. Workload horizons
differ, and the figure does not measure retirement-cohort recovery or memory
savings. This view uses the same validated raw runs as the CDFs.

For one shared metric across all five cases, start with the
[cross-application reuse comparison](cross-application-reuse.pdf)
([per-run values](cross-application-reuse.csv)). It shows control minus
protected reuse fractions at three existing CDF bucket edges, in percentage
points. These are exploratory display cuts with the existing all-issues
denominator, not a retirement-cohort measurement. Work horizons differ across
applications; the figure adds no new experiment or claim of memory savings.
The [common campaign presentation](../../../../docs/plans/application-memory-campaign.md#cross-application-presentation-and-paper-precedents)
specifies the memory-cost and recovery counterparts and their admission gates.

| View | Contents | Download |
|---|---|---|
| By metric (recommended) | Reuse; address footprint; selected allocator memory | [Three-page PDF](review-by-metric.pdf) |
| By application | SQLite; mruby; FFmpeg, with each application's available metrics | [Three-page PDF](review-by-application.pdf) |

Individual PDF/PNG sheets live under `by-metric/` and `by-application/`.
The index includes metric definitions and a coverage matrix: missing metrics
are explicit. Address interval union, carved pool extent and GC backing
remain distinct quantities. This reorganization uses the same validated
measurements; it adds no guest runs. Original mixed-layout figures below
remain available for existing links.

The figures validate **60 complete application processes: 54 fresh runs and
six archived Capstone SQLite controls**. mruby and FFmpeg contribute 48 fresh
processes (two workloads per application, four arms, three repetitions).
SQLite contributes six fresh PoisonCap processes with outer defaults on,
paired with its unchanged six Capstone controls. Every process passes the
complete application-output oracle. There is no trace replay; historical
outer-disabled PoisonCap curves are not used as new default measurements.

The comparison uses [the audited reference policy](../../poisoncap-policy.md):
4,096 quarantined entries, or at least 16 MiB held with at least one quarter
quarantined. The full-queue path includes the disclosed revocation correction.
Outer libc revocation remains enabled, asynchronous, with the published
8 MiB gate and every-free override disabled in both PoisonCap arms. There
is no sweep added to satisfy an ordinary application reuse request.

The authors provide a SQLite allocator port, not mruby or FFmpeg ports.
The latter therefore measure **published SQLite thresholds transferred to
these allocators**, with necessary correctness repairs and destructor
handling disclosed. They are not unchanged-artifact or author-provided ports.

## Original mixed-layout figures (retained for existing links)

[All three figures in one PDF](application-memory-published.pdf) ·
[LaTeX captions](figures.tex) · [Validation](validation.json)

- [Reuse CDFs](01-transferred-policy-reuse.pdf): common axes and denominator,
  all four arms for AO widths 8/16 and the two adapted FATE inputs. Lines
  show medians, bands the range across three processes. Repetitions coincide.
- [Selected memory ratios](02-transferred-policy-backing.pdf): peak GC backing,
  GC backing after rendering, and carved FFmpeg pool address extent. Every
  ratio divides a protected arm by its own platform's spatial control.
  Markers show median and range, with the 1× reference visible.
- [SQLite repeated work](03-sqlite-outer-defaults.pdf): four-arm reuse, the
  union of allocated address intervals, and selected allocator peak bytes
  through 17 complete speedtest1 main units. Grey marks the first unit.

| Workload | Sublet / Capstone spatial | PoisonCap temporal / spatial | Quantity |
|---|---:|---:|---|
| mruby AO 8 | 1.386× | 1.667× | Peak selected GC backing |
| mruby AO 16 | 1.386× | 1.833× | Peak selected GC backing |
| mruby AO 8 | 1.386× | 1.667× | Selected GC backing after render |
| mruby AO 16 | 2.078× | 1.833× | Selected GC backing after render |
| FFmpeg Xvid, 20 frames | 1.000× | 6.663× | Carved pool extent |
| FFmpeg resize, 150 frames | 1.000× | 12.840× | Carved pool extent |

Sublet preserves the FFmpeg control's observed lease reuse: 210/253 issues
reuse an address for Xvid and 1,724/1,825 for resize. PoisonCap reuses 0/253
and 510/1,825 respectively. These are observed application lease issues;
trusted destructor reissues are excluded consistently in all four arms.
The CDF includes first issues in its denominator and does not renormalize
to successful reuses. Its x axis counts intervening issues, not time or bytes.

For mruby AO 16, 75.54% of Sublet issues reuse a slot within 1,023 issues,
versus 0.332% for PoisonCap temporal. Peak selected GC backing is 750,720 B
for Sublet versus its 541,824 B control; PoisonCap uses 1,036,288 B versus
565,248 B. After rendering the Capstone control shrinks to 361,216 B, while
Sublet retains 750,720 B. The resulting 2.078× retention ratio is a
countercost, not a peak-memory improvement.

SQLite 3.22.0 performs 550,137 measured allocations per process. Sublet's
allocated interval union is 1,280,960 B from the first unit onward, equal to
its spatial control. PoisonCap temporal grows from 4,327,360 B after the
first unit to 5,113,792 B after the second, then stays flat through unit 17;
its spatial control stays at 1,283,008 B. Thus the steady address-coverage
ratio is 1.00× for Sublet and 3.986× for PoisonCap. This is cumulative
allocated-address coverage, not a physical working set or unbounded growth.
The selected allocator-peak ratio goes the other way: Sublet is 4.675×,
while PoisonCap ranges from 3.654× to 4.269×. Sublet's fixed allocator region
is included in that accounting. All six new PoisonCap inner phase ledgers
and reuse histograms exactly reproduce their historical outer-disabled
controls; the outer policy and retirement repairs were independently changed
and requalified.

## What the policy actually did

| Workload, PoisonCap temporal | Queue-full sweeps | Byte-threshold sweeps | Destructor sweeps |
|---|---:|---:|---:|
| mruby AO 8 | 52 | 0 | 0 |
| mruby AO 16 | 223 | 0 | 0 |
| FFmpeg Xvid | 0 | 0 | 17 |
| FFmpeg resize | 0 | 3 | 47 |

The FFmpeg port must restore persistent RefStruct contents before trusted
destructor callbacks. Those drains are counted separately. Snapshot backing
peaks at 1,378,944 B and 1,881,792 B for Xvid and resize. These bytes are
additional to the plotted carved extent, and are not silently counted as
intrinsic PoisonCap quarantine cost. Xvid never reaches the 16 MiB gate.
Resize does; its delayed reuse is consequently nonzero. Both end with some
quarantined spans inside the pool reservation; process exit reclaims it.

## Accounting and comparability

GC backing is peak/retained page count times the recorded per-page footprint.
It includes in-page observer storage and allocator metadata, plus CheriBSD
page rounding. FFmpeg extent is the high-water offset carved from its fixed
256 MiB reservation, including alignment padding. It is neither the full
reservation nor current live payload. Both FFmpeg platforms provision 8,192
records and the same payload capacity. The 256 MiB Capstone program region
is separate from its SDK's static 16 MiB malloc arena; the builder now checks
the application descriptor to prevent confusing those two settings.

These selected quantities exclude complete outer heaps, all adapter tables,
platform node storage and OS bookkeeping. They support claims about address
reuse, selected GC backing and pool expansion. They do not establish total
memory, physical working-set size, RSS, speed, or universal superiority.
Capstone uses the recorded node-reuse QEMU at 65,536 nodes; no timing claim
is made about that software collection mechanism.

Both FFmpeg builds use the same prepared 9.0.1 source and feature selection.
The library compiler flags end in `-O3`; drivers use `-O1`. Capstone uses
Clang 22 and PoisonCap Clang 17 with their required ABIs. mruby uses pinned
4.0.0-rc2; its GC translation units use `-O1`, but the Capstone interpreter
archive uses `-O2` while the PoisonCap interpreter uses `-O1`. This is a
remaining build-normalization limitation. The Linux and CheriBSD ABIs and
pointer sizes differ; within-platform ratios do not remove all adaptation
or compiler effects.

Both FFmpeg PoisonCap arms load the same staged libc containing the
output-constraint and complete poison-retirement fixes. The latter clears
retired payload before allocation, changing initialization work but not the
quarantine thresholds or requested memory. mruby retains its recorded libc
in both arms. Runtime and input identities are preserved in the build audit.
The first FFmpeg qualification shared a guest with independent correctness
diagnostics. The final archived comparison reruns both inputs and both arms
serially in a fresh guest, using the same binary and libc. No whole-guest
physical-memory or timing observations enter these figures. Three repetitions
check reproducibility, not coverage of the applications' input distributions.

## Evidence and reproduction

`summary.json` contains the 48 mruby/FFmpeg records, phase ledgers, exact bins,
absolute selected bytes, paired ratios, exclusions and raw-file hashes.
`sqlite-summary.json` and `sqlite-units.csv` preserve all 12 SQLite records
and their 17 end-of-unit ledgers. `policy-processes.csv`, `policy-reuse.csv`
and `policy-ratios.csv` expose the plotted values without requiring a figure
parser. [The SQLite build delta](sqlite-build-delta.json) verifies unchanged
workload, spatial source and compiler options, with only the named protected
retirement patch. SQLite uses the earlier matched `-O0` configuration and its
archived Capstone controls use 4,194,304 nodes, unlike the other applications'
65,536-node configuration.
The plotter rechecks raw application outputs, policy counters, effective
outer settings, loaded inputs and Capstone resource cleanup before drawing.
Failed qualification attempts remain separate evidence: the initial FFmpeg
SDK arena mistake, the libc ABI and retirement failures, and the initial
mruby AO16 timeout are not observations of zero reuse or memory exhaustion.

[Archive hashes and member hashes](archive.json) cover the raw records
(including excluded attempts) and the prepared build evidence. The bundles
are `/tmp/capstone/published-policy-20260928-raw.tar.gz` (437 files) and
`/tmp/capstone/published-policy-20260928-build-evidence-v2.tar.gz` (123 files).
Both archives were read back and every member hash verified. Private guest
SSH keys are excluded. [Build audit](build-audit.json),
[runtime identities](runtime-identities.json) and
[effective FFmpeg compiler settings](ffmpeg-configs.json) are also available
without extracting the bundles.

Run from the study worktree after restoring the evidence archives under
`/tmp/capstone`:

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-application-memory-paper.py \
  --out capstone/experiments/study/results/published-policy-20260928 \
  --policy-runs \
  /tmp/capstone/published-policy-20260928-capstone-runs \
  /tmp/capstone/published-policy-20260928-capstone-ffmpeg-runs \
  /tmp/capstone/mruby-published-policy-20260928-runs-v2 \
  /tmp/capstone/mruby-published-policy-20260928-ao16-runs \
  /tmp/capstone/ffmpeg-published-policy-20260928-final-runs \
  --sqlite-policy-runs /tmp/capstone/sqlite-published-policy-20260928-retired-runs
```
