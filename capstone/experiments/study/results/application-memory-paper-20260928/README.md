# Memory behavior in three complete applications

**Policy audit, 2026-09-28:** these are comparisons of different adapter
policies, not a common published-default PoisonCap comparison. SQLite uses
published thresholds with the full-queue correction; mruby sweeps on free-slot
exhaustion; FFmpeg sweeps before reissue and disables outer libc revocation.
SQLite's normalized memory/reuse campaigns also disable outer libc revocation.
The [reference-policy audit](../../poisoncap-policy.md) pins the published
sources and defines the replacement measurements. The revised figure footers,
captions and provenance make this distinction explicit. Data are unchanged;
no new default-policy application measurements are represented.

The [new published-threshold results](../published-policy-20260928/README.md)
are a separate campaign; use that directory for the replacement comparisons.

[Review PDF with captions](application-memory-figures.pdf) ·
[LaTeX figure snippets](figures.tex) ·
[Raw-data validation](validation.json) · [Provenance](provenance.json)

These five figures reanalyse existing complete application executions. There
are **96 reuse processes** (eight workloads × four arms × three repetitions)
and **12 separate SQLite memory processes**. No new application execution or
allocator trace replay is represented. The original campaign validators were
rerun: every raw-derived reuse record exactly matches the committed input,
including application-output hashes, counters and histograms. SQLite's
committed memory transcripts also pass their complete SQL-output oracles and
ledger checks. Four external raw archives match their recorded SHA-256 hashes.

The layout follows the paper's compact panels and serif typography, with one
legend, a fixed platform palette, redundant markers, and shared axes. Each
standalone PDF is vector graphics at **7.05 inches**, with embedded TrueType
fonts. The review PDF adds captions; the PNGs are inspection copies. The
manuscript itself is unchanged.

## Figures and supported conclusions

1. [Reuse and change from each control](01-reuse-and-control.pdf).
   The top row shows all four arms. Staggered marker positions distinguish
   coincident curves without moving observations. The bottom row subtracts
   each protected curve from its own spatial control. This reveals delayed
   reuse without choosing a favorable single gap threshold. SQLite's Sublet
   curve is unchanged; mruby's differs slightly; all four FFmpeg curves
   coincide. SQLite uses the available matched `main` workload, mruby the
   larger measured AO size, and FFmpeg the longer adapted FATE input. This
   selection is exploratory, not preregistered.
2. [SQLite memory over repeated useful work](02-sqlite-memory.pdf).
   Sublet preserves address coverage at **1.00×** its original; corrected
   PoisonCap reaches **3.99×** its own original. Both plateau within the
   observed 17 units. The companion panel preserves the countercost:
   Sublet's peak selected allocator bytes are **4.68×**, PoisonCap's **4.27×**.
   Table storage explains why the address result is not a total-memory win.
3. [mruby GC capacity and retained storage](03-mruby-memory.pdf).
   Over **4.22×** more slot issues, peak GC groups stay at **6** for Sublet
   and its control, and **9 versus 6** for PoisonCap temporal versus spatial.
   This is two observed work sizes, not an asymptotic scaling result. The
   second panel includes metadata and post-render retention: at AO 16,
   Sublet's selected bytes after rendering reach **2.08×** its control,
   versus **1.50×** for PoisonCap. The controls have different layout scope;
   these ratios cannot rank equal-scope full adaptation costs.
4. [FFmpeg snapshot backing](04-ffmpeg-snapshots.pdf).
   The selective PoisonCap adapter peaks at **224.4 / 167.1 KiB** on the two
   adapted FATE inputs. The repeated small input remains at **35.4 KiB**
   across 1, 4 and 16 streams; final snapshot backing is zero. Thus prompt
   reuse coexists with transient snapshot storage in this adapter. The data
   show neither a reuse advantage for Sublet nor history-proportional growth
   of PoisonCap's snapshot peak. No total-memory ranking follows.
5. [All eight workloads](05-all-workloads-reuse.pdf).
   The supplement retains the smaller AO input, both adapted FATE inputs,
   and all three repeated-stream sizes, under the same axes and definitions.

The maximum positive difference at any recorded histogram edge is:

| Workload | Sublet vs. its spatial control (pp) | PoisonCap vs. its spatial control (pp) |
|---|---:|---:|
| SQLite `speedtest1 main`, size 1, 17 units | 0.00 | 88.16 |
| mruby AO, width 8 | 0.42 | 87.42 |
| mruby AO, width 16 | 0.70 | 88.53 |
| FFmpeg adapted FATE, Xvid | 0.00 | 0.00 |
| FFmpeg adapted FATE, resolution change | 0.00 | 0.00 |
| FFmpeg repeated stream, 1 / 4 / 16 streams | 0.00 in each | 0.00 in each |

These are **percentage points of all allocation issues**, not percentages
of reused objects or estimates of additional memory. They summarize the
whole recorded CDF. The signed curves can cross: PoisonCap's mruby AO 16
curve briefly exceeds its control by 3.11 pp at a later gap. The full curve
and signed minimum remain visible; the positive maximum is not a claim of
pointwise dominance. No statistical test or confidence interval is inferred
from deterministic process repetitions.

## Definitions and absolute denominators

For histogram bin `b`, the inclusive upper edge is `2^(b+1)-1` issues.
`F(g) = 100 × reissues with release gap ≤ g / all successful issues`.
Immediate reuse has gap **one**, including the reissue event. The observer
tracks the same allocator start (arena/slot identity), not across-process
virtual addresses or every overlapping byte. Step curves show only cumulative
bin-edge information; the distribution within a bin is unknown. SQLite's one
setup allocation is disclosed in its source campaign and excluded from the
measured-unit issue denominator. mruby and FFmpeg include process startup.

The lower panels report `D(g) = F_spatial(g) − F_protected(g)`; the table
reports `max(0, max_g D(g))`. The maxima are descriptive over all 32 stored
bins, not a significance test or a preselected threshold. All repetitions
coincide in the plotted quantities. The plotter rejects missing/duplicate
processes, divergent repetitions, unreconciled counts and an occupied
saturating histogram bin instead of silently losing variation or censoring.

SQLite ratios use original-layout controls on both platforms and match units
within a repetition. The spatial peak denominators remain constant over the
16 measured units. Therefore the maximum plotted per-unit ratio also equals
the ratio of run peaks in this dataset. Selected bytes are live plus
quarantined spans plus allocator tables at the event peak; table storage is
constant per process, so no unrelated component peaks are added.

| SQLite arm | Final allocated interval union (B) | Peak selected bytes (B) |
|---|---:|---:|
| Capstone original | 1,280,960 | 1,404,831 |
| Sublet | 1,280,960 | 6,567,768 |
| CheriBSD original | 1,283,008 | 1,406,879 |
| Corrected PoisonCap | 5,113,792 | 6,005,919 |

mruby's selected bytes count GC group storage, including the 8,320-byte
observer in each group. Capstone counts payload plus page structure;
PoisonCap counts mapped group length including page rounding. This preserves
the source campaign's definition; it is not an instrumentation-free total.

| mruby arm | Peak bytes, both sizes | After AO 8 (B) | After AO 16 (B) |
|---|---:|---:|---:|
| Capstone spatial | 541,824 | 541,824 | 361,216 |
| Sublet | 750,720 | 750,720 | 750,720 |
| PoisonCap spatial | 565,248 | 565,248 | 565,248 |
| PoisonCap temporal | 847,872 | 847,872 | 847,872 |

The FFmpeg snapshot denominator is zero in spatial mode. Its figure therefore
uses **bytes**, not a ratio with an invented denominator. Snapshot backing is
one adapter component; neither unmeasured Sublet storage nor revoker storage
is entered as zero. Snapshot copy/poison operation spans are not plotted as
memory occupancy or DRAM traffic.

## Comparability and limits

These are paired, within-platform measurements of the stated implementations
and policies. The ratios do not remove ABI, allocator-layout, compiler or
kernel differences. The campaign sources retain exact pins and methods:

| Campaign | Build / control qualification | Source record |
|---|---|---|
| SQLite 3.22.0 | Matched `-O0`, same 129,055 × 64-byte atom capacity; original-layout denominators; corrected PoisonCap quarantine | [Memory](../sqlite-normalized-memory-20260927/README.md), [reuse](../sqlite-reuse-gaps-20260927/README.md) |
| mruby 4.0.0-rc2 | GC units `-O1`; existing Capstone interpreter archive `-O2`, CheriBSD interpreter `-O1`; PoisonCap spatial retains adapter layout | [Both AO sizes](../mruby-gc-memory-20260927/README.md) |
| FFmpeg 9.0.1 | Library effective `-O3` on both platforms, application driver `-O1`; PoisonCap spatial retains adapter layout | [Repeated streams](../ffmpeg-reuse-gaps-20260927/README.md), [adapted FATE](../ffmpeg-fate-four-arm-20260928/README.md) |

Both FATE CheriBSD arms use the same pinned L2-superpage kernel fix. SQLite
requires the documented enlarged emulator node capacity. Different workload
families are not pooled into a memory-overhead average. Scaled AO and adapted
FATE remain labeled as such; these are not default-size AO or official FATE
scores. The results are not a final uniformly optimized release campaign.

The available evidence supports preserved prompt reuse, selected backing
requirements, and their observed behavior as useful work increases. Complete
Capstone node storage, PoisonCap shadow/revocation storage, comparable physical
pages, and requested/live byte ledgers remain absent. Therefore these figures
do not establish total memory, physical working set, fragmentation, cache
locality, elapsed-time performance, or a universal PoisonCap lower bound.

## Reproduce and inspect

The committed inputs suffice for a figure-only redraw. `--verify-raw` also
requires the archived campaign trees and pinned binaries under `/tmp/capstone`
as described in the linked source records; it reuses their original validators.
The FATE archive is restored into an isolated temporary directory. No VM is
booted and the original results are not rewritten.

```sh
source capstone/tests/capstone-test-env.sh
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-application-memory-paper.py \
  --verify-raw \
  --out capstone/experiments/study/results/application-memory-paper-20260928
/tmp/capstone/application-memory/venv/bin/python -m unittest discover \
  -s capstone/experiments/study -p test_application_memory_paper.py
```

Remove `--verify-raw` for redraw from the checked summaries plus committed
SQLite memory logs. The generator exports [integer bin counts](reuse-bins.csv),
[issue/reissue denominators](reuse-counts.csv), [paired effects](paired-reuse.csv),
and absolute [SQLite](sqlite-memory.csv), [mruby](mruby-memory.csv) and
[FFmpeg](ffmpeg-memory.csv) memory records. Input and plotter hashes accompany
the artifacts. Five guard tests cover denominator semantics, incomplete or
duplicate matrices, invalid/overflow counts, changed demand and hidden run
variation. All five PDFs were visually inspected; embedded fonts and fixed
output widths were checked. The LaTeX snippets are supplied for later
manuscript integration, not compiled manuscript changes.
