# mruby 4.0.0-rc2: GC-slot reuse at two AO work sizes

This campaign runs the complete mruby interpreter on the upstream
`benchmark/bm_ao_render.rb` workload at widths 8 and 16 (the upstream default
is 64). The same interpreter binaries, AO source and GC observer serve both
sizes; only the width argument changes.
Only two stderr phase markers are inserted; rendering and the binary PPM
output are unchanged. An independent native interpreter produces the same
781-byte PPM with SHA-256
`cfd73bb1de97514b2adacbda3a377e5172aecbe48819f654071d1996359ca634`.
At width 8, the independent output is 203 bytes with SHA-256
`28060a9790593ace825b7552ab5d1878cf3e95be3be08ee738f28cde4cee5d11`.
The four arms are Capstone spatial GC, Capstone per-slot Sublet GC, PoisonCap
spatial GC and PoisonCap temporal GC. At each size, both Capstone arms use one
persistent Linux guest and both PoisonCap arms use one persistent CheriBSD guest. Each arm
has three separate processes per size, paired and interleaved by repetition.

All **24/24 process attempts pass** their exact independent output oracle.
Every repetition within an arm has the same 32-bin histogram and page-group
counts. They check reproducibility of this deterministic workload; GC events
inside one process are not independent statistical replicates, so no
confidence interval is inferred. Width 16 issues 915,981 GC slots per
process; width 8 issues 217,070. The protected Sublet and
PoisonCap spatial controls coincide in **every** gap bin; the three nearly
overlapping control curves in each CDF are intentional.

Width 16:

| Arm | Reissues by gap ≤15 | Reissues by gap ≤1,023 | Peak groups | Groups after render | Selected bytes per GC group |
|---|---:|---:|---:|---:|---:|
| Capstone spatial | 11,145 | 695,026 | 6 | 4 | 90,304 |
| Capstone + Sublet | 11,216 | 691,934 | 6 | 6 | 125,120 |
| PoisonCap spatial | 11,216 | 691,934 | 6 | 6 | 94,208 mapped |
| PoisonCap temporal | 0 | 17,896 | 9 | 9 | 94,208 mapped |

Thus 75.54% of Sublet's issues reissue a released slot within 1,023 later
issues, versus 1.95% for this PoisonCap temporal policy. Its peak GC group
count rises from 6 to 9, while Sublet's stays at 6. This is a strong
**logical-reuse** difference in the complete application. PoisonCap still has
906,765 same-slot reissues, versus Sublet's 909,946; most occur after longer
gaps. Total memory remains unresolved: Sublet's metadata grows from 8,384 to 43,200
bytes per group, and the port retains all-dead pages. At width 16 render completion,
Sublet retains 6 groups versus its own spatial control's 4; PoisonCap
temporal retains 9 versus its spatial control's 6. The selected peak
page-structure/payload bytes rise by 38.6% for Sublet and 50.0% for this
PoisonCap adapter; the post-render selected bytes rise by 107.8% and 50.0%
respectively. Capstone node memory and PoisonCap shadow/revocation storage
are absent from those selected figures.

The [two-size figure](gc-size-effect.pdf) shows the same pattern at width 8.
Its [eight-cell CSV](scale.csv) gives the exact values behind both panels.

| Arm at width 8 | Reissues by gap ≤15 | Reissues by gap ≤1,023 | Peak groups | Groups after render |
|---|---:|---:|---:|---:|
| Capstone spatial | 2,730 | 164,546 | 6 | 6 |
| Capstone + Sublet | 2,724 | 163,634 | 6 | 6 |
| PoisonCap spatial | 2,724 | 163,634 | 6 | 6 |
| PoisonCap temporal | 0 | 2,854 | 9 | 9 |

Sublet's prompt-reuse fraction is 75.38% at width 8 versus 1.31% for
PoisonCap temporal; at width 16 the corresponding fractions are 75.54% and
1.95%. Peak GC group counts stay at 6/6/6/9 across the four arms at both
sizes despite 4.22× more slot issues at width 16. This is a plateau of
selected GC page groups over these two work units, not a physical working-set
measurement. The two sizes have different post-render GC states. The
[width-8 CDF](ao8/gc-reuse-cdf.pdf) and [width-8 page figure](ao8/gc-page-groups.pdf)
preserve that smaller work unit's full bins and phases.

The observed boundary is the GC's 1,024-slot heap page. `issues` counts every
slot handed to an object; `releases` counts slots made dead by GC sweep.
For a later issue of the same slot, the release gap is the number of
intervening issue events. The 32 log₂ bins are retained as aggregate integer
counts; no freed-object capabilities or trace replay are retained. The
[reuse CDF](gc-reuse-cdf.pdf) divides each cumulative bin by **all issues**,
so slots without a same-slot reissue do not silently disappear from the
denominator. The [page-group figure](gc-page-groups.pdf) reports the maximum
and post-render number of GC heap pages, not resident OS pages or total RSS.

Both PoisonCap modes use one binary and identical mmap-backed GC page layout.
The temporal mode poisons a dead RVALUE after its destructor, quarantines its
slot outside payload, and synchronously revokes then clears and zeroes the
batch before reissuing any quarantined slot. The spatial control omits those
operations. Automatic outer libc revocation is enabled in **both** modes;
the named contrast is the nested GC policy. The adapter is an explicit mruby
implementation, not a published mruby PoisonCap port or an optimized lower
bound for every possible PoisonCap GC policy.
At width 16 the temporal adapter makes 169 explicit reclaim passes,
targets 73,140,480 bytes with `cpoison`, and clears and zeroes 72,870,160
bytes each before reissue. The remaining 3,379 poisoned slots are discarded
when private GC pages are unmapped at process exit; the quarantine count then
returns to zero. These are operation spans and policy events, not DRAM traffic
or CPU time. Sublet's 913,646 per-slot revokes are a different unit and are
not equated with PoisonCap reclaim passes.
At width 8 it makes 42 passes and poisons 17,222,160 bytes of slot spans;
the same accounting checks apply.

The [Capstone Sublet port](../../../../ports/mruby/musl/README.md) grants each
RVALUE slot beneath its GC page and revokes individual dead slots. Its
all-dead page retention and refusal of embedder-provided GC regions are
documented in that port. The gap observer adds 8,320 bytes of page-local
integer history to both Capstone modes. PoisonCap's observer additionally
needs a 128-byte quarantine bitmap per page. The reported GC page groups
exclude Capstone node storage, Sublet/runtime allocator rounding, PoisonCap
shadow and revocation storage, and OS allocator caches. The PoisonCap process
jemalloc ledger does not include these mmap GC pages. Neither figure is a
total-memory or physical working-set ranking.

The two GC translation units are built at `-O1` from the same pinned
4.0.0-rc2 source plus their stated adapters. The remainder of the existing
Capstone interpreter archive was built by its `-O2` port recipe; the
CheriBSD/PoisonCap interpreter uses `-O1`. ABI, libc, compiler, runtime and
outer allocator also differ across platforms. Interpret the within-platform
protected/spatial increments; the four absolute page-layout sizes are not
interchangeable. This is an application memory-behavior comparison, with no
time, security, bandwidth or hardware-cost claim.

`summary.json` and `ao8/summary.json` preserve every process's phase counters,
gap bins, output hash, and image hash. Their `bins.csv` files contain 384 bin
observations each. The [plotter](../../plot-mruby-gc-memory.py) checks all
24 process records across the two independent campaigns,
source and output hashes, policy modes, issue equality, histogram totals,
guest continuity and Capstone cleanup before drawing the PDFs. Raw guest
transcripts, build commands, images and source pins are kept in the external
archive identified by `archive.json`; guest private keys are excluded.
The [scale plotter](../../plot-mruby-gc-size-effect.py) also rejects a changed
AO source or interpreter binary between sizes.

The committed [matrix](matrix.json) and [four-arm binding](bindings.json)
admit both sizes through the generic `study.py` planner for subsequent runs.
With the raw archive restored and the named platform binaries present, the
planner emits 12 qualified points per platform and no unavailable cell. All
24 emitted points were checked against the existing runner verdicts on the
measured transcripts. Rebinding those transcripts to the planned point IDs
also reproduces both checked summary JSON files byte for byte through the
same plotter. This binding was assembled after the direct-runner
campaign; it does not turn these already collected runs into a pre-registered
campaign.

After restoring that archive under `/tmp/capstone`, redraw with:

```sh
source capstone/tests/capstone-test-env.sh
tar -C /tmp/capstone -xzf /tmp/capstone/mruby-gc-memory-20260927.tar.gz
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-mruby-gc-memory.py \
  --capstone /tmp/capstone/mruby-gc-v3-ao-run \
  --poisoncap /tmp/capstone/mruby-poisoncap-ao-campaign \
  --out capstone/experiments/study/results/mruby-gc-memory-20260927
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-mruby-gc-memory.py --width 8 \
  --capstone /tmp/capstone/mruby-gc-ao8-capstone-run \
  --poisoncap /tmp/capstone/mruby-gc-ao8-poisoncap-run \
  --out capstone/experiments/study/results/mruby-gc-memory-20260927/ao8
/tmp/capstone/application-memory/venv/bin/python \
  capstone/experiments/study/plot-mruby-gc-size-effect.py \
  --width8 capstone/experiments/study/results/mruby-gc-memory-20260927/ao8/summary.json \
  --width16 capstone/experiments/study/results/mruby-gc-memory-20260927/summary.json \
  --out capstone/experiments/study/results/mruby-gc-memory-20260927
```

For figure-only redraw without raw VM outputs, use `--summary` with the
checked `summary.json`.
