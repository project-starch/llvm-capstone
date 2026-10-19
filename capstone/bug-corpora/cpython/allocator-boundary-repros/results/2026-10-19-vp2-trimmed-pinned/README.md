# allocator-boundary on vp2: the trimmed-Lib configuration, pinned, five repeats per image (QEMU prediction)

These are **readings of the trimmed-Lib configuration**: the only one the vp2 board can carry, since the full 3.13 Lib
does not fit the 26,000,000-byte DECLARED-bake cap. Each case's prediction is the **set** of readings five pinned QEMU
runs showed, because this corpus is not run-to-run deterministic (below). This is the runtime lane's ruling of
2026-10-19; full-Lib equivalence is a stated caveat, not a gate.

## Configuration

- Images: t3-new7 cpython-lt0 `0d322044b9118713` (arm virtual-malloc) and cpython-lt1 `1034e6d3715e5355`
  (arm virtual-nested-pools), built with CPY_TEST_CAPI=1.
- Stdlib: the closure from `tools/board/stdlib-closure.py` with whole dynamic packages (95b12f2e80a5): 445 files,
  329 static plus all of `encodings`. Before that fix, case 17's trigger raised `LookupError: unknown encoding: hz`;
  it now runs, with EXIT 0 on every arm.
- Pin: `--guest-env PYTHONHASHSEED=0` and `norandmaps` on the guest kernel command line (kit qkit7-norand, a
  qkit7 copy whose wrapper appends it). Every run's `inputs.json` carries the guest's own reading under `guest`:
  kernel_cmdline `root=/dev/vda ro norandmaps`, hash_randomization 0, pythonhashseed "0", hash_abc
  1453079729188098211, identical in all 18 runs here.
- QEMU capstone-qemu b290f5656600 (svadu=false), module 5bfc38f808e1effd. Runner
  `runners/virtual/run-virtual.py` at e309263adc94, through one appvm session per run: collect 54 calls, boot 54/54
  steps, 32 verdicts each.

## The prediction (`prediction.json`: every case's set of signatures)

Signature: verdict, attribution, fault cause/offset/symbol, notes, infra. Absolute pcs are excluded.

| image | sha256/16 | arm | runs | stable | varying | varying cells (verdict sets) |
|---|---|---|---|---|---|---|
| lt0 | `0d322044b9118713` | virtual-malloc | 5 | 24 | 8 | 05 ['CAUGHT', 'NO-READING'], 07 ['CAUGHT', 'NO-READING'], 09 ['CAUGHT'], 16 ['MISSED', 'NO-READING'], 19 ['CAUGHT', 'NO-READING'], 27 ['CAUGHT', 'MISSED'], 30 ['CAUGHT'], 31 ['CAUGHT', 'NO-READING'] |
| | | caveat vs full Lib (3 pinned runs) | | | | trimmed readings outside the full set in 8 cells: 05, 07, 09, 16, 19, 27, 30, 31 |
| lt1 | `1034e6d3715e5355` | virtual-nested-pools | 5 | 28 | 4 | 03 ['CAUGHT', 'NO-READING'], 10 ['NO-READING'], 17 ['CAUGHT', 'MISSED'], 18 ['NO-READING'] |
| | | caveat vs full Lib (3 pinned runs) | | | | trimmed readings outside the full set in 6 cells: 03, 10, 17, 18, 28, 30 |

"Stable" means **not seen to vary in five runs**, not deterministic. The varying count grew with every pair of runs
(lt0 trimmed: 0 at n=1, 5 at n=3, 8 at n=5), so the sets are under-sampled.

**How a silicon run is judged** (the ruling): it must carry the SAME pin, with the guest's `guest` fields equal.
- A reading inside a case's set is agreement.
- Outside the set on a varying case is reported as "outside the sampled set".
- Outside on a stable case is a divergence.

## How the pin was found, including a retraction

| comparison (lt0 full Lib) | cells differing |
|---|---|
| two unpinned runs | 3 / 32 |
| two runs with PYTHONHASHSEED=0 only (runs/lt0-full-seedonly-*) | 7 / 32 |
| two runs with seed + norandmaps (runs/lt0-full-1, -2) | 0 / 32 |
| a third pinned run (lt0-full-3), and the pinned lt1 full pair | 2 / 32 each |

The seed alone does not pin the corpus. `norandmaps` took effect: every process's code base is equal under it and
different without it. The 0/32 at two runs was reported as "the pin", and was **retracted** the same day when the
third pinned run and the lt1 pair differed. The pin narrows the variance; it does not remove it. Most varying cells
swing between the defect's own site and a fault inside the allocator (`__capstone_mallocng_free+180`,
`memset+44`), or between a fault and a clean run: heap-state-dependent placement, not established. The remaining
source is open. An `-icount` (deterministic TCG time) pair is the next diagnostic and is not part of this record.

## Caveat: trimmed vs full Lib

The `caveat` rows count cases where some trimmed reading lies outside the three pinned full-Lib runs' set. With
3 full runs those sets are under-sampled too, so these are differences not explained by the variance observed, not
established trim effects. A smaller Lib does change the heap's history, which is what this corpus measures.

## Files

`runs/<name>/{verdicts.jsonl, inputs.json, SOURCE}`: the 18 bundles named above. SOURCE names the run directory
each came from.
