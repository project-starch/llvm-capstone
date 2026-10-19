# allocator-boundary on vp2: the trimmed-Lib configuration, pinned, five repeats per image (QEMU prediction)

These are **readings of the trimmed-Lib configuration**: the only one the vp2 board can carry, since the full 3.13 Lib
does not fit the 26,000,000-byte DECLARED-bake cap. Each case's prediction is the **set** of readings five pinned QEMU
runs showed, because this corpus is not run-to-run deterministic (below). This is the runtime lane's ruling of
2026-10-19; full-Lib equivalence is a stated caveat, not a gate.

## Scope correction (2026-10-19, after a one-variable module swap): lt1 is NOT a board prediction

This record ran on qkit7's deployed module 5bfc38f808e1effd. That module predates the stopgap and COLLECTS on QEMU.
The board runs the five-item module 39d8492d5c18216b, whose QEMU twin is a5dec6a32c49b3d3: it never collects, and no
vp2 silicon can, since the RTL refuses CSRUNV action 3 (Q-27). One pinned trimmed run per image on qkit7-norand-rev2
(qkit7-norand with only the module swapped to a5dec6a3; the record's exact argv; the same guest pin reading) gave:

- **lt1 (virtual-nested-pools): does not transfer.** CAUGHT 4, NO-READING 27, with 23 stable cases outside this
  record's sets.
  - At the first case the module printed `capstone_vm: collect skipped: collection is not available on bf3db3175
    (Q-27, S-entry); retired 328 available 497`, then three more within 30 ms. Four is the line's budget.
  - From there nearly every case, control re-run and negative control faults with cause=30 (INSUF_RESOURCES) at
    `_PyObject_Malloc`'s CDERIVE. The nested pools mint a record per pymalloc pool and exhaust the flat table.
  - The table cannot be made larger: the flat record count is capped at 65,536 because the node index is 16 bits
    (`capstone_vm.c` CV_FLAT_RECORDS_MAX, a larger value refused at insmod).
  - lt1's cases are therefore recorded as "no reading on bf3db3175 / tree A: needs collection", the pattern of the
    contract's churn rows, and lt1 does not go to the board until collection exists. The lt1 tables below are
    predictions for a COLLECTING module only.
  - This is the first corpus arm seen to collect. The collections census earlier (zero collections) covered the 26
    pilot bundles, not this corpus.
- **lt0 (virtual-malloc): stands.** 31/32 inside the sets, and `collect skipped` 0 on the console.
  - The one case outside a stable set is 15: the same trigger fault (cause 24 at visit_decref+0), but its
    negative_control.py also faulted (cause 28 in `__capstone_mallocng_free`), so the verdict is NO-READING.
  - That is the allocator-internal variance shape the record already shows. Case 15 is now listed as varying in
    `prediction.json` ({CAUGHT, NO-READING}, `added_by` naming this run), and the board judges it by set
    membership like the other varying cases.

Bundles: `runs/modswap-lt0`, `runs/modswap-lt1`.

## Configuration

- Images: t3-new7 cpython-lt0 `0d322044b9118713` (arm virtual-malloc) and cpython-lt1 `1034e6d3715e5355`
  (arm virtual-nested-pools), built with CPY_TEST_CAPI=1.
- Stdlib: the closure from `tools/board/stdlib-closure.py` with whole dynamic packages (95b12f2e80a5): 445 files,
  329 static plus all of `encodings`. Before that fix, case 17's trigger raised `LookupError: unknown encoding: hz`;
  it now runs, with EXIT 0 on every arm.
- Pin: `--guest-env PYTHONHASHSEED=0` and `norandmaps` on the guest kernel command line (kit qkit7-norand, a
  qkit7 copy whose wrapper appends it). The guest's own reading under `guest` in `inputs.json` (kernel_cmdline
  `root=/dev/vda ro norandmaps`, hash_randomization 0, pythonhashseed "0", hash_abc 1453079729188098211) is
  identical in the 16 runs that carry it.
  - CORRECTED: this line used to say "identical in all 18 runs". lt0-full-1, lt0-full-2 and the seed-only pair ran
    the earlier runner, 727b6648108c9551 (40a05de0f942), which has no ABR-ENV reading. Their pin evidence is the
    session plan (PYTHONHASHSEED=0 on every guest command) and the domain code= base (one base under norandmaps).
- QEMU capstone-qemu b290f5656600 (svadu=false), module 5bfc38f808e1effd (the swap runs: a5dec6a32c49b3d3).
- Runner `runners/virtual/run-virtual.py` at e309263adc94 (sha256/16 555d747e744961a5), except the four runs named
  above (727b6648108c9551). Each ran through one appvm session: collect 54 calls, boot 54/54 steps, 32 verdicts.
- Exact invocation. The session wrapper `tests/rtl-smoke/vp2-kit-2026-10-17/tier3-smoke/tier3-appvm-arm.sh
  --virtual-kit <kit> <OUT> <run-virtual.py> ARGS` runs the runner twice with ARGS unchanged:
  `run-virtual.py <OUT>/out-collect --state <session>/state ARGS --raw <OUT>/raw-collect`, then the same with
  `<OUT>/out` and `--raw <OUT>/raw-read`. ARGS are:
  - lt0: `--arm virtual-malloc --build <t3-new7>/cpython-lt0 --llvm-bin <llvm-bin-v6> --guest-env PYTHONHASHSEED=0
    --stdlib-closure <closure2>/closure.txt`;
  - lt1: the same with `--arm virtual-nested-pools --build <t3-new7>/cpython-lt1`;
  - `--timeout 300` (explicit in the last round, the default elsewhere);
  - no --controls-build and no --only;
  - the full-Lib runs omit --stdlib-closure.

## The prediction (`prediction.json`: every case's set of signatures)

Signature: verdict, attribution, fault cause/offset/symbol, notes, infra. Absolute pcs are excluded.

| image | sha256/16 | arm | runs | stable | varying | varying cells (verdict sets) |
|---|---|---|---|---|---|---|
| lt0 | `0d322044b9118713` | virtual-malloc | 5 + 1 swap | 23 | 9 | 05 ['CAUGHT', 'NO-READING'], 07 ['CAUGHT', 'NO-READING'], 09 ['CAUGHT'], 15 ['CAUGHT', 'NO-READING'] (added by the swap run), 16 ['MISSED', 'NO-READING'], 19 ['CAUGHT', 'NO-READING'], 27 ['CAUGHT', 'MISSED'], 30 ['CAUGHT'], 31 ['CAUGHT', 'NO-READING'] |
| | | caveat vs full Lib (3 pinned runs) | | | | trimmed readings outside the full set in 8 cells: 05, 07, 09, 16, 19, 27, 30, 31 |
| lt1 (collecting module only; NOT for the board, see Scope correction) | `1034e6d3715e5355` | virtual-nested-pools | 5 | 28 | 4 | 03 ['CAUGHT', 'NO-READING'], 10 ['NO-READING'], 17 ['CAUGHT', 'MISSED'], 18 ['NO-READING'] |
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
source is open.

**icount diagnostic, 2026-10-19:** two runs, `runs/lt0-icount-1` and `-2`, used the pinned, trimmed lt0 configuration
under `-icount shift=4,sleep=off` (deterministic TCG time). Their verdicts.jsonl files are byte-identical, including pcs
and evidence (sha256/16 3d3ea18712f24b89). The pinned pair without icount (lt0-trim-4 and -5) differs. The residual
variance is therefore consistent with host timing; this rests on N=2.
- The icount arms ran at --timeout 900 and the others at 300. No step timed out in any arm.
- The 138 residual console lines (grant bases, unittest timings) reached no verdict field in these two runs. That is an
  observation about two runs, not a property of the scorer.
- The diagnostic changes no ruled measurement.

## The board stage for lt0 (2026-10-19): three levers to fit the firmware cap

The trimmed configuration above does not fit the 26,000,000-byte DECLARED-bake cap, so lt0's board stage takes three
levers.
1. **Image:** `llvm-objcopy --discard-locals` applied to `0d322044b9118713` gives `20bcfdd1073976e0`. It removes the `.L`
   labels only; the loaded bytes are identical.
2. **Stdlib:** a guest-traced closure with whole dynamic packages, `stage-closure.txt` (350 files, sha256/16
   b41545623090d0cf), in place of closure2's 445.
3. **Bytecode only:** `--stdlib-sourceless`, which compiles each staged module with `compileall -b`, stages the bytecode
   beside it and drops the `.py`. It is in `runners/virtual/run-virtual.py` from this commit (sha256/16
   b98e90fc86fafe1a).

**Size.** Measured by a measure-only board-kit collect of the same image and staging (kit bkit7-rev2xmD):
- the manifest holds 574 files, and the stage is 23,276,602 bytes;
- the firmware is 25,856,520 B (bd536b0aab6ee89d), a margin of 143,480 B under the cap.

The measured tree and the run below differ only in the source-mtime field of one generated `.pyc` header and in the
session marker, both the same size.

**One pinned run against this record:** `runs/lt0-stage-1`, using the module-swap run's kit and module
(a5dec6a32c49b3d3). Its platform equals `runs/modswap-lt0` in every field but the runner. The guest pin reads back as
above. Judged against `prediction.json` lt0, with the evidence text in the signature:
- all 23 stable cells are inside their sets;
- of the 9 varying cells, 6 are inside;
- 3 are outside the sampled set: 05 (CAUGHT at `_PyDict_FromItems+144`, cause 28), 09 (CAUGHT at
  `unicodekeys_lookup_unicode+400`, cause 28) and 31 (CAUGHT at `mro_implementation_unlocked+316`, cause 24).

Each of those three has a verdict inside its set at a site not seen in the sampled runs. That is the allocator-internal
variance this record already shows. A board reading of the stage is judged against the lt0 sets, and a varying cell
outside them is reported as "outside the sampled set". Tally: CAUGHT 15, MISSED 14, NO-READING 3 (infra 1, not-reached
1, unattributed 1).

## Caveat: trimmed vs full Lib

The `caveat` rows count cases where some trimmed reading lies outside the three pinned full-Lib runs' set. With
3 full runs those sets are under-sampled too, so these are differences not explained by the variance observed, not
established trim effects. A smaller Lib does change the heap's history, which is what this corpus measures.

## Files

`runs/<name>/{verdicts.jsonl, inputs.json, SOURCE}`: the 18 bundles named above, plus the board stage's run
(`lt0-stage-1`) and the icount pair (`lt0-icount-1`, `-2`). SOURCE names the run directory each came from.
`stage-closure.txt` is the board stage's stdlib closure.
