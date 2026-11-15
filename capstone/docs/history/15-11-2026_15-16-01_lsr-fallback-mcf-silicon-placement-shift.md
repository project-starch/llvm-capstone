# LSR byte-IV fallback on silicon: mcf's +0.30 % came with a hot-code placement shift and no change to its hot loops (attribution pending)

The SPEC lane ran `-lsr-capability-byte-iv-fallback` (llvm-c13, arm FRCDKMSa3hy2) on silicon. It was NOT adopted:
spec-silicon Addendum 136, 0210dc7b23d4. The twin results that preceded it are in
`15-11-2026_09-15-01_lsr-byte-iv-fallback-twins.md`. This note records:
- why the mcf half missed;
- the directed test that measures the mechanism;
- the placement-controlled builds now available.

The attribution itself waits for the event-15 board pair (Addendum 137, spec-silicon 959f6ba3bb40).

## The reading

mcf matched pair, board 2, bitstream F, bkit75, specdiff 2/2 on both. Read from the two `board.txt` files:

| | control fa650b98b77a | c13 ad7832413d89 | delta |
|---|---:|---:|---:|
| wall time | 6,593.26 s | 6,612.07 s | +0.29 % |
| in-context run_cycles (end) | 160,036,368,452 | 160,521,883,087 | +0.30 % |
| mispredicts (hpm6) | 844.0 M | 801.8 M | −5 % |
| issue stalls (hpm7) | 94.43 G | 94.48 G | +0.05 % |
| scoreboard full (hpm8) | 333.2 M | 341.5 M | +2.5 % |

- The adoption rule required mcf not slower than +0.15 %. My prediction was −0.0..−0.5 %, from QEMU instruction counts.
- The board binaries are byte-identical to the twinned images.
- One same-image repeat exists in the survey: 3d7e62118636 on board 1 read 6,576.13 and 6,581.38 s (0.08 %).

## The fallback's own changes cannot make +485 M cycles

QEMU per-function user instructions, c13 minus base (`perf/lgep/placement/ufunc.py`, every window). Functions the
change does not touch agree to within ±40, e.g. spec_qsort −38 of 10.0 G. The −71.7 M total splits as:
- price_out_impl −75.9 M. One store loop of 38 M iterations goes from 8 to 6 instructions per iteration.
- primal_bea_mpp +16.9 M.
  - Its outer basket loop, 4.26 M iterations, gains an IV: the byte pass folded `perm[next]` into `perm[i]`'s byte IV
    with the loop-variant residual `next − i`.
  - It also spills two more capability callee-saved registers per call (+142 K ldc/stc).
- flow_cost −12.1 M.

primal_bea_mpp's 309 M-iteration arc loop and its 56 M basket update are instruction-for-instruction identical in both
arms. So is every other hot loop.

## What did change: where the hot code sits

Every function after the two changed ones moved by an odd number of half-words. The 2-mod-4 sites this created:
- cost_compare's entry: 0x22240 → 0x22222, 610 M calls;
- spec_qsort's return points after `c.jalr s0`;
- the arc loop head: 0x223ac → 0x223a6, a 32-bit `cincoffset`.

capstone_cv64a6 has SuperscalarEn = 0, so FETCH_WIDTH is 32. A fetch at address[1] = 1 whose instruction is 32-bit
emits nothing that cycle (`core/instr_realign.sv`, realign_bp_32, lines 103-110).

A whole-program estimate (`perf/lgep/placement/redirect.py`) counts taken-branch, call and return redirects into
32-bit instructions at 2 mod 4. Control flow is identical in both arms: 4,510,209,517 and 4,510,209,582 redirects.
- control: 653 M of them land at 2 mod 4;
- c13: 2,728 M.

At one bubble each that predicts +2.07 G cycles against +485 M measured. It is an upper bound, not a fit. The next
section shows why most bubbles hide: mcf is issue-stalled about 59 % of cycles (hpm7).

## Directed test: capstone-ariane `fetch-realign-loop-head` (branch fetch-realign-loop-head, d065cebd5)

- **Loop:** 3 instructions, all 32-bit, 512 iterations. The head is placed at each offset mod 16 under
  `.option norelax`, and every address was checked in the ELF.
- **Counters:** each arm reads mcycle and mhpmcounter3 programmed with event 15 (if_empty).
- **Setup:** seed 1.

**RETRACTION (2026-11-15 18:25).** The first table below was measured on the shared checkout's model at 062f4453b,
which is not the silicon's RTL. 062f4453b is not an ancestor of bitstream F (a948e4134) or G (9fd9d2530): 369 commits
and 44 core files separate them. instr_realign.sv is identical on every ref, but the cost is not set by that file
alone. Re-run on models built at F's and G's exact revisions:
- separate worktrees;
- Anvil regenerated and checked per run;
- verFiles listing only each worktree's RTL;
- provenance in perf/lgep/fgsim.

| head offset mod 16 | F a948e4134: cycles / if_empty per iteration | G 9fd9d2530 | 062f4453b (first reading, withdrawn for F and G) |
|---|---:|---:|---:|
| 0, 4, 8, 12 | 4.04..4.06 / 1.03..1.04 | 4.06..4.09 / 1.04..1.08 | 4.0 / 1.0 at 0, 8, 12; 5.0 / 2.0 at 4 |
| 2, 6, 10, 14 | 5.04..5.06 / 2.03..2.05 | 5.06..5.09 / 2.05..2.09 | 6.0 / 3.0 at 2; 5.0 / 2.0 at 6, 10, 14 |
| dependent DIV body, at 0 and at 2 | 5.61 / 0.02..0.03 | 5.62..5.63 / 0.03..0.04 | 5.61 / 0.0 |

On the silicon's RTL, a 32-bit head with bit 1 set costs exactly one fetch bubble per entry, and no other offset costs
anything. Withdrawn for F and G:
- the claim "the cost is a function of the offset, not only of bit 1";
- the 4/12 and 2-mod-16 values.

What stands:
- the 2-mod-4 mechanism;
- if_empty rising one for one with the penalty;
- the stalled DIV body hiding it;
- placement alone moving this loop by 25 %.

The redirect count's "one bubble per redirect into 2 mod 4" is exactly F's rule, so the +2.07 G bound above stands as
an upper bound. Addendum 137's bands are unaffected: they come from the measured +485 M, not from this test.

The first reading, on 062f4453b:
- The offset-0 arm, repeated last, reads 4.04.
- Event 15 rises one for one with the penalty.
- A stalled backend hides the penalty completely.

The first version of the test lost one arm's pad to linker relaxation: its 2-mod-4 arm came out at 0 mod 4. That
version was not pushed.

## Pending: the attribution (Addendum 137)

Board 3, F, firmware caplifive-opensbi hpm-events-ifempty 143fbd06 (mhpmevent3 = 15), kit bkit84, the same two
binaries. Pre-registered bands, in-context Δif_empty (c13 − control):
- **placement explains the cycles:** +340..+630 M;
- **refuted:** below +150 M;
- **partial:** in between.

## Placement-controlled builds: `-mllvm -align-all-functions=4` on both arms

Arms, with images built on all nine SPEC executables:
- `perf/cclever/arms/FRCDKMSa3f4` (llvm-c9);
- `FRCDKMSa3hy2f4` (llvm-c13 + fallback).

mcf:
- strip.dom 65f7a7395f3c3aff (f4b) and 62295e5e5b8a0a4d (f4h).
- All 285 functions sit at the same address mod 16 in both.
- Exactly 8 bodies differ, the functions the fallback changes: flow_cost, flow_org_cost, master, price_out_impl,
  primal_bea_mpp, primal_start_artificial, read_min and worker.
- QEMU twins against BIb: f4b +0 instructions exactly; f4h −71,697,768 (−0.269 %). Both have outputs identical to
  base, specdiff 2/2.
- 2-mod-4 redirects: 1,102 M and 1,061 M.

The predicted f4h − f4b is −0.00..−0.05 % in cycles, inside the ±0.15 % same-board spread. So an mcf-only f4 pair
decides nothing, and it is not run.

Image census, f4 against unaligned (`perf/lgep/placement/f4census.py`):
- **Method:** bodies are compared with immediates masked, because PC-relative low parts move with placement, and with
  trailing padding removed.
- **Control:** the c9-vs-c13 comparison lists exactly the fallback's changed functions.
- **Identical bodies:** mcf, x264, ldecod, nab, lbm and xz. Only placement changes there.
- **Real changes in perl, gcc and imagick:** `jal` → `auipc` + `jalr` at no fewer than 114, 269 and 184 call sites (counted from the most common edit shapes only). Some calls no
  longer relax to the short form after the text grows (perl +18 KB, gcc +92 KB, imagick +15 KB).
- So f4 is not placement-only for those three rows. They need twins before f4 could be a compiler setting.

perl full on the board-faithful stage (st-perl-a3 with each binary) is being twinned for both f4 arms
(`perf/lgep/chain-f4perl.sh`). perl is the row where the fallback's QEMU gain (−0.553 %) is large enough for a board
pair to decide anything.

## The lesson

A QEMU instruction twin cannot see placement. Any codegen change that shifts unchanged hot functions needs one of two
controls before its silicon delta is attributed to what it emits:
- a placement-controlled pair;
- event 15 read beside the cycles.
