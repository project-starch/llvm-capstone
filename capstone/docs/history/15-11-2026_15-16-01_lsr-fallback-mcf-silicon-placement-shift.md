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

## perl full: function alignment does not control the loop that carries the gain

Twins on the board-faithful stage (st-perl-a3 with each binary, `perf/lgep/chain-f4perl.sh`), shift 1, both
specdiff 2/2 with result files identical to the unaligned base's:
- strip.dom 82392ea004accec4 (f4b) and 03ccab6a160b0d7a (f4h);
- f4b against unaligned FRCDKMSa3: +548,235 user (+0.001 %), so the `jal` → `auipc` + `jalr` edits cost nothing that
  matters;
- f4h − f4b: −483,917,429 (−0.553 %), within 37 K of the unaligned pair's −483,880,297.

Placement check (`f4census.py`): in the f4 pair none of perl's 2,903 functions moves mod 16, and 99 bodies differ.
Without alignment, c13 moves 1,364 of them mod 4 and 2,722 mod 16.

Per-function split (`perf/lgep/placement/ufunc2.py`):
- **Why a new tool.** ufunc.py refused perl: one load delta places only 96.3 % of user instructions. The rest is
  - the stage's `natperl`, a static ELF at its link address that drives test.pl, 3.04 G, with identical per-PC counts in
    both arms;
  - shared-library code at randomized addresses.
  ufunc2 reports what the delta leaves outside .text as `?`, with its own A−B, which is exactly what it adds to the
  total. It errors if that exceeds 1 % of |A−B|.
- **Controls.**
  - Here the remainder reads −14,368.
  - Negative control: mcf's ELF on the perl profile leaves +50.9 G outside, and the tool errors.
  - mcf's f4 pair reproduces ufunc.py's split exactly.
- **Result.** S_regcppop accounts for −425,065,427 and Perl_yyparse for −66,223,443. Everything else nets +7.4 M.

S_regcppop, the paren-invalidation loop, 183.9 M iterations per run:

| | instructions per iteration | side block (110.6 M of them) | loop head (both entries are redirects) |
|---|---:|---:|---|
| f4b | 8 | 4 | 0xf8378, 8 mod 16, 32-bit `sub` |
| f4h | 6 | 3 | 0xf842a, 10 mod 16, 32-bit `cincoffset` |
| unaligned base c1744e87 | 8 | 4 | 0xf61ea, 10 mod 16 |
| unaligned c13 8cf05081 | 6 | 3 | 0xf627c, 12 mod 16 |

- **Instructions.** 2 × 183.9 M + 110.6 M = −478.4 M in the loop. The setup gains 4 instructions × 13.34 M, +53.4 M.
  Net −425.0 M.
- **Placement.** `-align-all-functions=4` fixes every function's entry. It cannot fix a loop head inside a function
  whose body the change rewrites. In the f4 pair the 2-mod-4 head is c13's; in the unaligned board images it is the
  base's.

Directed test capstone-ariane `regcppop-loop-head` (branch fetch-realign-loop-head, c83f7f98b):
- **What it runs.** Both loop bodies, every instruction's size kept from the f4 disassembly, the head at each even offset
  mod 16, perl's 60 % side-path split, entry through the `j` target as in the binary.
- **Substitution.** `add` stands in for `cincoffset`, because bare M-mode has no capability to offset.
- **Checks.** RVFI shows every head retiring 512 times and the side block 306 of 511 times. The B8 and H10 layouts match
  f4b and f4h one for one.
- **Setup.** Seed 1, models at F's and G's exact RTL (see the retraction above). The first run, on 062f4453b, is kept
  as a superseded row.

Cycles per iteration (warm-up and the drift repeat of B8 last: F 20.30 cold, then 13.68; G 20.02, then 13.73):

| head mod 16 | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F base body | 13.66 | 13.67 | 13.68 | 13.67 | 13.67 | 13.67 | 13.67 | 13.68 |
| F c13 body | 7.88 | 8.30 | 7.87 | 8.28 | 7.88 | 8.29 | 7.89 | 8.29 |
| G base body | 13.69 | 13.71 | 13.73 | 13.71 | 13.71 | 13.71 | 13.71 | 13.73 |
| G c13 body | 8.41 | 8.82 | 8.40 | 8.81 | 8.41 | 8.83 | 8.43 | 8.83 |
| 062f4453b base (superseded) | 13.66 | 13.67 | 13.70 | 13.67 | 13.67 | 13.68 | 13.67 | 13.69 |
| 062f4453b c13 (superseded) | 9.56 | 9.97 | 9.53 | 9.95 | 9.54 | 10.35 | 9.95 | 9.95 |

- **Base.** Bound in the backend at every offset on every model, so its head placement costs nothing.
- **c13.** It removes the dependent `sub` → `add` → `c.sd` address chain, worth far more than its two instructions. On F
  its head costs 0.41 cycles per iteration at bit 1 set.
- **The pairs.** On F, the f4 pair (B8 → H10) reads −5.38 cycles per iteration and the unaligned pair (B10 → H12)
  −5.79. On G, −4.88 and −5.29. On 062f the readings were −3.32 and −3.73; the 3-instruction model had predicted −1.6
  and −3.6.

Predicted in-context Δcycles, c13 − base, per full perl run:
- **S_regcppop.** The per-iteration cost × 183.9 M, plus the setup's +53.4 M instructions at one cycle each.
- **Denominator, derived.** ba-b3's full perl recorded no run_cycles, so 13,994 s × about 24 M cycles/s (mcf's and gcc's
  in-context rate on F) gives about 336 G.

| pair, model | S_regcppop | Perl_yyparse | rest | total | % of F's 336 G |
|---|---:|---:|---:|---:|---:|
| f4, F | −0.99 + 0.05 = −0.94 G | −0.02..−0.07 G | +0.01 G | −0.95..−1.00 G | −0.28..−0.30 % |
| unaligned (the board images), F | −1.06 + 0.05 = −1.01 G | −0.02..−0.07 G | +0.01 G | −1.02..−1.07 G | −0.30..−0.32 %, plus the term from 2,722 moved functions |
| f4, G | −0.90 + 0.05 = −0.84 G | −0.02..−0.07 G | +0.01 G | −0.85..−0.90 G | G's perl cycles are not known here |
| unaligned, G | −0.97 + 0.05 = −0.92 G | −0.02..−0.07 G | +0.01 G | −0.93..−0.98 G | |

Fidelity limits:
- the substitution of `add` for `cincoffset`;
- perl's trip count, about 13.8 iterations per call against the test's 512;
- the test's warm buffer.

Two earlier versions of this table went to the SPEC lane the same afternoon and are withdrawn:
- one from the 3-instruction model: −0.05..−0.15 % for the f4 pair;
- one from the 062f4453b model: −0.17..−0.18 %.
With them goes the recommendation built on them, that the f4 perl pair should not run.

- **The f4 perl pair.** On F it predicts about −0.29 %: past the ±0.15 % same-board spread and the 0.2 % bar, so it is
  informative. Its arms are not shippable images (`jal` relaxation changes in perl, gcc and imagick).
- **The real images.** An unaligned perl pair is predicted at −0.30..−0.32 % on F from the changed code. Its unknown
  term is the placement of the 2,722 moved functions, the kind of term that cost mcf +0.30 %. Run on the event-15
  firmware, Δif_empty reads that term beside the cycles.

## The lesson

A QEMU instruction twin cannot see placement. Any codegen change that shifts unchanged hot functions needs one of two
controls before its silicon delta is attributed to what it emits:
- a placement-controlled pair;
- event 15 read beside the cycles.

A directed test speaks for silicon only when its model is built at the bitstream's own RTL revision. The shared
checkout's HEAD was another line of the RTL, and both its tables had to be withdrawn.

Function alignment controls entries only. When the change rewrites the hot function itself (perl's S_regcppop), its
loop heads move with the body, and they must be read by hand in every image of the pair.
