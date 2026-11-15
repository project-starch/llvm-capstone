# riscv-base perl RVAL +0.33 %: an I-cache set conflict on the 16 KiB 4-way I-cache, not a fetch-alignment cost (model attribution, not confirmed on silicon)

Date: 2026-11-15. Lanes: compiler (this note), runtime/SPEC (the board readings, train PREREG Addenda 139 and 147).
Status: the cause is attributed in a model replay, with a predicted fix checked in the same model. It is not measured on
silicon: riscv-base kits have no HPM bracket, so no board counter can confirm it today.

## The reading

Addendum 139 put riscv-base on RVAL, RVF3 plus 0012's and 0013's generic parts. Board 3, bitstream
caplifive_rvbase_d31345cf3, the same kits and board as RVF3's rows:

| | RVF3 (5210db3eeedada2f) | RVAL (9304130748a03c57) | Δ |
|---|---:|---:|---:|
| perl makerand | 258.00 s (repeats 257.9, 258.0) | 267.11 s | +3.53 %, +9.1 s = about 228 M cycles at 25 MHz |
| perl test.pl | 12,365.02 s | 12,397.94 s | +0.27 %, +32.9 s = about 823 M cycles |
| perl full | 12,623.02 s | 12,665.05 s | +0.33 % (band -0.1 .. -0.4 %, missed on the slow side) |
| gcc train01 | 1,722.35 s | 1,720.98 s | -0.08 % (inside its band) |

The QEMU twins had perl full -0.24 % user instructions. makerand's window: 2,758,077,855 against 2,758,077,628 (-227).
So makerand executes the same instructions in both images and is 3.5 % slower.

## Not the loop-head rule

- **Placement diff, RVF3 against RVAL.** Every perl function after S_incpush_use_sep moved by exactly +4 bytes. The
  cause is one `jal memchr` that lld relaxed to `auipc+jalr` after the musl grew by about 200 B (0013's get_meta hunk
  and 0012's context pointer). A +4 shift leaves bit 1 of every address unchanged, so on F's and G's rule (a 32-bit
  loop head with bit 1 set costs one bubble per entry) it costs nothing.
- **The rule holds on riscv-base's own RTL.** I re-ran both directed tests on the riscv-base model,
  capstone-ariane d31345cf3, target baseline_cv64a6_imafdc_sv39, seed 1. A worktree, the CAPPRINT macro replaced by a
  register move, recipe perf/lgep/rvsim/chain-rv.sh, provenance PROVENANCE-RV.txt.
  - fetch-realign-loop-head: 4.04-4.06 cycles per iteration at head 0/4/8/12 mod 16, and 5.04-5.06 at 2/6/10/14.
  - regcppop-loop-head: equal to F's to within 0.01.
  - The readings are in both tests' headers, capstone-ariane fetch-realign-loop-head 525bd0dde (comments only).

So the alignment term predicts 0 for RVAL, and the 228 M cycles need another mechanism.

## The cache geometries differ

From each bitstream's config package (capstone-ariane core/include/):

| | I-cache | D-cache |
|---|---|---|
| riscv-base, baseline_cv64a6_imafdc_sv39 at d31345cf3 | 16 KiB, 4-way, 128-bit (16 B) lines: 256 sets, index bits 11:4 | 32 KiB, 8-way, 16 B lines, write-through |
| F and G, capstone_cv64a6_imafdc_sv39 (G 9fd9d2530) | 32 KiB, 8-way, 16 B lines: 256 sets | the same |

Both I-caches have 256 sets. The difference is 4 ways against 8. Both caches choose the replacement way with an LFSR.
(An earlier message from this lane to the runtime lane said the two configurations had the same geometry. That was
wrong and was corrected the same day; the instrument's header says so too.)

## The mechanism: two sets each hold 5 hot lines in 4 ways

The +4 shift keeps bit 1 but moves lines across 16 B boundaries, so hot code lands in neighbouring sets. From the twin's
capmix3 profile of makerand (lines executed at least 100,000 times, grouped by set; perf/lgep/l1sim/hotsets.py):
- RVAL: 281 hot lines in 176 sets, and two sets hold 5 each:
  - set 22: Perl_do_readline, S_sv_2iuv_common, Perl_sv_2pvbyte, Perl_my_atof2, Perl_find_runcv_where (about 1.9 M each);
  - set 198: Perl_pp_readline (3.8 M), Perl_report_uninit (3.8 M), S_glob_assign_glob, Perl_pp_sne, Perl_pp_index (1.9 M each).
- RVF3: no set holds more than 4.

Five lines cycling through a 4-way set miss on most passes.

## The instrument: l1sim, an L1 replay under QEMU

perf/lgep/l1sim/l1sim.c and l1sim2.c are QEMU plugins, run through perf/lgep/life/run-life.sh with PLUGIN=:
- Each executed TB touches its instructions' 16 B lines in order. Tags are physical (host address for code, guest
  PA for data). The set index lies inside the page offset, so virtual and physical agree on it.
- l1sim2 models five I-caches at once: 16k4 LRU and random, 32k8 LRU and random, and a 32 KiB 32-way LRU reference
  (64 sets) that nearly removes set conflicts.
- Counters snapshot at capmix3's marks, so makerand is window 0 and test.pl window 1. A missing mark is an ERROR
  (l1read2.py), never a zero.
- Controls: an RVF3 repeat agreed within 0.1 %. The pad relink below is the positive control. It was predicted to
  remove the conflict before it ran, and it did.

User-mode I-misses per window. All three runs: specdiff 2/2, the same images and dom flags.

| | RVF3 | RVAL | RVALpad (pad 4080) |
|---|---:|---:|---:|
| makerand, 16k4 LRU | 233,936 | 15,407,563 | 216,787 |
| makerand, 16k4 random | 261,121 | 8,726,532 | 248,392 |
| makerand, 32k8 random | 40,225 | 44,089 | 40,590 |
| makerand, 32k32 LRU | 22,645 | 23,011 | 23,185 |
| test.pl, 16k4 random | 2,803,273,251 | 2,834,010,596 (+30.7 M) | 2,831,720,763 (+28.4 M) |
| test.pl, 32k8 random | 1,397,190,667 | 1,417,267,861 (+20.1 M) | 1,415,733,064 (+18.5 M) |
| test.pl, 32k32 LRU | 1,088,664,415 | 1,100,582,671 (+11.9 M) | 1,102,474,940 (+13.8 M) |

Reading it:
- **makerand.** RVAL adds 8.5 M (random) to 15.2 M (LRU) misses on riscv-base's geometry. On F's and G's geometry the
  difference is +3.9 K, nothing. For the board's +228 M cycles, that implies about 15-27 cycles per miss. This is an
  inferred penalty, not a measured one.
- **test.pl.** RVAL's +30.7 M random-policy misses at 15-27 cycles give 460-830 M cycles, against the board's +823 M.
  The pad does not remove this excess, and the 32k32 reference keeps it too. So it is not perl's set mapping. It
  comes from RVAL's own code and libc layout, which test.pl's much larger footprint touches. RVAL also executes 191 M
  fewer instructions in this window than RVF3.

## The layout-controlled relink (built, not run on the board)

perf/alean/plain/relink:
- ctl/ relinks RVAL from its own SPEC objects (build_base_plain-rv64al.0000, make.out line 480, plain-cc-alean). It
  reproduces 9304130748a03c57, which proves the command is RVAL's link.
- A retained `.text.layoutpad` section (pad.S) between perl.o and perlapi.o moves perl's code after the relaxed call.
- A single 4092-byte pad, the size that restores RVF3's addresses mod 4096, does not work: it pushes more calls out
  of `jal` range, and lld's relaxation cascades.
- So the pad size was swept (sweep.sh) and each result scored by the twin's per-function weights (score.py; the
  weighted share of hot code not congruent to RVF3 mod 4096). P = 4080 is the best: pads/4080/perlbench_r =
  a7810f1081e2af74, staged as alean/plain/st-perl-pad4080 for the replay only.
- A second pad after the last object (sweep2.sh) was rejected: hot perl lands at 72 mod 4096.

RVALpad's makerand misses return to RVF3's level, as predicted. No board run: whether to swap riscv-base's perl row
for a layout-controlled image is the lead's decision. Addendum 147 records the decision taken: stock riscv-base stays
the baseline, the I-cache difference is disclosed, and RVAL's 2-set conflict is disclosed with its row.

## The baseline disclosure estimate

Over test.pl, riscv-base's 16k4 cache misses about 2.80 G times, against about 1.40 G with F's 32k8 (both RVF3,
random policy). At the inferred 15-27 cycles per miss, the difference is 21-38 G cycles, about 7-12 % of riscv-base
perl's test.pl cycles (12,365 s at 25 MHz = 309 G). This is a model estimate with an inferred penalty. It went to the
runtime lane as the disclosure number in Addendum 147.

## The Capstone side

The question: does the Capstone side carry a layout lottery like RVAL's? It was checked on the two Capstone perl layouts
the table uses, FRCDKMS (stage alean/st-perl-frcdkms, a binary swap of st-perl-a3, perlbench_r a5794577321947cb) and
FRCDKMSa3 (alean/st-perl-a3, c1744e87cb2244f6).
- Configuration: images-a0ed with the board's dom flags, l1sim2-noD.
- Both runs: specdiff 2/2, chain perf/lgep/l1sim/chain-l1c2.sh.
- The first attempt, on the camp stages without the dom flags, failed test.pl (specdiff 1/2), so it was not read.

User-mode I-misses per window:

| | FRCDKMS | FRCDKMSa3 | a3 - FRCDKMS |
|---|---:|---:|---:|
| makerand, instructions | 3,094,854,536 | 3,094,833,788 | -20,748 |
| makerand, 32k8 random (F/G) | 97,229 | 96,617 | -612 |
| makerand, 32k32 LRU (reference) | 73,411 | 74,235 | |
| makerand, 16k4 random (riscv-base's geometry, for scale) | 563,434 | 554,216 | |
| test.pl, instructions | 86,198,630,456 | 85,462,712,400 | -0.85 % |
| test.pl, 32k8 random | 2,451,459,706 | 2,411,879,187 | -39.6 M (28.44 against 28.22 per 1,000 instructions) |
| test.pl, 32k32 LRU | 1,918,522,977 | 1,894,099,684 | |
| test.pl, 32k8 excess over 32k32 | 532,936,729 | 517,779,503 | -15.2 M |

Reading it:
- **makerand.** There is no lottery. Under 8 ways, either layout is about 23 K misses above the reference. At the
  inferred 15-27 cycles that is under 0.7 M cycles, below 0.01 % of makerand. The two layouts differ by 612 misses.
  The static check agrees: no set holds more than 4 hot lines in either layout, against 8 ways (hotsets.py).
- **test.pl.** About half of a3's 39.6 M fewer misses comes with its 0.85 % fewer instructions. The layout-dependent
  part, the 32k8 excess over the near-conflict-free reference, is 15.2 M lower. At 15-27 cycles that is 0.23-0.41 G
  cycles, about 0.1 % of Capstone test.pl, in a3's favour. So at most about 0.1 % of a3's measured gain over FRCDKMS
  is layout.
- **Scale.** The Capstone side's layout term is at least an order of magnitude below RVAL's +3.5 % makerand draw.
  Its 8-way I-cache is the reason: the same code mapping that overloads a 4-way set fits in 8.
- **The penalty.** 15-27 cycles per miss is riscv-base's inferred figure. Capstone's miss path could differ. The
  conclusion rests on the miss counts, which barely move, not on the penalty.

## Caveats

- **What the model leaves out.** QEMU's TB-granular fetch has no wrong-path fetch, no next-line prefetch and no fill
  timing. Random replacement is xorshift, not the RTL's LFSR. Only user mode is counted.
- **The penalty.** It is inferred from makerand's delta. It is not measured.
- **The Capstone guest.** It halts at boot when l1sim2's memory callbacks are registered (cause 24 in firmware), so
  the Capstone replays run l1sim2-noD, the I-cache models only.
- **i_acc.** The Capstone runs' i_acc is not comparable with riscv-base's: QEMU forms different TBs there. Miss counts
  are comparable.
- **Silicon.** No board counter confirms any of this yet. riscv-base kits have no HPM bracket.

## The lesson

The memory instruction-twins-are-blind-to-placement gains a second mechanism. A uniform shift that keeps bit 1 can
still overload I-cache sets, and an instruction twin cannot see it. Replay the layout with perf/lgep/l1sim before
reading a cross-image cycle delta on riscv-base, whose I-cache has half F's ways. Read the geometry from each
bitstream's config package; never assume two bitstreams match.
