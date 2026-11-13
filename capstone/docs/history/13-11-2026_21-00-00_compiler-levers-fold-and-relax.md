# Compiler levers for the problem rows: the global-address fold and call relaxation (2026-11-13)

The runtime lane asked for the instruction excess of the Capstone rows over riscv-base (perl makerand +20.6 %, gcc
train01 +15.4 %) to be cut by compiler changes. Each change was to be twin-validated before the board: outputs
byte-identical and an instruction delta per benchmark. This records what was measured, what changed, and what was
left.

## Where the excess was

Per-PC twin profiles (capmix3, shift 2) of the arm-C images, by mnemonic. The Capstone-side counts are exact.
riscv-base is compared by class only, because its RVC pairs share a 4-byte bucket.

- **perl makerand:** the user excess is +566.9 M. Every global access is `auipc; addi %pcrel_lo; scc gp; access`,
  where riscv-base has `lui; access`. That is two extra instructions per access over 238 M executed `scc gp`
  derivations, about 476 M or 84 % of the excess. Of those derivations, 84 % have no use but as the base of a
  load/store in straight-line code (fold-eligible). In-loop re-derivation is 0 %.
- **gcc train01:** the excess is +1.65 G, with no dominant cause:
  - global derivations, 222 M executed `scc gp` (30 % fold-eligible in straight-line code);
  - `auipc+jalr` direct calls, 182 M executed, of which 94.6 M are within ±1 MiB;
  - shrink/shrinkto, 128 M;
  - `movc`/`cincoffset` against `mv`/`add`, about +210 M.
- **Ceiling.** On silicon, gcc's ratio is 1.58-1.81 against an instruction ratio of 1.15-1.17, so most of the silicon
  gap is not instruction count.

## The two changes

- **Fold** (`-mllvm -capstone-fold-global-lo`, compiler-global-lo-fold 2f3ff76eba7b, default off). It drops the
  `addi`: the capability is derived from the auipc's partial address and each access carries the `%pcrel_lo`. It
  relaxes the cursor-preservation rule for this fold alone. The project lead approved it on 2026-11-13, through the
  runtime lane. The conditions, and the 2 KiB gp margins in the virtual link script, are in
  `docs/design/capability-loop-addressing.md`, "The global-address exception".
- **Call relaxation** (the four codegen words `-Xclang -target-feature -Xclang +relax` in the SDK's `sdk.json`; no
  compiler change). lld relaxes `auipc+jalr` calls into `jal` where the target is within ±1 MiB. `capstone-cc` refused
  `+relax` only because it fixes the profile's features (9806524b1caa), not for a correctness reason. libc and the
  runtime stay unrelaxed. `CAPSTONE_SDK_EXTRA_CODEGEN` cannot add these words: it de-duplicates word by word
  (`ports/common/application/build-sdk.sh:34`).

## Validation

- **Relaxed images, structurally.** `perf/ccrelax/relaxdiff.py` compares every function and every data word of each
  relaxed image against arm C as locations: a code address as (function, normalised instruction index), any other as
  (section, offset). It covers calls, branches, auipc+addi pairs (including ones spilled through stack slots), jump
  tables, `.data` pointers, `.capstone_cap_init` and `.gct`. All 11 executables pass: 0 functions differ and 0 data
  words are unexplained. Four mutations of one relaxed image each fail it: a call target +4, a pcrel low part +8, a
  jump-table entry +4, a `.data` pointer +16. The first version passed the call-target mutation, because it reduced
  calls into a function to the function's name; that was fixed before any PASS was recorded.
- **Folded images.** `perf/cclever/foldcheck.py` checks that every folded cursor in all eleven F and FR images lies
  inside gp's bounds (binfmt-dom.h:156, :449). The tightest margin is 0x30414 bytes. An auipc moved past gp's end
  fails the check.
- **Flag off is inert.** Arm Z (this compiler, flag off, arm C's link script) builds perl byte-identical to arm C. gcc
  differs only in its embedded compiler-version string and the merged string pool that string moves.
- **lit:** CodeGen/Capstone and MC/Capstone, 168 tests; only the known `shared-patches-present.test` fails.
  `cap-fold-global-lo.ll` fails on the previous compiler.
- **QEMU gate** (`c1-gate/gt2/gate-fold`, a clone of the gate -capstone-direct-calls landed on): the runtime/virtual
  contracts at -O2, flag off, fold, and fold+relax. hosted passes 93/93 and fp (2 slots) 10/10 in all three. malloc and
  nodes fail with byte-identical failure sets in all three, the known ABI-7 environment. BEEBS, RV8 and authority
  were not run: they build under the physical profile, where the fold is inert by construction and `+relax` is
  refused.

## Twin results against arm C

Arm C: SDK 3e70344d8c3f, llvm-c4. R is call relaxation alone (llvm-c4). F is the fold alone (llvm-c6 = this branch)
with the padded link script. FR is both. The setup is the same in every arm: native twin, ids55, natio k3 kernel,
capmix3 shift 24. User instructions summed over each stage's commands:

| Benchmark (test input; gcc train01) | Arm C user | R: relax | F: fold | FR: both |
|---|---|---|---|---|
| 500.perlbench_r-test-sw | 4,241,793,708 | -32,757,874 (-0.77 %) | -205,490,571 (-4.84 %) | -238,247,457 (-5.62 %) |
| 502.gcc_r-train01 | 12,587,873,860 | -95,561,708 (-0.76 %) | -55,517,421 (-0.44 %) | -151,185,659 (-1.20 %) |
| 505.mcf_r-test | 26,715,621,763 | -17,541,116 (-0.07 %) | -71,039 (-0.00 %) | -17,612,140 (-0.07 %) |
| 519.lbm_r-test | 6,095,188,927 | -2,626,351 (-0.04 %) | +0 (+0.00 %) | -2,635,526 (-0.04 %) |
| 525.x264_r-test | 532,622,941,133 | -464,864,335 (-0.09 %) | -561,945 (-0.00 %) | -465,426,233 (-0.09 %) |
| 538.imagick_r-test-ref | 157,360,783 | -2,376,570 (-1.51 %) | -1,275 (-0.00 %) | -2,377,607 (-1.51 %) |
| 544.nab_r-test | 7,179,002,205 | -23,068,379 (-0.32 %) | -30,131 (-0.00 %) | -23,098,510 (-0.32 %) |
| 557.xz_r-test | 49,705,166,905 | -56,019,754 (-0.11 %) | -39,936 (-0.00 %) | -56,059,725 (-0.11 %) |

- **Outputs.** Every result file is byte-identical to arm C's in every arm. The one exception is x264's `log.dec`,
  whose line 6 is ldecod's wall-clock stamp (HH:MM of the run); it also differs between arm C's own A/B/C runs.
  Validation is unchanged in every cell.
- **perl makerand.** Window 0 alone: R -28.5 M, F -201.5 M (the fold-eligible count predicted 199.6 M), FR -230.1 M
  (-6.92 %). That takes makerand's excess over riscv-base from +566.9 M (+20.6 %) to +336.8 M (+12.2 %).
- **The full perl test** (makerand + test.pl with its ~300 sub-tests, on the working native test stage) is
  byte-identical, specdiff 2/2 in both, with test.out equal to riscv-base's. FR against C: user -1.72 G (-1.91 %).
  The stage uses `dom_grant_root`/`dom_null`/`dom_local_tmp`, and node_flat_records and native_slot_transport are
  dropped to fit the twin kernel's 511-byte command line; this applies to both arms.
- **Relax costs kernel work per exec of the image.** On the full perl test: kernel +25.5 M (+0.11 %), pages +7.8 k,
  about 26 more pages per exec. That is 1.5 % of the user saving. The fold alone does not cause it. On xz a per-PC pair
  puts the relaxed image's kernel delta at +82 k (idle and tick code), i.e. noise.

## On silicon (the runtime lane's reading, not re-checked here)

Kit bkit60 (the all-optimisations bitstream, bkit59's kernel), perl makerand: FR 389.36 s against arm C's 400.04 s
(-2.7 %), against an instruction delta of -6.92 %.

## Left, measured but not built

Each of these is at most about 0.4 % of gcc and nothing on perl:
- **Stack-object bounds** use register-form SHRINK (cursor read, size, add, shrink), where SHRINKTO with an immediate
  would do. That is about 11-20 M sites executed on gcc. Before building it, the SHRINKTO end semantics need settling:
  QEMU's helper ends at cursor + imm and checks size <= end - cursor, while current bitstreams end at
  cursor + imm - 1 (`-capstone-shrinkto-end-adjust`). Under the adjust, an object ending at its parent's end would
  fail QEMU's check.
- **Bounded global capabilities re-derived inside loops** (scc + shrinkto per iteration after a spill): about 16 M
  sites on gcc.
- **A no-op `mv a1, a1`** in every jump-table dispatch.

The unused AND-mask commit (143df5bccb25) shows no excess at this resolution: `and`+`andi` net about +30 M on gcc,
within the RVC-bucket error.
