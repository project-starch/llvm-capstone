# SPEC CPU2017 on silicon: riscv-base, Capstone temporal, Capstone spatial

This is a live table: rows are filled as board runs land. Pre-registrations, predictions and the run-by-run trail are
in [`../2026-11-08-train/PREREG.md`](../2026-11-08-train/PREREG.md) (Addendum 107 onward for this table).

## Status (updated 2026-11-14 12:15 +08)

- **riscv-base:** complete, all 8 rows.
- **Capstone temporal:**
  - The table below holds the best silicon reading per row, with the bitstream named on each row.
  - The final uniform configuration (one bitstream and one image set for every row) is being measured now.
  - 5 of 8 rows are already below the 1.20 target. perl, gcc and mcf are above it.
- **Capstone spatial:** all 8 images are built (arm FRCDKMSsp).
  - The SDK differs from temporal in 4 libc mallocng members plus the runtime's malloc-mapping object, and nothing else.
    This is shown by member hashes, and a control on the memset switch fires.
  - Every benchmark object is byte-identical to temporal FRCDKMS's.
  - QEMU twins (outputs must equal temporal's) are landing. gcc passes: outputs identical, user instructions +0.01 %, so
    in instruction count spatial is no cheaper. Lifetimes cost per-instruction latency (CDERIVE/CREVOKE and node work),
    which only silicon shows. Spatial gcc is queued on board 1; the other rows follow their twins.
- **Expected completion:** short rows today; x264 and the full perl test take ~10 h and ~5 h per arm, so the full
  three-arm table is expected on 15 Nov.

### Running now (three FPGA boards, serialized by bookings)

| Board | Now | Then |
|---|---|---|
| 1 | temporal mcf, batch / FRCDKMS (until ~13:00) | 1. temporal imagick and nab, batch / FRCDKMS<br>2. **spatial gcc**, batch / FRCDKMSsp (the first spatial row; twin outputs identical to temporal)<br>3. mcf, gcc and perl makerand on bitstream **F**, for the one-variable F-vs-batch pairs<br>4. F's tag-path counters (mcf, gcc) |
| 2 | x264, fdk / FR (until ~15:00) | bitstream **B** (old tag cache + tag-path counters): mcf and gcc, for where the memory time goes |
| 3 | full perl, fdk / FRCDK (until ~16:00) | the long final rows (x264, full perl) on the chosen final bitstream |

If F beats the batch on mcf and gcc (readings due ~16:00-18:00), F becomes the final temporal configuration, and the
table records the switch.

### Bitstreams referenced here

| Short name | Full name | sha256 | What it adds |
|---|---|---|---|
| riscv-base | `caplifive_rvbase_d31345cf3` | (sealed) | stock CVA6 |
| mlp | `caplifive_ids55-native-tagcache-mlp_b98f5a517` | 5e8421a5… | Capstone, 55-bit lifetime ids, native route, lifetime-record (tag) cache, memory-level parallelism |
| fdk | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k_3c6b9906c` | 141f840e… | mlp + FP fix + HPM counters + LDC/STC bypassing the DYN unit (v3) + DTLB 64 entries + SHRINKTO/CAPBOUND fix + I-cache 32 KiB 8-way |
| batch | `…-ic32k-r72-stcnoclaim_a47b588a1` | ff36870d… | fdk + R-72 (PCC fetch-end check) + lever (a) (an STC claims rs1 only when it is UNINIT) |
| F (next) | `…-ic32k-r72-stcnoclaim-twc16k-merge-hpmtag_a948e4134` | ec675c2b… | batch + a 16 K-entry tag-word cache (2 MB reach, was 32 KB) with merge-on-write, plus tag-path HPM counters. Sealed 14 Nov; first board runs pending |
| B (control) | `…-ic32k-r72-stcnoclaim-hpmtag_afe4e643e` | 5f8273b4… | batch + tag-path HPM counters only (old tag cache); a diagnostic, not a table configuration |

### How we got here (temporal ratio, problem rows)

| Step | perl | gcc train01 | mcf |
|---|---|---|---|
| 13 Nov, arm C (mlp, old compiler) | 1.551 (makerand) | 1.720 | 1.484 |
| + new compiler FR, all-optimisations bitstream | 1.509 (makerand) | 1.701 | 1.485 |
| + DYN bypass v3, DTLB 64 (final 6d0e754d5) | 1.398 (makerand) | 1.644 | 1.447 |
| + I-cache 8-way (fdk) | 1.347 (makerand); full test 1.394 | 1.506 | 1.440 |
| + RVC images (FRCDK) | 1.300 (makerand) | 1.475 | — |
| + R-72, lever (a) (batch) | 1.305 (makerand) | 1.467 | 1.434 |
| + C8, memset (FRCDKMS) | 1.278 (makerand) | **1.420** | running |

perl makerand is the quick sub-run used for A/B work; the table's perl row is the full test.

### Open caveats a reader should know

- **x264 correctness.** It is checked by the five decoded frames' sha256 against plain-rv64r's board run, because the
  native x264 stage has no imagevalidate step. specdiff therefore reads "0/3" on x264, which is not a failure.
- **riscv-base xz.** It is the no-cache build. In QEMU the mapping cache moves xz by 0.07 %.
- **Cross-board spread.** The same image on boards 1 and 3 differs by 0.9 % on perl makerand, and run-to-run spread on
  one board is 0.06 %. Ratios are same-board where possible: perl full and x264 run on board 3, the riscv-base board.
- **Tag-cache fill-time safety (ISSUES R-74, open).** It is verified in simulation on one window instance per design
  (batch and F) with the FPGA's crossbar setting, and forging controls fire. It is not proven for all timings. This
  bears on safety claims, not on these performance numbers.

## The three arms

| Arm | Bitstream | Software |
|---|---|---|
| **riscv-base** | stock CVA6, `caplifive_rvbase_d31345cf3` | plain RV64 build `plain-rv64fre`: static musl with the same allocator policy as Capstone (mapping cache with page rounding, eviction, in-place large realloc via mremap), K2N kernel |
| **Capstone temporal** (spatial + temporal safety) | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_a47b588a1` (sha256 `ff36870d…76e1`) | FRCDKMS images (below), native route, heap lifetimes ON |
| **Capstone spatial** (spatial safety only) | the same bitstream | the same compiler and benchmark objects; the SDK differs only in `CAPSTONE_HEAP_LIFETIMES=OFF`, so only in the allocator objects (libc mallocng, the runtime's malloc-mapping). An allocation gets exact bounds (CSHRINK) under its mapping's lifetime node instead of a node of its own (CDERIVE), and free does no CREVOKE. Size classes, the mapping cache, eviction and in-place realloc are unchanged |

The rule for the baseline: riscv-base keeps its own RTL (stock CVA6) and shares everything else that both arms
can share: compiler front end and flags, static linking, kernel, inputs, measurement, allocator policy.

**Capstone images (FRCDKMS).**
- Compiler: llvm-capstone llvm-c9 = `compiler-capbase-rvc-c8` 29027126f39b.
- Flags:
  - the global-lo fold with a 2 KiB-padded gp region;
  - call relaxation;
  - standard RVC plus capability-base RVC;
  - cost-per-use off;
  - C8, the inline run-time-aligned 16-byte memcpy.
- SDK: the arms' SDK with the `CAPSTONE_MEMSET_CAP_ZERO` libc switch (`runtime-memset-cap-zero-arms` beb1bbabe26a).
- Allocator: mapping cache, eviction and in-place realloc.

**Sizes.** gcc runs at train01; the rest at test. Ratio = Capstone seconds / riscv-base seconds. Correctness is
specdiff PASS on every row. x264 is checked against plain-rv64r's five frame hashes, because its native stage has no
imagevalidate step.

## Table: best measured so far (updated as runs land)

Each Capstone temporal number is the best silicon reading for that row so far, with its configuration. The last
two columns are the uniform final configuration (batch bitstream + FRCDKMS images), temporal and spatial, which is
the table being measured now. A row moves into them as its run lands.

| Benchmark | riscv-base (s) | Capstone temporal, best so far (s) | ratio | configuration | final config: temporal ratio | final config: spatial ratio |
|---|---|---|---|---|---|---|
| 500.perlbench_r (full: makerand + test.pl) | 12,623.02 | 17,601.71 | 1.394 | fdk / FR | pending | pending |
| 502.gcc_r (train01) | 1,722.35 | 2,446.30 | 1.420 | batch / FRCDKMS | 1.420 | pending |
| 505.mcf_r | 5,125.02 | 7,349.81 | 1.434 | batch / FR | running | pending |
| 519.lbm_r | 1,077.50 | 994.66 | **0.923** | fdk / FR | pending | pending |
| 525.x264_r | 35,613.24 | 36,219.53 | **1.017** | mlp / arm C | pending | pending |
| 538.imagick_r | 11.08 | 12.93 | **1.167** | fdk / FR | pending | pending |
| 544.nab_r | 679.92 | 716.18 | **1.053** | fdk / FR | pending | pending |
| 557.xz_r | 4,993.81 | 5,196.56 | **1.041** | fdk / FR | pending | pending |
| geomean | | | 1.172 | | | |

Bold = below the 1.20 target. perl makerand alone, the quick A/B sub-run used for debugging only: 257.90 s on
riscv-base against 329.70 s on batch / FRCDKMS, ratio 1.278.

Configurations in the table:
- **mlp / arm C:** `caplifive_ids55-native-tagcache-mlp_b98f5a517`, the older compiler, and the same allocator.
- **fdk:** `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k_3c6b9906c`.
- **batch:** fdk plus R-72 plus lever (a), the bitstream in the arm table above.
- **FR:** the global-lo fold plus relaxation.
- **FRCDK:** FR plus RVC.
- **FRCDKMS:** FRCDK plus C8 and the memset switch.

The riscv-base xz reading is its no-cache build. In QEMU the mapping cache does not move xz: 59.618 G against
59.576 G instructions.

## Where the time goes

**Per-row counters.** Each Capstone row's native counters line carries hpm3..8: L1I miss, L1D miss, DTLB walk,
mispredict, issue stall, scoreboard full. The rows are in perf/natstage/RESULTS on the measurement
host, and are summarised here when the table closes.

**Bitstream F.** A 16 K-entry tag-word cache with merge-on-write (capstone-ariane a948e4134) is the next bitstream.
It targets the tag read paid on L1D misses to capability-holding pages. If it helps, its rows supersede the temporal
column, and the change is recorded here.
