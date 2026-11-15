# SPEC CPU2017 on silicon: riscv-base, Capstone temporal, Capstone spatial

This is a live table: rows are filled as board runs land. Pre-registrations, predictions and the run-by-run trail are
in [`../2026-11-08-train/PREREG.md`](../2026-11-08-train/PREREG.md) (Addendum 107 onward for this table).

## Status (updated 2026-11-15 14:36 +08)

- **riscv-base:** complete, all 8 rows.
- **Capstone temporal:**
  - The table below holds the best silicon reading per row, with the bitstream named on each row.
  - The final uniform configuration (one bitstream and one image set for every row) is being measured now.
  - 7 of 8 rows are below the 1.20 target on F. Only mcf (1.283) is above it.
- **Capstone spatial:** all 8 images are built (arm FRCDKMSsp). gcc train01 on F: 1,975.51 s (ratio 1.147) against temporal's 1,989.28 s on the same bitstream, board and kit, so the temporal increment on gcc is +0.70 % (+0.6 % on the batch bitstream). mcf on F: 6,308.53 s (ratio 1.231) against temporal's 6,576.13 s, a temporal increment of +4.2 % (outside Addendum 113's pre-registered spatial band of -3..+0.5 % from temporal, on the fast side); L1D misses are 4.9 % lower than temporal's. imagick 1.134, nab 1.021 and lbm 0.919 on F are within 0.2 % of temporal. xz 1.001 (temporal 1.001). x264 0.870 on board 2 (temporal's 30,975.87 s ran on board 3: a cross-board pair). Full perl on F: 13,981.09 s on board 1 (ratio 1.108). Its temporal pair ran on board 3 (14,151.55 s), so the +1.2 % temporal increment on perl is cross-board, against a ~0.9 % cross-board spread. The spatial column is complete.
  - The SDK differs from temporal in 4 libc mallocng members plus the runtime's malloc-mapping object, and nothing else.
    This is shown by member hashes, and a control on the memset switch fires.
  - Every benchmark object is byte-identical to temporal FRCDKMS's.
  - QEMU twins (outputs must equal temporal's) are landing. gcc passes: outputs identical, user instructions +0.01 %, so
    in instruction count spatial is no cheaper. Lifetimes cost per-instruction latency (CDERIVE/CREVOKE and node work),
    which only silicon shows. Spatial gcc is queued on board 1. The mcf and perl twins also pass (outputs identical, user +0.00 % and +0.02 %).
    The other twins have finished too: imagick, nab, lbm and xz rc 0 with full specdiff. perl full test is 2/2 on
    the board-matching kernel (images-a0ed). x264 reads specdiff 0/3, as every native x264 stage does: its frames are
    checked by hash on the board run.
- **LSR compiler change c13: NOT adopted** (Addendum 136). mcf built with it ran +0.29 % slower on the same board
  (in-context cycles +0.30 %); the rule allowed at most +0.15 %. The full-perl run on c13 was dropped.
- **Node-table pool: ADOPTED** (Addendum 133). Temporal full perl on F with `node_table_pool=4` (released lifetime
  tables are reused instead of allocated and zeroed for every exec) ran 14,005.42 s, against 14,151.55 s without it on
  the same board: -1.03 %, inside the predicted -0.5..-3 %. specdiff 2/2. The pool is a change to the Capstone kernel
  module only, and riscv-base has no such table. The temporal perl cell is now 1.110. Spatial full perl reruns on the
  pool next on board 2. Caveat: the kit does not read the pool's hit counter, so the evidence that the pool ran on
  silicon is the kernel tree checksum and the parameter, not a counter.
- **Expected completion:** short rows today; x264 and the full perl test take ~10 h and ~5 h per arm, so the full
  three-arm table is expected on 15 Nov.

### Running now (three FPGA boards, serialized by bookings)

**F is the final temporal bitstream** (Addendum 113). It beats the batch by -10.8 % on mcf and -18.7 % on gcc with the
same images. The "final config" columns are being re-measured on F.

| Board | Now | Then, in order |
|---|---|---|
| 1 | bitstream **G** (overlap lever v6c): flash, then mcf and gcc with tag-path counters (Addendum 132, from 14:23) | perl makerand on G |
| 2 | spatial full perl on the node-table pool (Addendum 133, from 14:34) | the nested lane's tshark heartbeat boot (Addendum 134) |
| 3 | allocator step FRCDKMSa3 full perl (Addendum 121, from 11:14) | spatial gcc untuned (Addendum 135) |

- **F's scope.** The gain is the tag-word-cache change, reach and merge together, at N = 1 per row.
  - It comes at unchanged L1I/L1D misses, DTLB walks and mispredicts.
  - On mcf, the issue-stall drop covers 18.1 of the 19.75 G-cycle fall.
  - On gcc, issue plus scoreboard stalls cover only 40 % of the 11.3 G-cycle fall. The tag-path counters account for
    all of it: the adapter's tag-phase cycles fall by 11.8 G, mostly tag writes (1.29 G → 0.42 G).
  - Tag-path counters, config B (old cache) → F, same images:
    - mcf: tag reads 1.62 → 0.81 G, tag writes 0.82 → 0.24 G, tag-cache miss rate 48 % → 24 %;
    - gcc: tag reads 157 → 73 M, tag writes 1.29 → 0.42 G;
    - L1D misses and page fills unchanged on both.
  - Repeats with the tag-counter firmware on the same board: mcf +0.08 %, gcc +0.04 %.
- **Next bitstream.** F2 (F + targeted fill invalidation, eeb654104) is sealed and flash-ready. L1D 64 KiB
  (b0f02bac7) failed timing and is not flashed.
- **gp-relative globals: NOT adopted** (Addendum 120). On F, gp sped up perl makerand by 8.7 % but slowed gcc train01
  by 0.62 % (1,992.35 → 2,004.71 s). On riscv-base it sped up gcc by 0.60 %, so gcc's ratio worsens from 1.157 to
  1.171. The pre-registered rule (Addendum 114) rejects gp on both arms when gcc's ratio worsens. The table keeps
  FRCDKMS and RVF3.

### Bitstreams referenced here

| Short name | Full name | sha256 | What it adds |
|---|---|---|---|
| riscv-base | `caplifive_rvbase_d31345cf3` | (sealed) | stock CVA6 |
| mlp | `caplifive_ids55-native-tagcache-mlp_b98f5a517` | 5e8421a5… | Capstone, 55-bit lifetime ids, native route, lifetime-record (tag) cache, memory-level parallelism |
| fdk | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k_3c6b9906c` | 141f840e… | mlp + FP fix + HPM counters + LDC/STC bypassing the DYN unit (v3) + DTLB 64 entries + SHRINKTO/CAPBOUND fix + I-cache 32 KiB 8-way |
| batch | `…-ic32k-r72-stcnoclaim_a47b588a1` | ff36870d… | fdk + R-72 (PCC fetch-end check) + lever (a) (an STC claims rs1 only when it is UNINIT) |
| **F** (final) | `…-ic32k-r72-stcnoclaim-twc16k-merge-hpmtag_a948e4134` | ec675c2b… | batch + a 16 K-entry tag-word cache (2 MB reach, was 32 KB) with merge-on-write, plus tag-path HPM counters. Sealed 14 Nov; the final temporal and spatial bitstream (Addendum 113) |
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
| + C8, memset (FRCDKMS) | 1.278 (makerand) | **1.420** | 1.438 (FR's 1.434 stays the best) |

perl makerand is the quick sub-run used for A/B work; the table's perl row is the full test.

### Open caveats a reader should know

- **gcc no longer uses a gcc-only module tuning** (Addendum 135). The gcc stages used to write module parameters for
  their run, and no other row did. On F's kernel only `arena_pin=1` took effect: the other two parameters
  (step_budget_ns, resume_quantum) do not exist there, and those writes failed. Measured on the same board, arena_pin
  is worth 0.67 % on gcc (1,989.28 s tuned against 2,002.68 s untuned). The table's gcc cells are now the untuned,
  uniform default configuration. The tuned 1.155 is kept only as disclosed tuning. Spatial gcc untuned is queued on
  board 3.

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
| **Capstone temporal** (spatial + temporal safety) | **F**: `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim-twc16k-merge-hpmtag_a948e4134` (sha256 `ec675c2b…`). Until 14 Nov evening it was batch, `…_a47b588a1` (`ff36870d…76e1`) | FRCDKMS images (below), native route, heap lifetimes ON |
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
two columns are the uniform final configuration (bitstream F + FRCDKMS images), temporal and spatial, which is
the table being measured now. A row moves into them as its run lands.

| Benchmark | riscv-base (s) | Capstone temporal, best so far (s) | ratio | configuration | final config: temporal ratio | final config: spatial ratio |
|---|---|---|---|---|---|---|
| 500.perlbench_r (full: makerand + test.pl) | 12,623.02 | 14,005.42 | **1.110** | **F** / FRCDKMS + node-table pool | **1.110** (F + pool) | **1.108** (F, no pool yet; board 1, cross-board) |
| 502.gcc_r (train01) | 1,722.35 | 2,002.68 | **1.163** | **F** / FRCDKMS, no module tuning (tuned: 1,989.28 s, 1.155) | **1.163** (F) | **1.147** (F, tuned; untuned pending) |
| 505.mcf_r | 5,125.02 | 6,576.13 | 1.283 | **F** / FRCDKMS | 1.283 (F) | 1.231 (F) |
| 519.lbm_r | 1,077.50 | 990.36 | **0.919** | **F** / FRCDKMS | **0.919** (F) | **0.919** (F) |
| 525.x264_r | 35,613.24 | 30,975.87 | **0.870** | **F** / FRCDKMS | **0.870** (F) | **0.870** (F, board 2) |
| 538.imagick_r | 11.08 | 12.58 | **1.135** | **F** / FRCDKMS | **1.135** (F) | **1.134** (F) |
| 544.nab_r | 679.92 | 695.34 | **1.023** | **F** / FRCDKMS | **1.023** (F) | **1.021** (F) |
| 557.xz_r | 4,993.81 | 4,998.36 | **1.001** | **F** / FRCDKMS | **1.001** (F) | **1.001** (F) |
| geomean | | | 1.055 | | **1.055** (F) | **1.047** (F, gcc tuned) |

Bold = below the 1.20 target. perl makerand alone (the quick A/B sub-run, debugging only): 257.90 s on riscv-base
against 288.58 s on **F** / FRCDKMS, ratio 1.119 (-12.5 % against the batch). The full test on F ran on board 3, the riscv-base board: makerand 292.76 s + test.pl 13,858.79 s = 14,151.55 s, ratio 1.121 (fdk/FRCDK 17,051.48 s: -17 %). Board 2's attempt at 21:07 never booted, because switches were left
at an odd value by the previous run (Addendum 119), and no benchmark ran.

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
