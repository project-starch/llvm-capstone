# SPEC CPU2017 on silicon: riscv-base, Capstone temporal, Capstone spatial

This is a live table: rows are filled as board runs land. Pre-registrations, predictions and the run-by-run trail are
in [`../2026-11-08-train/PREREG.md`](../2026-11-08-train/PREREG.md) (Addendum 107 onward for this table).

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
