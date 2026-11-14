# SPEC CPU2017 on silicon: riscv-base, Capstone temporal, Capstone spatial

This is a live table: rows are filled as board runs land. Pre-registrations, predictions and the run-by-run trail are
in [`../2026-11-08-train/PREREG.md`](../2026-11-08-train/PREREG.md) (Addendum 107 onward for this table).

## The three arms

| Arm | Bitstream | Software |
|---|---|---|
| **riscv-base** | stock CVA6, `caplifive_rvbase_d31345cf3` | plain RV64 build `plain-rv64fre`: static musl with the same allocator policy as Capstone (mapping cache with page rounding, eviction, in-place large realloc via mremap), K2N kernel |
| **Capstone temporal** (spatial + temporal safety) | `caplifive_ids55-native-tagcache-mlp-fpieee-hpmreplay-dynbypass3-dtlb64-shrinkfix-ic32k-r72-stcnoclaim_a47b588a1` (sha256 `ff36870d…76e1`) | FRCDKMS images (below), native route, heap lifetimes ON |
| **Capstone spatial** (spatial safety only) | the same bitstream | the same compiler and benchmark objects; the SDK differs only in the heap object (heap lifetimes OFF) |

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

## Table

| Benchmark | riscv-base (s) | Capstone temporal (s) | ratio | Capstone spatial (s) | ratio |
|---|---|---|---|---|---|
| 500.perlbench_r (full: makerand + test.pl) | 12,623.02 | pending | | pending | |
| 502.gcc_r (train01) | 1,722.35 | 2,446.30 | 1.420 | pending | |
| 505.mcf_r | 5,125.02 | running | | pending | |
| 519.lbm_r | 1,077.50 | pending | | pending | |
| 525.x264_r | 35,613.24 | pending | | pending | |
| 538.imagick_r | 11.08 | pending | | pending | |
| 544.nab_r | 679.92 | pending | | pending | |
| 557.xz_r | 4,993.81 | pending | | pending | |
| geomean | | | | | |

**Readings on earlier configurations** (the same riscv-base column):

| Row | Ratio | Configuration |
|---|---|---|
| perl full | 1.394 | fdk / FR |
| perl makerand | 1.278 | batch / FRCDKMS |
| mcf | 1.434 | batch / FR |
| lbm | 0.923 | fdk / FR |
| x264 | 1.017 | mlp / arm C |
| imagick | 1.167 | fdk / FR |
| nab | 1.053 | fdk / FR |
| xz | 1.041 | fdk / FR |

The riscv-base xz reading is its no-cache build. In QEMU the mapping cache does not move xz: 59.618 G against 59.576 G
instructions.

## Where the time goes

**Per-row counters.** Each Capstone row's native counters line carries hpm3..8: L1I miss, L1D miss, DTLB walk,
mispredict, issue stall, scoreboard full. The rows are in perf/natstage/RESULTS on the measurement
host, and are summarised here when the table closes.

**Bitstream F.** A 16 K-entry tag-word cache with merge-on-write (capstone-ariane a948e4134) is the next bitstream.
It targets the tag read paid on L1D misses to capability-holding pages. If it helps, its rows supersede the temporal
column, and the change is recorded here.
