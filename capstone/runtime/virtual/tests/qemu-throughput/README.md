# QEMU host-throughput experiment

Branch `perf/qemu-throughput`, based on the full-SPEC-qualified native branch at
`2b25f1714a5e`. This experiment measures emulator host time, not RTL cycles.
No capability checks, emulated semantics or runtime defaults are weakened.

Four controls use the same patched kernel, rootfs, OpenSBI and GCC test input:

| Arm | Emulator | Machine / guest executable |
| --- | --- | --- |
| stock-plain | Upstream QEMU 8.1.1 | `virt`, ordinary RV64 GCC |
| fork-plain | Capstone QEMU | `virt`, identical ordinary RV64 GCC |
| platform-plain | Capstone QEMU | `virt-capstone` with native features, ordinary RV64 GCC |
| platform-native | Capstone QEMU | Same Capstone platform, Capstone GCC |

Only the first pair holds the machine/CPU flags and application binary identical.
The platform controls deliberately expose different discovery and kernel paths.
The native application has different compiler/libc/ABI provenance. Their wall-time
ratio cannot be labeled emulator-only overhead or predicted hardware slowdown.

## Build and run

Keep all external sources and artifacts below `$CAPSTONE_TMP_ROOT`. The build
script exports pinned upstream/fork commits from an available QEMU repository
and configures both with identical options and the same host toolchain. It may
fetch QEMU's declared build dependencies. It does not install either binary.

```sh
source capstone/tests/capstone-test-env.sh
capstone/runtime/virtual/tests/qemu-throughput/build-pair.sh "$QEMU_REPO" "$FRESH_BUILD_DIR"
python3 capstone/runtime/virtual/tests/qemu-throughput/measure.py \
  --stock "$FRESH_BUILD_DIR/stock-build/qemu-system-riscv64" \
  --fork "$FRESH_BUILD_DIR/fork-build/qemu-system-riscv64" \
  --images "$IMAGES" --firmware "$FIRMWARE" \
  --plain "$PLAIN_GCC" --native "$CAPSTONE_GCC" \
  --input "$GCC_TEST_INPUT" --reference "$GCC_REFERENCE" \
  --host-cpu "$HOST_CPU" --icount 0 --out "$FRESH_RESULT_DIR"
```

`IMAGES` contains `Image` and `rootfs.ext2`. Inputs remain external; no SPEC sources
are vendored. Run again without `--icount 0` to assess instruction-counting mode
separately. All arms within a cohort must have the same setting. The default
order is A/B/C/D/D/C/B/A in fresh guests, with 50 compilations per window and one
untimed warm-up per guest. Run after builds finish, on an otherwise quiet host.
`--host-cpu` pins the entire process tree; record that choice when interpreting
results. This is a small repeated GCC test, not a representative full SPEC mix.

## Timing and acceptance

The staged runner records host monotonic time and QEMU process CPU ticks upon
receiving trusted `HOST_WINDOW_BEGIN` and `HOST_WINDOW_END` serial lines. Boot,
lock wait, stage creation, warm-up and output comparisons lie outside the window.
Compilation includes guest process startup/exit. Compiler diagnostics are written
to guest files so serial output does not dominate timing. Every timed output and
the warm-up must match the official reference exactly. Failed runs are retained.

Serial delivery adds endpoint jitter; Linux CPU accounting has tick resolution.
Use windows of several seconds, inspect individual samples and report CPU time
alongside wall time. Coalesced, incomplete, repeated or misordered markers fail
closed. Timing runs reject profiling plugins. Host profiling, if added, must be
a separate diagnostic run. These measurements neither count guest instructions
nor establish cycles per instruction on hardware.

`test-host-window.py` checks marker parsing and rejection behavior. Existing
staged-runner defaults are retained; `--machine`, `--cpu` and `--host-window`
are explicit measurement controls.

## Preliminary measured result

The [qualification record](qualification.json) contains matched local builds,
two samples per arm/mode, 16 fresh guests, 800 timed compilations and 16 warm-ups.
Every output matches the official reference. Median QEMU process CPU times:

| Arm | `icount=0`, 50 compilations | No icount, 50 compilations |
| --- | ---: | ---: |
| stock-plain | 4.490s | 4.355s |
| fork-plain | 4.825s | 4.910s |
| platform-plain | 2.990s | 2.890s |
| platform-native | 32.570s | 28.265s |

The matched scalar fork penalty is 7.46% with icount and 12.74% without it.
Native/plain on the Capstone platform is 10.89x/9.78x; this combines software
and emulation effects. Removing icount reduces native CPU time by 13.22% here.
It is not yet a qualified replacement for deterministic functional test mode.
The faster platform-plain control has additional CPU/machine differences,
including `x-pcrel=false`; its cause requires a separate control.

No QEMU semantics or test defaults changed. Next profile host helpers separately,
control PC-relative translation explicitly, and qualify any proposed faster
functional-test configuration on timer/preemption and safety cases. Two samples
on one host and a small repeated input are preliminary evidence, not a general
performance characterization. Host frequency and unrelated host activity were
not experimentally controlled; CPU affinity and observed load are recorded.
