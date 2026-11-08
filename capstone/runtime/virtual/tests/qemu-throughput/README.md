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
closed. Timing runs reject profiling plugins. Host profiling must be
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

No QEMU semantics or test defaults changed. Next optimize only measured helper costs,
control PC-relative translation explicitly, and qualify any proposed faster
functional-test configuration on timer/preemption and safety cases. Two samples
on one host and a small repeated input are preliminary evidence, not a general
performance characterization. Host frequency and unrelated host activity were
not experimentally controlled; CPU affinity and observed load are recorded.

## Host CPU profile

[Profile qualification](profile-qualification.json) records three substantial
profiles: 50 native compilations with icount, 50 without, and 500 plain
compilations with icount. All outputs and warm-ups match; all three profiles
have zero lost samples (6,488 / 5,679 / 5,828 samples). An additional short plain
control passes. Failed pilots are retained and excluded.

Disjoint shares of sampled **host userspace cycles**, rounded:

| Self-cost group | Native, icount0 | Native, no icount | Plain, icount0 |
| --- | ---: | ---: | ---: |
| Translated guest code | 27.4% | 18.4% | 41.7% |
| Capability representation / moves | 12.8% | 14.2% | 0.0% |
| Tag metadata | 11.2% | 12.2% | 1.0% |
| Capability load/store helpers | 8.1% | 7.8% | 0.0% |
| Translation-block dispatch | 7.7% | 9.7% | 33.9% |
| Lifetime/access validation | 5.6% | 6.1% | 0.2% |
| Dirty-bitmap bookkeeping | 5.0% | 5.5% | 0.3% |
| Other QEMU and host libraries | 22.2% | 26.1% | 22.9% |

This is distributed emulator work. The largest native self symbol is bounds
`decode.isra.0` at 3.46%, followed by `cap_compress` (3.03%) and
`cap_mem_map_physical_lock` (2.97%). Inclusive store-helper cost is 16.36%,
but overlaps the rows above and must not be added to them. DWARF unwinding
across JIT code can be incomplete. Plain percentages have a much smaller
per-compilation denominator; compare shares cautiously.

Source inspection of the pinned fork identifies these candidates, **not yet
implemented or measured**:

1. `target/riscv/cap_mem_map.c`: range removal repeats per-granule metadata
   lookup after already finding the page. Clear bitmap ranges per page/word
   and reuse the lookup. Preserve locking, every affected granule, conservative
   tag-presence metadata and atomic publication of bytes/tags. A one-entry
   page cache already exists. Even one vCPU shares RAM with device threads,
   so simply removing the lock is unsafe.
2. `target/riscv/cap_compress.c`: cursor representability compresses, unpacks
   and decodes at both cursors. Investigate reusing derived representation
   without changing exact bounds, invalidation or representability semantics.
3. Investigate block dispatch and dirty tracking independently. Lifetime-table
   page watching and fresh authority reads are correctness mechanisms; the
   profile does not justify deleting them.

No measured share is a promised speedup. These samples explain host emulation
cost, not RTL cycles, and do not separate guest Linux from guest userspace.

### Reproduce profiling

Use the same `measure.py` arguments as above, adding `--host-perf /path/to/perf`
and `--arm platform-native`. Keep `--iterations 50` for native; use
`--arm platform-plain --iterations 500` for the longer scalar control.
Use a fresh result directory for every cohort. Omit `--icount 0` for the
separate no-icount profile. Then summarize each VM:

```sh
python3 capstone/runtime/virtual/tests/qemu-throughput/summarize-profile.py \
  --perf /path/to/perf --vm "$FRESH_RESULT_DIR/00-platform-native/vm"
```

The runner enables QEMU `-perfmap`, attaches perf to QEMU threads at the begin
marker, stops it at the end marker, and saves raw data, commands and the JIT
map. Attachment and stopping have endpoint latency. Sampling uses `cycles:u`,
199 Hz, DWARF stacks of 8192 bytes and a 128-page ring. It requires host perf
permission; it does not modify system permissions. The analyzer refuses to
replace a conflicting `/tmp/perf-PID.map`; use an isolated `/tmp` for archived
runs with PID collisions. Keep the exact symbol-bearing QEMU binary available.
Symbol classifications are specific to this pinned binary and need review
for a different build, especially compiler-generated names.

These runs used a scratch-extracted Ubuntu perf 6.8 package on host Linux
6.18.7. The profiler inherited CPU 2 affinity and significantly perturbed wall
time. Every result is marked `profiled`; use only the earlier **unprofiled**
cohort for throughput claims. Four profiler lifecycle tests and five marker
tests pass. Raw profiles, logs, package files, JIT maps and commands are in the
archive identified by the qualification record; no benchmark sources are
committed. Defaults and QEMU semantics remain unchanged.

## QOM debugging control and tag-clear census

The [follow-up qualification](qom-census-qualification.json) changes the next
optimization priority. A new build differs in exactly one recorded Meson option:
`qom_cast_debug=false` (`--disable-qom-cast-debug`). It retains `-O2`, ordinary
assertions and capability checks. Unprofiled baseline/candidate/candidate/baseline
runs use CPU 2, icount0, fresh guests and unchanged guest artifacts:

| Workload window | Baseline mean QEMU CPU | Candidate mean QEMU CPU |
| --- | ---: | ---: |
| 20 native GCC compilations | 13.030s | 13.030s |
| 100 plain GCC compilations on Capstone platform | 5.935s | 5.850s |

All 480 timed outputs and eight warm-ups match. The plain control's mean CPU
reduction is 1.43%; native shows no measurable gain. With two observations per
build and uncontrolled host frequency, this is a preliminary control, not a
reason to change the default. Existing profile shares are not guaranteed savings.

A separate diagnostic build counts physical tag-clear calls during ten native
GCC compilations, excluding boot and warm-up through a host-controlled shared
byte at the serial markers. All ten outputs and its warm-up match. It observes:

| Granules touched per call | Calls |
| --- | ---: |
| One | 38,711,545 (99.893%) |
| Two | 41,346 (0.107%) |
| Three or more | 0 |

The largest caller is the scalar-store helper (`op_helper.c:4009`, 26,640,141
calls), followed by untagged capability stores (`:3512`, 7,458,880 calls).
Of the one-granule calls, 36,458,664 find existing page metadata and 2,252,881
do not. Finding metadata does not imply that the bit was set; this census counts
clear requests, not tag transitions. It also does not measure lock contention
or cycles per call. Other workloads may have different size distributions.

**Next optimize the one-granule path and reuse its already-found page entry.**
Whole-page batching is not the first priority for this measured GCC workload.
Keep the physical-map lock and clear every overlapping granule, including partial
writes; validate tag writers, stale/live controls and exact GCC output before
unprofiled paired timings. No such source optimization has been implemented yet.
Bounds-calculation simplification remains a separate candidate.

The archive contains the comparison driver and commands, both candidate binaries,
final diagnostic patch, diagnostic runner, counters, build options and logs.
The census uses fixed source-site labels and counters under the existing tag lock;
it does not add guest instructions or change the ISA. Marker-observation latency
makes the endpoints approximate. Its results are explicitly marked diagnostic
and must not enter timing comparisons. The first census pilot used JIT return
addresses and exhausted its bounded caller table; it is retained and excluded.
Reproduce by applying `tag-census-final.patch` to the pinned fork and using the
archived diagnostic runner; the earlier `instrument.py`/pilot patch is historical.
No production runner, emulator pin, test default or native RTL handoff changed.

## Page-lookup reuse candidate

Implemented in the QEMU repository on
[`perf/tag-clear-lookup`](https://github.com/project-starch/capstone-qemu/tree/perf/tag-clear-lookup),
commit `1d6bf3548c`. The superproject's default QEMU pin remains unchanged.
[Qualification and complete samples](tag-lookup-qualification.json).

`cap_mem_map_physical_remove_range_locked` now keeps the page entry it found and
clears the overlapping granules directly. The existing page split, physical-map
lock, tag publication and conservative page summary are unchanged. This is
lookup reuse, not bulk bitmap clearing. Machine-code inspection confirms one
`find_entry` call site instead of two in this function.

The candidate uses exactly the baseline build options, including QOM cast
debugging. Fresh baseline/candidate/candidate/baseline guests on CPU 2, icount0:

| Window | Baseline mean CPU | Candidate mean CPU | Difference |
| --- | ---: | ---: | ---: |
| 20 native GCC compilations | 13.005s | 12.995s | -0.077% |
| 100 plain GCC compilations | 5.945s | 5.935s | -0.168% |

All 480 outputs and eight warm-ups match. These tiny differences are unresolved
at two samples per build and this host's noise; **no useful speedup is established**.
Keep the candidate available for review rather than advancing the default pin.
Next investigate local bounds/representability calculation reuse separately.

Correctness checks pass:

- Three tag-map unit tests: concurrent publication/read/write/mapping operations,
  sticky summaries, and 962 independently checked byte intervals. Range cases
  include partial granules, crossing pages, absent metadata, zero length and
  the last byte of the address space. Surviving tags retain exact bounds.
- 35 checked-copy instruction cases; 15 removed encodings at each of M/S/U,
  rejected linear mint followed by successful copyable mint; 12 raw-transport
  privilege/feature configurations with tag-copy and scalar-clear checks.
- Existing NONLIN virtio mapping-race arms 24/25, including byte and tag oracles.

The old full tag-writer suite is **not** a passing gate: most arms mint linear
capabilities and exit with cause 29 on this copyable-only branch. Baseline arms
0/2 reproduce the failure unchanged. The multi-hart arm is unsupported. Keep
those logs and the two passing NONLIN device cases separate; this qualification
is not a full writer-suite or full-SPEC result. No ISA or RTL changes are needed.

To reproduce, check out the candidate QEMU commit and build with the baseline
configure options in `build-pair.sh` (do not disable QOM cast debugging). Run
`tests/unit/test-capstone-tag-map` from its build tree, the current copyable-only
and checked-copy gates, then `measure.py` against baseline/candidate binaries.
The evidence archive includes the driver, exact commands/options, candidate
patch and binary, disassembly, successful checks and excluded failures.
