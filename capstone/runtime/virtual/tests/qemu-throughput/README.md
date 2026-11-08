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

## Cursor-window comparison candidate

QEMU branch
[`perf/cursor-representability`](https://github.com/project-starch/capstone-qemu/tree/perf/cursor-representability),
commit `ecd71075da`, is based directly on the original fork pin. It does not
include the tag-lookup candidate. [Qualification](cursor-qualification.json).

The old predicate compresses the source and fully decodes it at both old and
new cursors, then compares bounds. Both decoded bounds share the same
cursor-dependent window term. The new predicate compares that term directly,
preserving compression rounding, old-cursor unpacking, all 64 new-cursor bits,
unsigned wraparound and sealed exclusions. It neither removes the security
predicate nor caches authority. The algebra is documented beside the code.
Machine code no longer calls `decode` twice from the predicate.

All configure options match the baseline. Fresh guests use CPU 2, icount0,
and baseline/candidate/candidate/baseline order in each cohort:

| Window | Baseline mean QEMU CPU | Candidate mean QEMU CPU | CPU reduction |
| --- | ---: | ---: | ---: |
| 20 native GCC compilations | 12.945s | 12.615s | 2.55% |
| 50 native GCC compilations (confirmation) | 32.685s | 31.565s | 3.43% |
| 100 plain GCC compilations | 5.915s | 5.865s | 0.85% |

All 680 timed outputs and 12 warm-ups match the official reference. The repeated
native result supports a modest host CPU reduction for this input; it is not a
full-SPEC or RTL speedup. Host frequency and unrelated load are uncontrolled;
individual samples and wall times remain in the qualification record.

Correctness checks:

- 1,306,912 comparisons against the original full two-decode predicate: zero
  mismatches, both keep/clear outcomes, 30/55-bit IDs, rounded/inexact sources,
  cursor-bit edges, sealed forms and arbitrary 64-bit cursors. Undefined-behavior
  sanitization passes. Always-keep and new-cursor-canonicalization mutations
  are both detected.
- Existing codec: 6,014,628 checks pass. Separately pinned RTL bounds model:
  10,640 window-edge cases agree; its generator skips 25 inexact sources,
  which are separately covered by the larger differential test.
- New copyable CINCOFFSET edge fixture: all four compact/exact and 30/55-bit
  combinations pass on both baseline and candidate. The fixture checks tags,
  scalar cursor after tag loss and source preservation without legacy ops.
- All 35 checked-copy instruction cases pass on the candidate.

The QEMU commit includes `tests/capstone-codec/run-representable.sh` and
`tests/copyable-only/cursor-window.sh`. The archived commands, candidate binary,
patch, disassembly, logs and RTL model hash reproduce the experiment. The
superproject default pin and native RTL handoff remain unchanged pending broader
qualification. This candidate is worth carrying forward; the earlier tag-lookup
and QOM-debug experiments do not have a demonstrated native benefit.


## Bounds-only refinement and integration candidate

QEMU [`perf/qemu-throughput`](https://github.com/project-starch/capstone-qemu/tree/perf/qemu-throughput)
at `80eaf2c89b` combines the cursor-window predicate, bounds-only encoding
(`925418d22c`), and additional in-place CINCOFFSET/SCC edge tests. The encoder
is shared with full capability compression, preserving its rounding. The
predicate avoids packing metadata it never uses. No ISA or RTL change is needed.
[Qualification and individual samples](bounds-only-qualification.json).

A three-way confirmation uses original/cursor/bounds/bounds/cursor/original
order, 50 native GCC compilations per fresh guest, CPU 2 and icount0:

| Build | Mean QEMU CPU per window |
| --- | ---: |
| Original fork | 32.660s |
| Cursor-window predicate | 31.530s |
| Bounds-only refinement | 31.010s |

The combined change uses **5.05% less host CPU** than the original fork;
the incremental refinement saves 1.65% in this cohort. A separate 20-compilation
comparison agrees in direction (1.27% incremental reduction); its plain control
is unchanged. All 780 timed outputs and 14 warm-ups match. These are host
emulation results for one input, with two samples per variant, not RTL timings.
The earlier differential, sanitizer, codec, RTL-model and instruction gates
also pass on the refinement.

Six short SPEC entries (GCC, Imagick, both specrand entries, LBM and NAB) pass
`qualify-spec.py`: their outputs match hashes from the previous officially
qualified full run, and the guest image validator runs for Imagick. This is
an exact-output regression check, **not a fresh official comparator execution**.
The helper verifies input and guest-platform hashes before execution. Broader
qualification remains necessary before advancing the default QEMU pin.

Removing two additional temporary capability copies did not improve throughput:
20-compilation native means were 12.475s baseline versus 12.545s candidate
(about 0.56% slower). The [experiment](move-copies-qualification.json) remains on
`perf/cursor-move-copies`; its runtime change is excluded from integration.
The extra in-place cursor tests are retained. No security checks were removed.


## Optional LTO build and functional clock modes

With the same source (`80eaf2c89b`), `-O2`, assertions and QOM cast debugging,
link-time optimization further reduces native GCC host CPU. The three-way
50-compilation confirmation gives 32.630s original, 31.105s bounds-only and
30.000s bounds-only with LTO: **8.06% less than original**, or 3.55% incremental.
All 780 timed compilations and 14 warm-ups match. The 20-compilation cohort
agrees in direction. [Samples, build settings and safety checks](lto-qualification.json).

LTO is optional: its observed link step took 17.9s versus 0.5s without LTO.
This can pay off over longer campaigns but adds cost to each emulator rebuild.
The explicit builder preserves the baseline options:

```bash
source capstone/tests/capstone-test-env.sh
capstone/runtime/virtual/tests/qemu-throughput/build-candidate.sh \
  "$QEMU_REPO" 80eaf2c89b "$CAPSTONE_TMP_ROOT/qemu-candidate" --lto
```

Omit `--lto` for a normal build; use a fresh output directory. The builder records
the source revision, configure command/options, build logs and binary hash.
`QEMU_BUILD_JOBS` and `QEMU_LTO_THREADS` control build parallelism only.

`run-musl-io.py`, `run-directory.py`, `run-spec-tests.py` and `qualify-spec.py`
now accept explicit `--no-icount` for functional tests. Their default remains
icount0. The LTO candidate passes I/O, task lifecycle (including compute
stop/resume/kill), directory/dup/timer checks, and both 41-scenario race groups
with their negative controls in **both** modes. Six short SPEC entries also
match the qualified baseline in icount0. Full-suite checks continue separately.
Wall-clock mode does not provide deterministic guest instruction counts.

The fresh LTO profile has zero lost samples; all 100 diagnostic GCC outputs and
one warm-up match. Its largest named self costs include capability loads/stores
and dirty-bitmap updates. LTO changes inlining, so individual symbol shares
cannot be compared directly with the original non-LTO profile. The profile's
wall and CPU times are excluded from clean timing comparisons.


## Additional source candidates: no useful gain established

The [next three-way comparison](additional-candidates.json) keeps the accepted
LTO build as its baseline and tests two changes independently:

| Build | Native GCC mean CPU (20 compilations) | Plain mean CPU (100 compilations) |
| --- | ---: | ---: |
| Accepted LTO baseline | 11.975s | 5.825s |
| Single-bit atomic bitmap path | 12.020s | 5.850s |
| Sufficient local-cursor predicate | 11.930s | 5.890s |

All 720 timed outputs and 12 warm-ups match. The bitmap specialization is
slightly slower; the small-cursor native shift (0.38%) is unresolved and its
plain control moves the other way. **Neither runtime change is integrated.**
QEMU branches `perf/bitmap-single-bit` (`d1289780e4`) and
`perf/cursor-small-step` (`665949035b`) retain the experiments for review.

The bitmap path keeps the same atomic OR and ordering, and passes the bitmap
unit suite. The cursor shortcut proves that equal upper 53 cursor bits imply
equal window keys for every nonnegative encoding exponent. It preserves old
wide-ID canonicalization and all new-cursor bits. Its 1,388,832 differential
checks, sanitizer and three negative controls pass, including an intentionally
too-large shortcut; codec and independent RTL-model checks also pass. Both
candidates pass the actual cursor-edge and 35 checked-copy cases. These
correctness results do not establish a useful speedup.


## Timing a full workload command

`qualify-spec.py` and `native-task/run-spec-tests.py` accept
`--host-window-case INDEX --host-cpu CPU`. The zero-based index selects one
original benchmark command. The shared VM runner measures QEMU process CPU
and host wall time around that command, including its exec/exit. Boot, workload
staging, output collection and image validators are outside the window. There
is no warm-up invocation; use fresh guests and reversed-order repetitions when
comparing binaries. Correctness validation still covers all original commands.

Timing cannot be combined with `--live-output`; that observer changes the work
being measured. Keep `--no-icount` explicit and compare like clock modes.
The result is saved under `execution.host_window`, with affinity, case index
and scope; these are host emulator measurements, not guest or hardware cycles.

The [tool qualification](workload-window-qualification.json) records 29 host
regressions and a real GCC invocation with exact qualified output, CPU affinity
and `--no-icount`. Its single-run duration is not a performance comparison.

## Host compiler and clock-mode controls

The [compiler qualification](compiler-build-qualification.json) compares matched
O2/LTO, O3/LTO, Clang/LTO and host-tuned builds. Clang 18 with LLD is the best of
these controls; host tuning did not help. A separate reversed-order confirmation
uses 50 GCC compilations per fresh guest (one untimed warm-up):

| Build / mode | Mean host CPU seconds | Reduction from original |
| --- | ---: | ---: |
| Original fork, icount0 | 32.640 | — |
| Bounds/cursor changes, GCC O2/LTO, icount0 | 29.895 | 8.41% |
| Same changes, Clang O2/LTO, icount0 | 28.620 | 12.32% |
| Clang O2/LTO, explicit no-icount | 23.920 | 26.72% |

All 1,360 timed compilations and 24 warm-ups across the matrix and confirmation
match. Clang also passes the cursor/checked-copy checks, codec differential and
sanitizer checks, and all eight task/timer/race groups in both clock modes.
Clang requires QEMU `2bc6e19ae9`: braces scope an existing cleanup lock guard
inside its switch arm. GCC variants use `80eaf2c89b`. Assertions and QOM cast
debugging remain enabled. Full SPEC qualification is separate and still pending.
LTO increases link time; these remain explicit build choices.

```sh
CC=clang CXX=clang++ LDFLAGS=-fuse-ld=lld \
  capstone/runtime/virtual/tests/qemu-throughput/build-candidate.sh \
  "$QEMU_REPO" 2bc6e19ae9 "$FRESH_BUILD_DIR" --lto
```

The [scalar PC-relative control](pcrel-qualification.json) rules out `x-pcrel`
as the explanation for platform-plain's speed: identical `virt` runs take 9.660s
with it and 9.680s without it; platform-plain takes 5.935s. All 600 timed outputs
and six warm-ups match. `measure.py --pcrel true|false` requires explicit scalar
arms. The separate configuration controls below resolve that part of the gap.
These results measure host emulator cost, not hardware cycles.

## Host configuration and two further source controls

The [configuration qualification](host-configuration-qualification.json) isolates
the scalar CPU, machine and kernel flags. On the original fork, 100 plain GCC
compilations take 9.715s with the common CPU settings and 5.905s with only
`x-capstone-u-mode=true` added. Adding the remaining platform CPU, machine and
kernel settings gives 5.935s, 5.915s and 5.920s respectively. The legacy mode
calls a tag-removal helper after ordinary stores; the U-mode platform uses the
TLB store hook when needed. All 1,000 timed outputs and ten warm-ups match.
This does not mean that capability security makes a program faster: these are
different emulator paths for the same scalar binary. A separate stock/fork
profile points to the fork's existing page-specific SFENCE.VMA path as another
difference. The [full-flush reversal](sfence-attribution-qualification.json)
confirms it: the page-specific path takes 6.03s versus 9.61s with full flushing
in the forward cohort, and 4.99s versus 7.99s in reverse (37.25% / 37.55% less
CPU). Upstream takes 8.98s / 7.56s. Host speed shifted between the cohorts;
both directions agree. This page-flush optimization already exists in the
native handoff baseline. The diagnostic full-flush build is not a candidate.

`measure.py` exposes `--scalar-cpu common|u-mode|platform`,
`--scalar-machine virt|virt-capstone` and `--scalar-kernel off|native` only for
explicit fork scalar arms. Native measurements retain their existing profile.

The same qualification records two practical build/host controls. Clang ThinLTO
uses 28.945s versus full LTO's 28.530s for 50 native compilations (1.45% more CPU).
Its observed link step is 35.2s versus 72.8s; these are individual build steps,
not a repeated build benchmark. Use `--thin-lto` explicitly when that tradeoff
helps iteration. On this hybrid host, the same full-LTO binary takes 11.375s
on CPU2 versus 16.270s on CPU17 for 20 compilations. Record and hold affinity
fixed; do not mix host-core differences into source-code speedups.

The [runtime-path comparison](runtime-path-candidates-qualification.json)
revisits the local-cursor shortcut under Clang and tests removal of STC's
redundant dirty-page update. The write probe already handles TLB_NOTDIRTY;
tag locks, access checks and full-span code invalidation remain intact.

| Clang/LTO candidate | GCC CPU, 20 compilations | x264 CPU, three-frame excerpt |
| --- | ---: | ---: |
| Existing cursor/bounds baseline | 11.375s | 42.795s |
| STC dirty-page reuse | 11.110s | 42.155s |
| Local-cursor shortcut | 11.285s | 41.560s |

All outputs match. The x264 derivative uses original frames 50–52 and checks
the actual new bitstream, full-sized YUVs and three-frame completion receipt;
it is not an official SPEC command or score. Its baseline samples drift from
42.28s to 43.31s, so the mean differences are preliminary. These two changes
are being combined and measured directly before integration.

The STC store/tag/code fixture passes on baseline and candidate in compact/exact
bounds with both ID widths. Separate dirty-log assertions exercise reset and
reprobe; intentionally broken dirty-probe and code-invalidation builds fail.
Instrumented runs are excluded from timing. Combined task/race and full-SPEC
qualification is still required before advancing the default pin.

## Counting host instructions separately from elapsed time

`measure.py --host-perf /path/to/perf --host-perf-mode stat` counts host
userspace instructions, cycles, branches and branch misses within the existing
guest workload markers. The default perf mode remains stack sampling (`record`).
Both modes set `profiled=true`; their timings must not enter clean comparisons.
Counter mode writes `vm/host-perf-stat.csv` and records its exact command. It
does not generate a JIT map or sample call stacks.

These are QEMU's host instructions, including translated guest kernel work,
not guest ISA instruction counts. Perf attaches after the begin marker, so
small boundary losses are possible; use long, repeated windows. Inspect counter
running coverage before comparing counts. On hybrid hosts the unused PMU can
report `not counted`; require all four events on the active PMU and retain the
raw report. Permission errors or a failed perf process fail the run.
