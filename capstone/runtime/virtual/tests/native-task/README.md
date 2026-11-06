# Experimental Linux task-return backend

Branch `native/linux-task-integration`. This is a working first implementation,
not completion of the full [acceptance plan](../../../../docs/plans/native-linux-task-integration.md).
The [transition contract](../../../../docs/design/virtual-capstone/native-linux-tasks.md#linux-61-first-backend-transition-contract)
explains context tracking, exec failure and synchronous lifetime rules.

The kernel owns the protected task context and mm namespace. `.dom` exec installs
no scalar trampoline mapping or context FD. Architecture return enters CSRUNV
and loops through Linux return work after each transition. Existing checked-copy,
exchange ABI, confinement and providers remain. No linear ownership instructions
are introduced. The model's 5 ms quantum supplies escape opportunities; this is
not direct hardware timer delivery. Caught handlers, full restart semantics,
clone/exec services, SMP and native audit/seccomp/tracing remain outside this slice.

## Build and activation

Use a disposable Linux 6.1 tree with this branch's
`../../unpinned/linux-6.1-unpinned.patch`, `../../kernel-native-mprotect.patch`
and `../../kernel-native-gcc.patch` already applied, plus its prepared platform
configuration. Matching QEMU is pinned at `dd041bac5ad9c145c3872a50804ca714a26b1c88`.
Do not apply the experimental patch to the measured baseline tree.

```bash
source capstone/tests/capstone-test-env.sh
# KERNEL_BUILD: disposable prepared Linux tree; CROSS_COMPILE: Linux compiler prefix.
bash capstone/runtime/virtual/install-native-task.sh "$KERNEL_BUILD"
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" olddefconfig
make -C "$KERNEL_BUILD" ARCH=riscv CROSS_COMPILE="$CROSS_COMPILE" -j12 Image
```

The installer applies the architecture patch and stages matching runtime sources
as a built-in driver. It enables `RISCV_CAPSTONE_TASK`, disables SMP and selects
generic virtual CPU accounting with forced user context tracking. The Kconfig
option defaults off and execution additionally requires `capstone_vm.native_task=1`.
For an external matching QEMU source checkout, set `CAPSTONE_QEMU_SOURCE` before
staging. Kernel test variants use `make CV_TEST_HOOKS=1 CV_RACE_TEST=1 Image`.
These deliberate fault/race hooks are absent from the ordinary build.

Boot arguments (no insmod):

```
capstone_vm.native_task=1 capstone_vm.dom_binfmt=1
capstone_vm.native_dispatch=1 capstone_vm.checked_copy=1
capstone_vm.native_slot_transport=1 capstone_vm.unpinned=1
capstone_vm.supervisor_slots=1
```

Use the shared `../../run-staged.py`, ordinary OpenSBI, 4 GiB guest memory and
matching QEMU CPU properties:

```
x-capstone-supervisor-slots=1,x-pcrel=false,x-capstone-raw-moves=true,x-capstone-test-csrs=false,x-capstone-checked-copy=true,x-capstone-context-transport=true
```

Build these fixtures with `build.sh OUTPUT`, setting `CAPSTONE_NATIVE_CC` to the
copyable SDK compiler and `CROSS_COMPILE` to the Linux compiler prefix. Build
current `../native-integration` probes with its build.sh. Stage `gate.sh`,
`compute.dom`, `task-driver`, `probe.dom`, `blocked.dom`, `entry-driver` and the
external GCC artifacts `gcc-test.dom`, `t1.c`, `reference.s` in one directory.
GCC is SPEC 502.gcc_r test/t1.c, `-O3 -finline-limit=50000`; no suite is vendored.
Require `NATIVE_TASK_EXIT:0`, zero fixture failures and the exact-output marker;
run-staged's completion marker alone is not a passing oracle.

For deliberate admission refusal, use the test kernel, write `1` to
`/sys/module/capstone_vm/parameters/task_refuse_once`, then execute `compute.dom`.
It must die with SIGSEGV; subsequent ordinary qualification must still pass.
The invalid action raises an S-origin exception before domain activation,
exercising the early context-tracking recovery and exception-table fixup.

For the existing race suite, use **32 slots** in both kernel and QEMU; its
injected racer requires an additional context. Apply each row of
`../native-race/scenarios.txt`, run `pipe-fill race.dom` with that row's arguments,
and use the unchanged `../native-race/judge.py` oracle. One-slot blocking/task
transport is tested separately by `entry-driver` and `task-driver`.

## Verified evidence

[Qualification record](qualification.json): 51 ABI/authority/protection checks,
10 entry/blocking checks, compute STOP/CONT/KILL, ptrace refusal, absence of
trampoline/control FD, twelve repeated blocked-task kills, admission-exception
recovery, reference-exact GCC, and all 41 copy-race scenarios (17 negative
controls). Compute fixture reports 0.501139 seconds user time and zero rounded
system time. Scalar Linux tools continue to run. The external-module variant also
builds; shared native-core C90 declaration warnings remain.

The initial stale probe binary expected ENOSYS for a now-confined pathname;
the rebuilt current source passes all 51 checks. The initial one-slot race suite
refused its racer setup; this was not counted as passing. Full register/FP/bounds
transport matrices, all raw-platform negatives, exhaustive leak/failed-exec race
injection and the remaining Linux policy integration gates are still open.

## Matched preliminary GCC comparison

`measure-gcc.py` takes paths to QEMU, Image/System.map/rootfs, firmware, capprof,
profrun, GCC, input and reference; see `--help`. It runs fresh guests in
ioctl/task/task/ioctl order. The only mode change is `capstone_vm.native_task`;
both use the same built-in kernel, config, binary, input and output oracle.

[Four-run record](gcc-comparison.json), mean attempted instructions:

| Entry | Total | Kernel including built-in adapter |
| --- | ---: | ---: |
| Existing ioctl | 50,928,947.5 | 26,954,892 |
| Architecture task return | 51,037,165.5 | 27,063,220.5 |

Task return is **0.2125% more total work**, about **0.4019% more kernel work**
in this small test. Other work is effectively unchanged. This first version
returns through Linux work checks after every transition and accounts domain
execution as userspace; it does not yet optimize/batch those transitions.
The result supports the architectural change, not a performance win. Do not
compare these absolute counts directly with older 54M runs: this kernel has
changed SMP/accounting configuration and builds the adapter in-tree.

Raw logs, profiles, kernel images/configuration and Argon review are archived at
`$CAPSTONE_TMP_ROOT/native-task/native-task-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/native-task-evidence.tar.gz`).
SHA-256: `b8286ed6a5252441d0589fa1a9631fc20b468d508a7e064a797cbd19047008fd`.
Astra xhigh reviewed entry/context tracking, task lifetime and ptrace/exec races;
Argon independently reviewed the patch. Review findings were checked against
Linux 6.1 source; neither review substitutes for the runtime gates.


## Syscall path and direct-register measurement

[Full factorial record](syscall-comparison.json): 128 valid profile windows across
eight fresh guests, two observations per entry/ABI/case/fast-path setting.
All return-count checks and output-byte checks pass, with no kernel diagnostics.
`build-syscalls.sh OUTPUT` builds the existing budget fixture for ordinary RV64,
Capstone exchange, forced scalar exchange and direct register calls.
`measure-syscalls.py --help` lists external artifacts; run with `--fast 1` and
`--fast 0` to separate specialised dispatch from the direct argument ABI.
The latter still changes parsing/copy/provider paths, so it is not a pure cost
measurement of register moves alone.

Mean system-wide attempted instructions **per call**, new task backend:

| Operation | Ordinary Linux | Exchange | Registers, generic dispatcher | Registers, specialised paths |
| --- | ---: | ---: | ---: | ---: |
| getpid, forced kernel request | 601 | 10,715 | 5,264 | 1,345 |
| read, 1 byte | 1,697 | 15,908 | 9,603 | 4,205 |
| write, 1 byte | 2,667 | 16,929 | 10,560 | 5,194 |
| read, 64 KiB | 39,996 | 215,131 | 86,462 | 69,969 |
| write, 64 KiB | 80,050 | 293,606 | 126,287 | 113,427 |

The existing direct-register specialised path saves 73.56% for 1-byte read,
69.32% for 1-byte write, 67.48% for 64 KiB read and 61.37% for 64 KiB write
against exchange on this backend. Kernel work for 1-byte read falls from
13,845 to 4,175 attempts (about 70%); most of the reduction is kernel work,
with application marshalling also disappearing. Checked copies and bounds/
permission enforcement remain. This is a probe ABI comparison; GCC/musl has
**not** been switched wholesale to that ABI.

Important scalar control: ordinary Capstone musl `getpid()` uses launch metadata,
measuring 255 instructions and **zero kernel instructions** here. Calling
`syscall(SYS_getpid)` would still meet that SDK shortcut. The forced-exchange
probe calls `__capstone_delegate_ints` to bypass the shortcut deliberately.
The 10,715-to-1,345 comparison is therefore a transport comparison, not an
argument to replace cached getpid: preserving that cache is cheaper still.

The task-entry change by itself remains a cost: the direct getpid path takes
609 attempts through the old ioctl service loop versus 1,345 through task return
with userspace accounting. The cheap repeated service loop need not exit its
ioctl on every request. These are separate effects: direct argument dispatch
helps substantially; changing ownership/return integration alone does not.

Small cases use 100 calls per window; 64 KiB cases use one call. Buffer pages
are touched before profiling, stdin is a prefilled pipe, writes target a tmp
file, and setup/diagnostic output are outside the window. Counts include loop
and wrappers, plus system-wide interrupt work. Only two repeats per cell were
run; they are preliminary instruction counts, not latency/cycle estimates.
The ordinary static RV64 reference uses the same patched kernel, not stock
Linux with a separately matched libc/compiler. Preserve those distinctions
when using the table to choose the next optimisation.

Syscall logs, profiles, input hashes and probes:
`$CAPSTONE_TMP_ROOT/native-task/syscall-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/syscall-evidence.tar.gz`), SHA-256
`91b3e92a7fb0eeec926ba9c04fadba93cb34d3e432cf87211565b315e18ac00f`. The kernel/model artifacts are the same as the task GCC record.

## Opt-in musl small-I/O migration

`CAPSTONE_MUSL_NATIVE_IO=ON` now routes `read`, `write`, `pread64` and `pwrite64`
through the direct-register ABI **only for original requests of 1–65536 bytes**.
Selection happens once before effects. A direct short count/error is final:
no refill, fallback or replay. Zero and oversized original requests, including
small trailing chunks, retain exchange semantics. Checked copies, bounce buffers,
buffer capability authority, identity/time caches and the existing completion
lifecycle remain. This is still a limited prototype, not all-native libc.

The SDK option defaults off and is rejected for physical applications. Build
both comparison SDKs from this source with identical options and libc archive,
changing only this option. The SDK manifest reports `native_io`. Reinstall this
branch's built-in adapter and boot with `capstone_vm.native_io=1` in addition to
the task flags above. It requires `native_task=1` and advertises launch wire v3:
a 16-byte `{abi=1, features=1, max_io=65536, reserved=0}` extension after the
88-byte header. The application image stamp remains v2. New SDKs accept launch
v2 without direct routing; unsupported v3 feature contracts disable routing.
No speculative ecall probes are used. **The global v3 boot option requires rebuilt
applications**; old startup decoders reject v3 and can fail after exec's point
of no return. Keep it off for mixed old/new SDK deployments.

Integration exposed an ABI detail: scalar operations in the pinned model write
the low 64 bits but can leave stale upper payload. A computed `strlen` count
therefore failed strict native shape checks despite constant-count probes passing.
The libc bridge now canonicalizes actual scalar fd/count/offset values in an
explicitly cleared register. LCC type checks preserve malformed capability
arguments for rejection. The assembly stub also clears the number and selector
registers. Kernel upper-word, raw-tag and buffer authority checks are unchanged.
Tagged fd/offset regressions pass; this fixture does not exercise a tagged count
with an admitted low cursor. Moving its bounded capability to such a cursor
clears the tag under compressed-bounds rules, so that is not a valid negative test.

[Qualification](musl-qualification.json) covers v3 specialised/generic dispatch
and v2 fallback. Each includes file/boundary/dirty-scalar tests, short pipe/EAGAIN,
51 authority checks, 20 syscall-coverage checks, blocking and task lifecycle.
Both 41-scenario pipe and regular-file race suites pass the unchanged oracle,
including all 17 deliberately broken controls. Regular-file short results remain
short; libc does not refill after the injected copy fault. Host ASan/UBSan launch
tests cover extension/table truncation, offset boundaries and unknown features.
Astra xhigh and Gemini Argon reviewed the implementation and scalar correction;
findings were checked against pinned model code and generated assembly.

Reproduce fixtures and gates (set external artifact/compiler paths):

```bash
source capstone/tests/capstone-test-env.sh
# CAPSTONE_EXCHANGE_CC and CAPSTONE_NATIVE_CC: matching SDK capstone-cc wrappers.
# CROSS_COMPILE: ordinary Linux compiler prefix.
bash capstone/runtime/virtual/tests/native-task/build-musl-io.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/run-musl-io.py io \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --probes "$PROBES" --out "$OUT"
# Repeat into fresh output directories with --fast 0 and with --feature 0.
# Modes race and race-file require the CV_RACE_TEST/CV_TEST_HOOKS kernel.
# The runner selects one slot for IO/lifecycle, 32 for the injected racers.
```

### GCC result and route attribution

Relink the **same GCC object files and final link arguments** with each SDK.
`measure-musl-gcc.py --help` lists the required external artifacts; the two GCC
arguments are `--gcc` (exchange) and `--gcc-direct`. It runs four fresh guests in
exchange/direct/direct/exchange order with the same task kernel and feature
announcement. Profiling has route tracing disabled. All four exact-output checks
pass. [Measured record](musl-gcc-comparison.json), mean attempted instructions:

| Musl routing | Total | Kernel including adapter |
| --- | ---: | ---: |
| Exchange | 51,037,402.5 | 27,063,328 |
| Small direct I/O | 51,058,377 | 27,085,867.5 |

That is **+0.0411% total, +0.0833% kernel**: no observed GCC speedup in this small
preliminary sample. Userspace/firmware other work decreases 1,565 attempts. Do not
interpret interrupt-sensitive totals or two repetitions as a latency result.

Repeat the command with `--routes` and a new output directory for two separate
**diagnostic** guests. [Route record](musl-gcc-routes.json): each arm records 31
service completions; only one 69-byte `read` moves to registers. Three `writev`
requests, pathname/status/limit/signal and other recorded services stay exchange.
The trace is not an exhaustive accounting of allocator/mapping/lifetime work;
`count` is the transport's third argument, not always bytes (for writev it is the
vector count). The direct-I/O probe savings therefore do not predict a speedup
for this GCC test. Further work should attribute the remaining kernel work by
service and phase before expanding the ABI solely on these microbenchmarks.

Raw final logs, profiles, binaries, kernel images, input hashes, source diff and
Argon reviews are archived at
`$CAPSTONE_TMP_ROOT/native-task/musl-io-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/musl-io-evidence.tar.gz`), SHA-256
`08df4188697990d7f14556c04e1ba61bf606267aeb9be8f6014ff186beccdac1`.
The root filesystem and firmware are external artifacts identified by hashes;
the archive does not vendor SPEC sources or duplicate the base root filesystem.

### Short SPEC test-input comparison

[Matched four-program record](musl-spec-short-comparison.json) extends the same
SDK comparison to ImageMagick, lbm and nab, selected from workloads previously
estimated to take at most five minutes. Each new benchmark has two complete
fresh-guest runs per variant, with order reversed for the second pair. GCC reuses
its unchanged four-run record above. All three new programs were relinked from
identical objects with the two existing SDKs; no runtime change was made.

| Test input | Exchange total attempts | Direct total attempts | Total change | Kernel change |
| --- | ---: | ---: | ---: | ---: |
| GCC | 51,037,402.5 | 51,058,377 | +0.0411% | +0.0833% |
| ImageMagick | 112,356,920 | 112,314,757.5 | −0.0375% | −0.2332% |
| lbm | 6,657,715,796 | 6,651,726,854.5 | −0.0900% | −0.6355% |
| nab | 7,242,697,263.5 | 7,242,441,522.5 | −0.0035% | −1.3880% |

Values are means, with kernel including the built-in adapter. All twelve new
runs pass the original SPEC comparators: lbm absolute tolerance `1e-7`, nab
relative tolerance `0.01`/`skipreltol=2`, and ImageMagick's image validator plus
its `0.01` relative-tolerance output comparison. Image validation is outside the
profile window. GCC retains its exact-output oracle. All eight staged data files
for the three new programs match the original SPEC input/compare files by hash.

The initial nab staging omitted its nested input directory and failed before
completing the benchmark; those attempts are retained in the raw archive and
excluded from the table. Corrected runs copy the complete directory tree.
Elapsed host durations include shared-runner lock waits and must not be used
as benchmark timings. These remain preliminary instruction measurements, not
hardware speedups or reportable SPEC scores. The largest observed reduction in
total work is only 0.09%; this slice works but does not materially reduce total
work for these test inputs.

Raw profiles, serial logs, output validation, input audits, object/binary hashes
and the linking/running/validation scripts are in
`$CAPSTONE_TMP_ROOT/native-task/spec-short-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/spec-short-evidence.tar.gz`), SHA-256
`dacb6be869ac79b416a8557e37faa3b6dfc9ad5ca39f01015677f289f53f8be6`.
Scripts record the external artifact paths used for this run; the QEMU, kernel,
SDK and GCC artifacts are the same as the preceding musl migration record.
No benchmark sources or data files are committed to the repository.

### Syscall-heavy workload through real musl

`musl-copy.c` copies the same resident 1 MiB file using ordinary musl `read` and
`write`, with 64-byte, 4-KiB and 64-KiB buffers. One application object is linked
against the exchange and direct-I/O SDKs. File creation/population, buffer setup,
full output comparison, final offset/length checks and diagnostics are outside
the measured loop. Both files are populated before measurement, so this measures
CPU work in the cached I/O path, not disk latency. Loop/return-count checks remain
inside the window; capability checking and kernel copy rules are unchanged.

[Four-guest record](musl-copy-comparison.json): exchange/direct/direct/exchange,
reversed case order in the second pair, twelve valid windows, every return count
and full copied output verified. Means of two observations per cell:

| Buffer | read/write calls | Exchange total attempts | Direct total attempts | Total reduction | Kernel reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64 bytes | 32,768 | 555,982,127 | 189,516,342 | 65.91% | 64.96% |
| 4 KiB | 512 | 12,367,353.5 | 4,813,633.5 | 61.08% | 53.62% |
| 64 KiB | 32 | 4,989,767.5 | 2,670,841 | 46.47% | 20.85% |

This confirms that the libc integration retains a substantial saving when the
workload spends its time on eligible I/O. The 64-byte loop uses about 2.93 times
fewer instruction attempts, not a measured 2.93x hardware speedup. Buffering also
matters: either 64-KiB variant performs much less total work than either 64-byte
variant. These results explain why the limited call mix in the short SPEC tests
does not show a comparable whole-program gain.

Reproduction, with the same SDK and external platform artifacts as above:

```bash
source capstone/tests/capstone-test-env.sh
# CAPSTONE_EXCHANGE_CC / CAPSTONE_NATIVE_CC name the matching SDK wrappers.
bash capstone/runtime/virtual/tests/native-task/build-musl-copy.sh "$PROBES"
python3 capstone/runtime/virtual/tests/native-task/measure-musl-copy.py \
  --qemu "$QEMU" --images "$IMAGES" --firmware "$FIRMWARE" \
  --plugin "$CAPPROF" --probes "$PROBES" --out "$OUT"
```

Raw profiles, serial logs, binaries, common object and source scripts:
`$CAPSTONE_TMP_ROOT/native-task/musl-copy-evidence.tar.gz`
(mirror `p12:/var/tmp/capstone-native-task/musl-copy-evidence.tar.gz`), SHA-256
`8b502739784acb0c673d413673782c5129dc4c6194039d18392a23fa260f3e32`.
Platform artifacts are identified by hashes and are unchanged from the musl
migration record. No wall-clock or hardware-latency claim is made.
