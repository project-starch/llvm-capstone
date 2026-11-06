> On `experiment/native-gcc-preliminary`, the bounded GCC coverage extension
> below now completes the unchanged test input. The original qualification
> sections describe `prototype/copyable-only`; they are not a claim that all
> syscalls or native process/signal semantics are implemented.

# Copyable-only kernel syscall prototype

Default off, single hart, root-only experimental `.dom` loader. This is the
narrow Point 2 kernel consumer, not full native Linux task integration (Point 3).
The same Linux task enters the domain through a small U-mode ioctl trampoline;
file operations and checked copies run inside the module. No legacy ownership
instructions are restored. The existing launcher and its Seccomp filter remain.

## Required components

Use this branch's QEMU, copyable SDK, tag-aware unpinned Linux patches and the
additional `../../kernel-native-mprotect.patch` against Linux 6.1. The latter
exports a GPL wrapper around normal Linux `do_mprotect_pkey`, retaining its VMA,
permission and security checks. Rebuild Image and the module against that kernel.
Use ordinary OpenSBI, not a historical patched firmware image.

The explicit module profile is `dom_binfmt=1 native_dispatch=1 checked_copy=1
native_slot_transport=1 unpinned=1`. QEMU must enable checked copy and context
transport; the module validates device-tree ABI 1 / transport 6 before use.
Native activation also requires the mprotect export; incomplete configurations
fail at module load. General transport without checked copy uses version 5.
`native_fast=0` selects the shared generic dispatcher; `native_fast=1` enables
the specialized scalar and flat-buffer paths. Both use the same security checks.

Supported calls: read, write, pread64, pwrite64, close, lseek, getpid/getppid/gettid,
uid/gid variants, realtime/monotonic clock_gettime and sched_yield. Other calls
return ENOSYS. Direct calls require the explicitly selected `native-syscall.S`
stub; the existing SDK is not globally switched to a new ABI.

The private FD broker retains file references and excludes control descriptors.
Exec refuses inherited Seccomp and non-root callers. Authority comes from
immutable escaped capability registers and the request epoch, never a scalar
address. Registered-arena checks precede irreversible I/O. Mapping holds defer
physical unmapping, but logical retirement takes effect immediately. Each copy
chunk rechecks authority under exclusion, including after faults and rescheduling.
Slot eviction saves all lifetime roots and rebinds SATP on restoration. Metadata
protection changes are refused. Page-subrange mprotect within a dynamic arena is
preserved and tested; this is not a whole-grant-only protection profile.

## Reproduction

Source `capstone/tests/capstone-test-env.sh` first. Set `CROSS_COMPILE`,
`CAPSTONE_NATIVE_CC` (the copyable SDK wrapper), and `KERNEL_BUILD` to matching
builds. Keep outputs below `$CAPSTONE_TMP_ROOT`:

```sh
source capstone/tests/capstone-test-env.sh
bash capstone/runtime/virtual/build-adapter.sh "$CAPSTONE_TMP_ROOT/native/adapter"
bash capstone/runtime/virtual/tests/native-integration/build.sh "$CAPSTONE_TMP_ROOT/native/probes"
bash capstone/runtime/virtual/tests/native-race/build.sh "$CAPSTONE_TMP_ROOT/native/races"
```

`run-copyable.py` takes mode `qualify`, `race`, `budget` or `admission`, and
required `--qemu`, `--images` (Image, rootfs.ext2 and optional System.map),
`--firmware`, `--module`, `--probes`, `--out` (a fresh directory). It stages its
own guest disk, selects the model features, and rejects kernel diagnostics,
failed unloads, missing checks and invalid/incomplete profiling windows.

* `qualify --slots 1`: production module and probes; runs both dispatchers,
  51 authority/ABI/mprotect checks plus 10 entry checks each. Two actually
  blocked native processes force transport through one hardware slot.
* `race --fast 1` and separately `race --fast 0`: race module and race probes;
  all 41 frozen scenarios, including 17 deliberately broken controls.
* `budget --launcher PATH --plugin PATH`: production module and probes;
  45 windows, three repetitions with alternating arm order. Add
  `--fast 0 --arms native` for 15 generic comparison windows.
* `admission --features-off`: default module still loads; incomplete/native
  profiles are rejected. Separately use an old kernel without the wrapper and
  `--expect-native-refusal`, with the matching model enabled.

## Measured result

[Machine-readable qualification](../copyable-only/results/native-integration.json)
records exact binary hashes and evidence. Medians of system-wide QEMU attempted
instructions per completed call, not elapsed time or hardware cycles:

| Call | Plain Linux | Launcher | Native fast | Change vs launcher |
|---|---:|---:|---:|---:|
| getpid | 196 | 255 | 602 | +136% |
| read 1 byte | 1,342 | 7,895 | 3,604 | -54% |
| read 64 KiB | 41,033 | 277,558 | 70,902 | -74% |
| write 1 byte | 2,273 | 8,349 | 4,984 | -40% |
| write 64 KiB | 90,227 | 374,462 | 119,873 | -68% |

The launcher already caches getpid. Reads use a filled pipe; writes use a regular
file and verify every byte outside the measured window. Startup, printing and
verification are excluded. Small cases contain 100 calls, large cases one call.
The generic native path remains slower for small calls (read 1 byte: 9,164);
its retained checks qualify the fast-path comparison, not a blanket speedup.

Kernel plus module medians: read 1 byte 3,860 -> 3,576; read 64 KiB
184,861 -> 70,867; write 1 byte 4,447 -> 4,950; write 64 KiB
293,394 -> 119,838. Thus a small write reduces total work while increasing
kernel/module work. Classification uses kernel System.map and actual loaded
module executable-section addresses; remaining PCs include user/firmware work.

## Qualification limits

Qualified native operation uses unpinned payload backing and one hart. The
existing launcher regression passes pinned and unpinned modes. The race hooks
exercise real second domain threads at deterministic points, not scheduler-chosen
SMP interleavings. No claim covers DMA, new-generation identity reuse, full
fork/exec/signals, transparent syscall restart, RTL or a native full SPEC run.
Interrupted restartable operations return EINTR. Bounce buffers remain. This
branch is a reproducible prototype, not a production-security certification.


## GCC SPEC test-input follow-up

The unchanged `502.gcc_r` test invocation (`t1.c -O3 -finline-limit=50000 -o
output.s`) was attempted on the qualified binaries. Both launcher runs completed
and matched the reference byte-for-byte: 61,561,685 and 54,997,297 total QEMU
attempts. Both native modes exited 1 without producing the assembly output;
their aborted windows are explicitly invalid as performance results.

A separate diagnostic module (excluded from measurements) logged refused
`openat`, `newfstatat`, `writev`, `rt_sigprocmask`, SDK `SIGACTION`, `sysinfo` and
`prlimit64`. The existing GCC binary uses the SDK exchange ABI, so this tests
the kernel consumer's compatibility path, not a newly relinked register-ABI GCC.
The observed calls are the current failure frontier, not a complete requirement
inventory. Supporting these calls safely and rerunning the reference validator
is required before quoting any GCC speedup for Point 2. The microbenchmark
reductions above do not establish a GCC improvement.

[Exact outcomes, attribution and hashes](../copyable-only/results/gcc-native-check.json).


## Preliminary native GCC result

The experimental branch adds checked kernel handling for open/stat, sysinfo,
read-only resource limits, signal masks and exchange-ABI vector I/O. Paths use
private checked snapshots with the existing confinement policy. Vectors retain
exchange capability authority, snapshot the offset array once and issue one
Linux vector operation. Interrupted vector operations normalize internal restart
codes to EINTR; failed output copying accounts for the published prefix.
The extra `../../kernel-native-gcc.patch` factors/export kernel-buffer interfaces
from Linux's actual stat/sysinfo/prlimit implementations. Apply it after the
existing unpinned and native mprotect patches and rebuild Image plus module.
Missing optional helper symbols return ENOSYS at use; this is not blanket
module-load qualification of the extension on older kernels.

Two fresh QEMU guests run launcher/native and native/launcher on the same
kernel, module, GCC binary, input and reference. All four assembly outputs match
the reference byte-for-byte. Means of two runs per arm:

| Scope | Launcher | Native | Reduction |
|---|---:|---:|---:|
| Total instruction attempts | 58,461,488 | 54,077,206.5 | 7.499% |
| Kernel + module | 32,541,147 | 29,008,302 | 10.855% |

The kernel part decreases while module work increases. Startup and termination
are included (profrun fork through child exit); guest boot and module loading
are outside the windows. This is a small SPEC test input, not a large-compilation
speedup or hardware timing result. Order effects are substantial: do not compare
only the warm native run to the cold launcher. The generic discovery run also
matched the reference; the final paired record uses native_fast=1, but this GCC
binary still uses the exchange ABI, not the direct-register fast ABI.

Five SDK SIGACTION requests and one IOCTL_BUF request remain explicitly refused
in each native GCC run. GCC tolerates these on this unsignalled test; no fake
success or launcher fallback is used. Therefore this result is a preliminary
engineering signal, not full GCC/Linux syscall compatibility or a security
qualification. Native entry still uses the ioctl trampoline. Caught signals,
transparent restart, positional exchange vectors, arbitrary ioctl commands,
process creation and direct-register nested capabilities remain outside scope.
The refusal counter is not a complete inventory of every possible refusal.

Astra reviewed the extension; the restart/prefix findings were corrected before
the final measurements. Both dispatch modes pass 20 added file/vector/system
checks and the 51 existing authority/ABI/protection checks. The latter's former
openat-unsupported check now requires EACCES for a forbidden `/proc/self/mem`
path; the unknown syscall must still return ENOSYS. This is not a rerun of the
full RT matrix for the new vector operation.

### Reproduce

Source `capstone/tests/capstone-test-env.sh`; build the adapter against the new
kernel and run `build.sh OUTPUT` with the existing copyable SDK. This additionally
builds `gcc-coverage.dom`. Run it and `probe.dom` with native_fast=0 and 1 using
the profile above. The coverage probe needs a writable granted working directory
and creates `coverage.bin`.

`gcc-preliminary.py` requires explicit paths via `--qemu`, `--images` (Image,
rootfs.ext2, System.map), `--firmware`, `--module`, `--launcher`, `--gcc`,
`--profrun`, `--input` (unchanged SPEC test t1.c), `--reference`, `--plugin` and
`--out` (fresh directory). It creates both guests, rejects invalid profiling
windows/nonzero exits/kernel diagnostics, validates exact output, and writes
`summary.json` with per-run totals, attribution and input hashes. Keep benchmark
binaries/inputs outside the repository under `$CAPSTONE_TMP_ROOT`.

[Final measured inputs and qualification](../copyable-only/results/gcc-native-preliminary.json).


## Plain / launcher / native GCC comparison

The follow-up runs the cached ordinary RV64/musl GCC alongside both Capstone
variants on the same Image, QEMU, firmware and measurement plugin. Six fresh
guests cover all six arm permutations, so every arm runs first, second and third
twice. All 18 outputs match the exact same reference. Mean attempted instructions:

| Arm | Total | Relative to plain | Kernel + module |
|---|---:|---:|---:|
| Plain RV64/musl | 32,892,670.3 | 1.000x | 13,510,592.3 |
| Capstone launcher | 58,402,387.2 | 1.776x | 32,481,869.8 |
| Capstone native prototype | 54,029,147.0 | 1.643x | 28,960,167.8 |

Native saves 7.488% versus launcher but still uses 64.259% more instructions than
plain. These are startup-inclusive small-input results, not hardware timings.
The cached plain binary uses the previously recorded RV64 compiler and pristine
musl 1.2.5; compiler revisions/libc configurations are not matched to Capstone.
This is a comparison of the existing stacks, not an isolated ISA-cost experiment.
Plain uses the same patched measurement kernel, not a separately built stock
Linux kernel. The native signal/ioctl and ABI limitations above still apply.

To reproduce, supply `--plain PATH_TO_PLAIN_GCC` to `gcc-preliminary.py` in
addition to the existing arguments. This runs three cyclic orders. Repeat into
a fresh output with `--reverse-rotations` to run the other three orders; combine
the six samples per arm from both `summary.json` files. Input hashes must match
between batches. Benchmark sources/binaries remain outside the repository.

[All runs, ratios, provenance and evidence](../copyable-only/results/gcc-three-arm.json).


## GCC startup / compilation / teardown partition

`gcc-phases.py` uses the same binaries and input with a host-only capprof extension.
No guest instruction, GCC source or input is changed. The first boundary is after
the first instruction of `init_cgraph`, the first call in `compile_file`; the
second is after the first instruction of `statistics_fini`, reached in `finalize`
after the assembly output file is closed. Startup therefore includes GCC option,
backend and language initialization, not only OS loading. Compilation includes
compile_file initialization, translation, output and output-file close. Teardown
includes the remaining GCC finalization, libc exit and Linux/module cleanup.
One instruction at each boundary belongs to the preceding phase.

Two complete cold invocations per arm, each in its own fresh guest, give mean
system-wide instruction attempts:

| Phase | Plain | Launcher | Native prototype |
|---|---:|---:|---:|
| Startup and initialization | 14,845,056.0 | 26,659,188.0 | 24,268,340.5 |
| Compilation and output | 13,186,188.0 | 25,836,295.0 | 24,205,330.0 |
| Finalization and exit | 4,860,032.5 | 9,064,907.5 | 8,718,852.5 |

All six full outputs match the reference. Every run has exactly three phases,
two authenticated boundary entries at one image base, and phase counts summing
to an independently snapshotted outer-window total. A wrong instruction
fingerprint is rejected even though GCC still produces the correct output.
Repeated anchor arrivals from faults are permitted; multiple completed boundary
candidates, missing boundaries or out-of-order boundaries are rejected.

Of the native-minus-plain excess in these cold runs, 54.656% occurs outside the
compilation phase and 45.344% inside it. This does not mean the outside portion
is removable: it includes necessary GCC initialization and process cleanup.
Within compilation, kernel+module work is 12.619M native versus 4.447M plain,
accounting for about 74% of that phase's extra work. Flat PC attribution shows
additional memset, mapping lookup, kernel allocation/free, module page-table
scans and fault resolution. It does not identify the caller responsible for
every allocation or copy. The remaining PC bucket must not be labelled pure ISA
or compiler overhead; compiler/libc differences and firmware remain included.

These all-cold totals should not be substituted for the mixed warm/cold means
from the six-order experiment. The same prototype signal/ioctl limitations
remain. The result motivates inspecting memory-management work during compilation,
rather than assuming that a native task entry alone removes the measured delta.

To reproduce, apply `capprof-phases.patch` to `contrib/plugins/capprof.c` from QEMU
commit `dd041bac5ad9c145c3872a50804ca714a26b1c88` in a scratch copy. Build that file
as a shared plugin with `cc -fPIC -Wall -O2`, glib-2.0 pkg-config CFLAGS, and the
matching QEMU `include/qemu` headers. Supply the resulting `.so` through
`--plugin` to `gcc-phases.py`, with the same required input paths as
`gcc-preliminary.py`, including `--plain`. The default is two runs per arm;
`--arms plain --repetitions 1 --negative-boundary` must fail the boundary oracle.
The profiler's normal start/stop behavior is unchanged unless `phases=1` is set.

[Phase samples, boundaries, checks and evidence](../copyable-only/results/gcc-phases.json).
