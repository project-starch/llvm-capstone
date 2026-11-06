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
