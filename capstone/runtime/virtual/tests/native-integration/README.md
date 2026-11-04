# Native dispatcher integration experiment

Default off. `native_dispatch=1` selects `CV_STEP_NATIVE` for the native loader.
It requires `dom_binfmt=1`, `checked_copy=1`, `native_slot_transport=1`, a matching
model, and the kernel protection wrapper in `kernel-native-mprotect.patch`.
Use the existing unpinned kernel patches as well when `unpinned=1` is selected.
The trampoline owns no grants. Faults and syscall services run inside the ioctl;
signals, exit and the quantum still return through Linux.

The S dispatcher is shared by the executable double and the kernel adapter.
Its actual copy provider is U's checked helper. Direct buffers use the paused
register slot and escape epoch. Legacy SDK exchange buffers use the actual
exchange capability plus a validated offset. Metadata reads and replies also
use the checked helper. Scalar addresses never substitute for buffer authority.
Slot import must restore the checked-copy epoch before the next check.

Implemented Linux calls: read, write, pread64, pwrite64, close, lseek, getpid,
getppid, gettid, uid/gid variants, realtime/monotonic clock_gettime and sched_yield.
They use private snapshots and held file references. Unsupported calls return
ENOSYS, including vectors, paths, new descriptors, sockets and process creation.
Inherited seccomp is refused at exec. Restartable interruption returns EINTR;
authenticated signals and transparent restart are not implemented.

The protection wrapper invokes Linux's existing mprotect implementation with
its normal permission and security checks. It preserves the predecessor's
page-range protection behavior inside a dynamic arena, without changing grant
identity. **This is not qualification of the frozen whole-grant compatibility
requirement.** Reconciliation of partial protection remains a landing blocker.

## Reproduction

Use a private lane directory selected by `LANE_ROOT`. It contains `src/` with
the runtime, `capstone/tests/capstone-test-env.sh`, and `llvm/CMakeLists.txt`;
`platform/` with the matching model, kernel, firmware, rootfs, SSH server and
adapter; and `tmp/`. The build script uses the existing cap4 SDK and hosted-ab
cross compiler. All build and run output belongs under that directory.

1. Build the kernel with the existing unpinned patches and the protection
   wrapper. Build the adapter against that exact kernel. Put its module in
   `platform/adapter/module/` and refresh the platform hashes.
2. Run `build.sh` on the build host with sufficient disk space. It builds the
   guest probe, plain pipe supplier and three budget binaries.
3. Through the host gate queue, run `bash run-guest.sh`. A fresh output directory
   is required; `NATIVE_OUT` selects it. The driver disables binfmt_misc and
   requires all 24 checks, no forbidden guest diagnostics, and successful unload.
4. Build a separate module with `KCFLAGS=-DCV_NATIVE_UNSAFE_BOUNDS`. Select it
   with `NATIVE_MODULE`, a fresh `NATIVE_OUT`, and `EXPECT_BOUNDS_FAILURE=1`.
   The same probe must fail tight-bound refusal and corrupt the adjacent
   sentinel, while still refusing forged and revoked pointers. This unsafe
   build is only a negative control; never replace the safe platform module.
5. Through the gate queue, run `bash run-budget.sh`, then
   `python3 summarize.py OUTPUT > OUTPUT/attempts.tsv`. `BUDGET_OUT` selects
   a fresh directory. The 27 windows cover three repetitions of plain,
   launcher and native 1-byte reads, 64 KiB reads and getpid. The pipe is fully
   loaded before exec; buffer pages are touched before each window. All guest
   attempts inside the window, including the loop, are retained.

The probe tests sequential object revocation and reuse, kernel tag clearing,
multi-page output and authority refusal. It does **not** implement forced
same-address-space thread switches, mapping reuse or COW-retry revocation.
Those controls, both-host nine-gate parity, hosted 93/93 and census qualification
remain mandatory before landing. The new syscall-entry test hook is point 5;
the existing fault hook remains point 4. No gate oracle has been changed.
