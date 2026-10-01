# Trusted-Linux feasibility gate: ordinary guest control

The first question is whether the selected QEMU, firmware, Linux image and
cross-compiler can run a small ordinary Linux process with the OS operations
needed by the protected-process experiment. This gate answers that question
without treating CPU support as process protection.

`probe.c` uses an anonymous two-page `mmap`, faults both pages in, changes a
page's permission with `mprotect`, allocates and frees a 64-byte object,
forks a child, transfers bytes through a pipe into a user buffer, waits for the
child, and unmaps the range. It reports whether `malloc` reused the same
numerical address. It never dereferences the freed pointer. An unprotected C
program cannot test the proposed stale-capability contract by writing such a
dereference: the compiler may remove or transform undefined behavior.

The runner compiles this probe as a static RISC-V Linux binary, puts it on a
scratch ext4 disk, and boots the existing rootfs read-only under the pinned
QEMU binary with `x-capstone-u-mode=true`. It logs in, mounts the second disk,
checks the probe's exact success line and shell exit status, and writes an
optional path-free record with input hashes. The raw serial log stays in
scratch space. `--control-missing-probe` leaves the binary off the disk and
must fail; this catches a runner that mistakes login or the shell prompt for
probe completion. Use the common QEMU lock supplied by the test environment.

From a superproject checkout, with separately built QEMU and Linux images:

```sh
source capstone/tests/capstone-test-env.sh
python3 capstone/tests/trusted-linux-feasibility/run.py \
  --image-dir /path/to/build/images \
  --qemu /path/to/qemu-system-riscv64 \
  --cc /path/to/riscv64-linux-gcc \
  --log "$CAPSTONE_TMP_ROOT/trusted-linux-feasibility/baseline.log" \
  --record "$CAPSTONE_TMP_ROOT/trusted-linux-feasibility/baseline.json"
```

The initial `result.json` records one passing run with the experimental CPU
property enabled. It used the available Buildroot **glibc** toolchain and
guest images, so it is an OS/control baseline, not the project's target musl
application profile. The result reports `protected_process: false` by design.
It is not evidence for a tagged kernel copy, protected `malloc`, preserved
capability registers, preemption of a protected process, or safe retirement.

The next gate is a process that Linux explicitly selects for protection. It
must retain tagged pointer identity across a syscall and a context switch,
obtain authority for an ordinary Linux mapping, and fault a retained old
pointer after `free` and same-address reuse while the fresh pointer succeeds.
The process must also observe ordinary PTE restrictions and receive a
recoverable `EFAULT` for an invalid syscall buffer. A debug-minted capability
in a bare-metal guest cannot close this gate. Compare three ways of preserving
pointer identity across Linux's register saves and copies—explicit tagged
save/restore, automatic trap capture, and disjoint metadata propagation—by
their complete hardware, kernel, libc and compiler costs. No trap mechanism is
selected by this control run.

Before performance results, freeze one matched application workload and the
acceptable limits for throughput, tail latency, memory/metadata use and
hardware area. Run the same functions on the ordinary Linux control and the
protected process. QEMU wall time is useful for test duration, not a hardware
performance claim. A feasibility result must list every disabled OS function,
kernel change and runtime workaround alongside its measurements.
