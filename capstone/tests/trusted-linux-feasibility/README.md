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

The current `result.json` records a passing run with the experimental CPU
property enabled. It used the available Buildroot **glibc** toolchain and
guest images, so it is an OS/control baseline, not the project's target musl
application profile. The result reports `protected_process: false` by design.

The next candidate uses `kernel-context/capstone_s_context.c`. It executes
S-mode STC/LDC from a module loaded into the **actual booted Linux kernel**,
before any protected U-mode process is selected. It saves an untagged scalar;
the bare-metal QEMU gate separately checks a tagged capability. Build the
module against the same prepared kernel source as `Image`, in scratch space:

On the tested host, that prepared source and the booted image report Linux
6.1.0. The Buildroot configuration still names 6.1.26, so use the image and
prepared source identities in the result record when reproducing this gate.

```sh
source capstone/tests/capstone-test-env.sh
module_build="$CAPSTONE_TMP_ROOT/trusted-linux-kernel-context"
mkdir -p "$module_build"
cp capstone/tests/trusted-linux-feasibility/kernel-context/{Makefile,capstone_s_context.c} "$module_build/"
make -C /path/to/buildroot/build/build/linux-custom ARCH=riscv \
  CROSS_COMPILE=/path/to/buildroot/build/host/bin/riscv64-buildroot-linux-gnu- \
  M="$module_build" -j90 modules
python3 capstone/tests/trusted-linux-feasibility/run.py \
  --image-dir /path/to/buildroot/build/images \
  --qemu /path/to/qemu-system-riscv64 \
  --cc /path/to/buildroot/build/host/bin/riscv64-buildroot-linux-gnu-gcc \
  --kernel-module "$module_build/capstone_s_context.ko" \
  --log "$CAPSTONE_TMP_ROOT/trusted-linux-kernel-context.log" \
  --record "$CAPSTONE_TMP_ROOT/trusted-linux-kernel-context.json"
```

The runner requires the module's kernel marker, successful `insmod`, and the
ordinary Linux probe's marker and exit status. `--control-missing-module`
omits the module from the guest disk; the ordinary probe still completes, but
the combined gate must fail. A QEMU source mutation that restores the prior
S-mode requirement for active protected U execution also fails this gate.
The [pinned candidate result](kernel-context-result.json) records the input
hashes and controls. The next gate needs a Linux-selected process with a
tagged register saved at trap entry and restored after a syscall and scheduling.

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
