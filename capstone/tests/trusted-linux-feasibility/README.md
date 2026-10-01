# Trusted-Linux feasibility gates

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
hashes and controls. The bounded process gate below now tests one tagged
register through Linux trap entry, a syscall and a task switch.

It is not evidence for a tagged kernel copy, protected `malloc`, all-register
preservation, or safe retirement.

## Bounded Linux process gate

`protected.S` runs as an ordinary Linux U-mode process. It `mmap`s one page,
asks the experimental kernel to select a protected lifetime namespace, and
receives a tagged capability in `s2`. Its first store faults in the page through
Linux's normal lazy allocation, then retries. It dereferences `s2` after
`getpid`, creates a child with a separate `mm`, and blocks in `wait4`.
The parent requires the kernel's switch-away counter to be positive and
reads/writes through `s2` again. The child runs unprotected. The shell reports
the final exit code; this process deliberately uses no user-buffer syscall or
libc yet.

The exact [kernel patch](linux-trusted-u.patch) applies to
`transcapstone-linux` revision `830b3c68c1fb1e9176028d02ef86f3cf76aa2476`;
`git apply --unidiff-zero --check` can verify the zero-context patch before
application. It is also committed
locally on `riscv/trusted-u-entry` as `a56307461a68`; the original repository
denied the push, so the patch makes this gate reviewable without that remote.
Build the patched source with `CONFIG_CAPSTONE_TRUSTED_U=y` and the
same Buildroot cross-compiler used for the guest image. This option is
experimental and requires QEMU's `x-capstone-u-mode=true`. Place the built
`Image` with the boot firmware and rootfs in a scratch image directory;
verify that this is the directory passed to `--image-dir`. For example:

```sh
source capstone/tests/capstone-test-env.sh
make -C /path/to/transcapstone-linux O="$CAPSTONE_TMP_ROOT/trusted-linux-kbuild" \
  ARCH=riscv CROSS_COMPILE=/path/to/riscv64-buildroot-linux-gnu- -j90 Image
cp "$CAPSTONE_TMP_ROOT/trusted-linux-kbuild/arch/riscv/boot/Image" \
  "$CAPSTONE_TMP_ROOT/trusted-linux-images/Image"
python3 capstone/tests/trusted-linux-feasibility/run.py \
  --image-dir "$CAPSTONE_TMP_ROOT/trusted-linux-images" \
  --qemu /path/to/qemu-system-riscv64 \
  --cc /path/to/riscv64-buildroot-linux-gnu-gcc --protected \
  --log "$CAPSTONE_TMP_ROOT/trusted-linux-protected.log" \
  --record "$CAPSTONE_TMP_ROOT/trusted-linux-protected.json"
```

Run the same command with `--control-strip-protected-tag`, a separate scratch
log and record. This control passes only if the guest faults with cause 24 at
the named first `s2` store and exits 132. Adding
`--control-wrong-fault-site` to that run must fail even though the earlier
fault has the same cause. `--control-missing-protected` must also fail.
The [positive record](protected-result.json) and
[control record](protected-strip-control.json) include hashes of the kernel image, QEMU binary,
firmware, rootfs, compiler, sources and guest binaries. The raw serial log
remains in scratch space. This gate checks one tagged register, one protected
`mm`, one possible hart, syscall entry/return, a Linux page-fault retry and a
task switch. It does not check `tp`/`sp` or other capability registers,
signals, ptrace, protected fork inheritance, checked syscall copies, frame
reclaim, object revocation or a real allocator. M1 remains open.

The next gate must extend this one-register process to full tagged register
preservation, obtain authority for ordinary Linux mappings without a debug
mint, and fault a retained old
pointer after `free` and same-address reuse while the fresh pointer succeeds.
The process must also observe ordinary PTE restrictions and receive a
recoverable `EFAULT` for an invalid syscall buffer. A debug-minted capability
in a bare-metal guest cannot close this gate. Compare three ways of preserving
pointer identity across Linux's register saves and copies—explicit tagged
save/restore, automatic trap capture, and disjoint metadata propagation—by
their complete hardware, kernel, libc and compiler costs. The bounded gate
uses explicit tagged save/restore for `s2`; it does not select a general ABI.

Before performance results, freeze one matched application workload and the
acceptable limits for throughput, tail latency, memory/metadata use and
hardware area. Run the same functions on the ordinary Linux control and the
protected process. QEMU wall time is useful for test duration, not a hardware
performance claim. A feasibility result must list every disabled OS function,
kernel change and runtime workaround alongside its measurements.
