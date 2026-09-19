# Allocator-independent fault recovery tests

This standalone CMake project downloads no application sources and does not
build or link any PostgreSQL or Sublet allocator code. It exercises the
[shared entry and Linux policy](../../domain-faults.md).

From the repository root, set the existing toolchain environment, then:

```sh
source capstone/tests/capstone-test-env.sh
FAULT_WORK=/tmp/capstone/runtime-fault-recovery
FAULT_SOURCE="$CAPSTONE_REPO_ROOT/capstone/runtime/tests/fault-recovery"
TOOLCHAINS="$CAPSTONE_REPO_ROOT/capstone/ports/common/cmake/toolchains"

cmake -S "$FAULT_SOURCE" -B "$FAULT_WORK/native" -G Ninja
cmake --build "$FAULT_WORK/native"
ctest --test-dir "$FAULT_WORK/native" --output-on-failure

# CAPSTONE_BUILDROOT_DIR must name the prepared guest build; its existing
# userspace library and compiler are used, without rebuilding firmware.
cmake -S "$FAULT_SOURCE" -B "$FAULT_WORK/linux" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAINS/linux-guest.cmake" -DCMAKE_BUILD_TYPE=Release
cmake -S "$FAULT_SOURCE" -B "$FAULT_WORK/domain" -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAINS/capstone-domain.cmake" -DCMAKE_BUILD_TYPE=Debug \
  -DCAPSTONE_DOMAIN_FAULT_RECOVERY=ON \
  -DCAPSTONE_FAULT_LINUX_BUILD="$FAULT_WORK/linux"
cmake --build "$FAULT_WORK/linux"
cmake --build "$FAULT_WORK/domain"

# Required: select QEMU containing the local-trap-delivery change linked in
# the runtime contract. An old QEMU must not pass the recovery arms.
export CAPSTONE_QEMU_BINARY=/path/to/trap-delivery/qemu-system-riscv64
ctest --test-dir "$FAULT_WORK/domain" --output-on-failure
```

Set `Python3_EXECUTABLE` at configure time if the QEMU runner dependencies
(including pexpect) are in a virtual environment. The shared runner serializes
access to the guest image. Raw logs and hashed input manifests stay in the
external build's `results/` directory.

The seven QEMU arms are:

- Bounds violation, untagged load, destroyed application SP/GP and a fault
  during region sharing. Each checks its exact fault instruction/cause and
  three re-entry attempts: the client body ran once and never resumed.
- Legacy RWX vector and damaged recovery state: expected fail-stop controls,
  each in its own disposable VM. These deliberately demonstrate the limits.
- A same-boot supervisor runs a healthy client, four faulting clients and a
  healthy client again. Every faulted child must die from actual SIGSEGV;
  ordinary exit 139 is not accepted. No allocator is involved.

Native tests verify 16 combinations of normal/fault results, optional cleanup,
ignored SIGSEGV and blocked SIGSEGV; fault cleanup occurs exactly once and never
on a normal result. Classifier mutations and rejected CMake configurations
exercise the negative paths as well as the passing controls.
Two further native cases connect stdout to a closed-reader pipe or a full
blocking pipe: neither may replace SIGSEGV with SIGPIPE or stall termination.
These controls fail against the earlier policy that wrote a diagnostic first.

`domain.c` and `launcher.c` are minimal client examples. They use a small shared
counter region for observations, not an application-specific metadata protocol.
Domains retain monitor resources after the process exits; this is a bounded
test, not proof of complete reclamation or a long-running restart service.

## Verified standalone run

The [2026-09-18 results](results/20260918-qemu.json) record all three native
CTests and all seven QEMU arms on the isolated `dev`-based runtime branch,
including input, compiler and emulator fingerprints. Recovery cases and the
process supervisor completed normally; fallback cases intentionally stopped
their disposable VM and returned runner status 1. Both are checked outcomes,
not skipped tests. The generic process arm used no library diagnostic marker:
its oracle required exact fault PCs and actual child signal deaths, followed
by healthy execution.

