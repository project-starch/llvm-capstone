# Cooperative client-fault recovery in QEMU

The default legacy C-mode fault path terminates QEMU, not a Linux process.
Clients built with `CAPSTONE_DOMAIN_FAULT_RECOVERY` can instead return through
their domain caller. This requires the matching QEMU local-trap-delivery change;
an old emulator still stops on their faults.

Matching QEMU implementation:
[`77d69353b7`](https://github.com/project-starch/capstone-qemu/commit/77d69353b7),
branch `runtime/1-domain-trap-delivery` (stacked on the capability-atomics branch).
Build that checkout in a separate build directory and select its binary through
`CAPSTONE_QEMU_BINARY`. No shared firmware or default emulator is replaced.

This is a **cooperative runtime**, not monitor-enforced containment of malicious
C-mode code. It uses existing instructions and the existing horizontal trap path,
without a new CIH protocol, kernel interface, or firmware image.

## Integrating any client

No PostgreSQL, Sublet, or allocator dependency is required. The shared CMake
helper, entry assembly, Linux policy library and standalone tests live outside
the allocator ports. Recovery remains opt-in, and other ports are not silently
switched to it.

For a freestanding domain using the CALL/REGION_SHARE entry ABI:

```cmake
# Configure with the capstone-domain toolchain and
# -DCAPSTONE_DOMAIN_FAULT_RECOVERY=ON. Enable ASM in the project.
add_subdirectory("${CAPSTONE_REPO_ROOT}/capstone/runtime" capstone-runtime)
add_executable(client client.c)
capstone_configure_domain(client DATA_BYTES 262144 STACK_BYTES 262144)
```

`capstone_configure_domain` adds the shared entry, linker script and resource
declaration. Do not also add those sources or define `CAPSTONE_DOMREQ_*` yourself.
The declared data includes application globals, capability tables and stack;
the helper adds 256 bytes of recovery state when enabled, without reducing the
application's declared budget. Application-specific headers/libc remain the
client's responsibility. Implement `domain_main(result_or_region, function)`;
see the [standalone domain](tests/fault-recovery/domain.c).

For the separately compiled Linux launcher, add the runtime directory and link
`Capstone::LinuxFaultPolicy`. The library has no dependency on `libcapstone`;
it takes an optional cleanup callback so the launcher retains resource ownership:

```c
#include <capstone/linux-domain-fault.h>

static void close_client(void *context) {
    (void)context;
    /* Close the launcher's device/resources; this callback must return. */
    capstone_cleanup();
}

/* After call_dom(), join application workers, then check the result BEFORE
 * consuming the domain's output. Normal results return without cleanup. */
capstone_domain_exit_on_fault(result, close_client, NULL);
```

On the reserved fault result the helper invokes cleanup once,
restores the default SIGSEGV disposition, unblocks
it in the calling thread and raises it. That terminates the **hosting process**,
not just a domain or thread. Invoke this from ordinary launcher code, not a
signal handler. A supervisor that hosts multiple domains and must stay alive
can instead inspect `CAPSTONE_DOMAIN_FAULT_RETVAL` and apply its own policy;
it must never treat the quarantined domain as healthy.

The policy does no diagnostic I/O: stdout/stderr may be closed or backed by a
full pipe. Reporting belongs to the launcher or its surviving supervisor.
Cleanup callbacks must return and must not depend on output making progress.

The [minimal launcher](tests/fault-recovery/launcher.c) shows the complete path.
Allocator-specific adoption is a separate change: existing launchers must opt
in and stop their application workers before applying this process policy.

## Entry and return contract

`my_first_domain/start.S` splits a 256-byte recovery context off the upper end
of the incoming data/stack region before making the remaining stack non-linear.
Application stack bounds exclude this context. `cscratch` retains its capability.
The context holds the unique sealed return capability, CALL result pointer,
global pointer, normal stack top, and dispatch number. Normal and fault returns
consume and explicitly clear the same return-capability slot.

The runtime installs an **execute-only** local `ctvec` before global capability
initializers and client code. Execute-only is the QEMU opt-in convention for this
runtime: the legacy monitor's RWX vector assumes an S/U-mode trap frame and must
not receive internal C-mode monitor faults. Missing, invalid, or non-opt-in vectors
retain the existing report-and-halt behavior. Host S/U-mode trap delivery is unchanged.

On a delivered fault, QEMU reports its original cause and PC. The assembly
trampoline disables `ctvec` first, then uses the reserved context without calling
C or trusting the interrupted SP/GP. For a CALL it writes
`CAPSTONE_DOMAIN_FAULT_RETVAL` (`0x0FA017ED`) and returns to the monitor. The domain's
new entry point only returns the fault sentinel; it never runs the client or its
initializers again. This is cooperative quarantine, not monitor-side destruction.

REGION_SHARE has no result pointer: its argument is the shared region. A fault
during sharing quarantines the domain **without writing into that region**;
the next CALL reports the sentinel. The share operation itself does not gain an
immediate error result.

## Boundaries

- The shared entry and standalone launcher are integrated here.
  Other launchers must adopt the shared policy or explicitly handle the reserved
  result; other entry/yield ABIs need their own recovery support. Do not return
  the sentinel as a normal result.
- Faults before handler installation, damaged recovery state, helper assertions,
  firmware/kernel failures, and explicit CSR tampering are not contained. A fault
  inside recovery retains fail-stop rather than recursively invoking the handler.
- No FPGA behavior is claimed. The QEMU opt-in convention is not a silicon ISA
  guarantee or a replacement for the monitor-level recovery design.
- Closing the device does **not** reclaim all domain/region resources: the current
  driver and monitor retain allocations and finite domain slots. Repeated short
  tests work within those limits; an indefinitely running service needs a separate
  ownership/destruction implementation.
- The existing global CIH is an interrupt scheduler, not the caller's monitor
  continuation. Routing synchronous faults there unchanged would resume or
  reschedule a faulted domain, not implement this return path.

## Regression coverage

The [standalone runtime suite](tests/fault-recovery/README.md), built without
PostgreSQL sources, an allocator, or a replay-region configuration,
checks exact fault PCs and causes, three attempts to re-enter each quarantined
domain, out-of-bounds and untagged accesses, destroyed application SP/GP, and a
fault during region sharing. Negative cases exercise a legacy RWX vector and
damaged recovery state. Classifier unit tests reject missing data, wrong PCs,
resumed clients, normal exit status 139 masquerading as a signal, and missing
post-fault liveness. Native tests cover ordinary results, optional cleanup,
ignored/blocked SIGSEGV, closed/full stdout pipes, and build-option/resource accounting.

The generic process test uses `waitpid`, not just a shell exit code: healthy
client → four faulting clients → healthy client, all within one VM boot.

Installing a vector repeatedly also exercises QEMU's CCSRRW zero-register fix:
discarding a tagged old CCSR value into `x0` must not tag the zero register and
make a subsequent `INIT(..., x0)` raise an unexpected-operand fault.

## Review and adoption boundaries

This feature has three layers: QEMU delivers the synchronous exception to the
opted-in local vector; the shared domain entry returns a fault result and
quarantines the client; a Linux launcher chooses how to handle that result.
The policy library implements process termination, while the re-entry test
demonstrates a caller inspecting the result without terminating itself.

The QEMU change is reviewed in
[capstone-qemu PR #6](https://github.com/project-starch/capstone-qemu/pull/6).
The runtime, public CMake helper and standalone tests can be reviewed and built
directly on the LLVM repository's `dev`, without any allocator-port branch.
[PostgreSQL integration PR #52](https://github.com/project-starch/llvm-capstone/pull/52)
is separate from this generic layer; its application cleanup, region sizing and
allocator-specific security fixtures do not belong in the runtime contract.
