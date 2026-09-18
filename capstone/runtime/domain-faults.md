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

The PostgreSQL guest launcher recognizes the sentinel, stops its optional output
thread, closes its Capstone device, restores/unblocks SIGSEGV and raises it.
The Linux process dies; a supervising process and the VM can continue. A normal
domain return still follows the existing path.

## Boundaries

- Only this entry runtime and its PostgreSQL launcher are integrated. Other
  launchers must explicitly handle the reserved result; other entry/yield ABIs
  need their own recovery support. Do not return the sentinel as a normal result.
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

The [PostgreSQL fault suite](../ports/postgres/memory-contexts/security-tests/run-fault-isolation.py)
checks exact fault PCs and causes, three attempts to re-enter each quarantined
domain, out-of-bounds and untagged accesses, destroyed application SP/GP, and a
fault during region sharing. Negative cases exercise a legacy RWX vector and
damaged recovery state. Classifier unit tests reject missing data, wrong PCs,
resumed clients, normal exit status 139 masquerading as a signal, and missing
post-fault liveness.

The process test uses `waitpid`, not just a shell exit code: healthy Generation
client → Generation/Slab/Bump reset-UAF clients each terminated by SIGSEGV →
healthy Generation client, all within one VM boot. Ordinary client tests cover
all four allocators in both spatial and Sublet configurations.

Installing a vector repeatedly also exercises QEMU's CCSRRW zero-register fix:
discarding a tagged old CCSR value into `x0` must not tag the zero register and
make a subsequent `INIT(..., x0)` raise an unexpected-operand fault.
