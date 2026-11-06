# Copyable-only virtual prototype

## Contract

All application memory capabilities are copyable. Keep protected tags, bounds,
permissions, per-address-space lifetime identity, CDERIVE, CREVOKE and privileged
mapping retirement. Remove exclusive ownership, consuming ordinary transfers,
linear carving, and the associated legacy initialization/reclamation protocol.
Allocation disjointness is an allocator responsibility. Initialization at
isolation boundaries remains a kernel/runtime responsibility.

No legacy ownership instructions may execute, even in trusted bootstrap. Do not
turn DELIN into a no-op or translate an old linear mint request into a copyable
one silently. Preserve privileged raw tag transport for Linux page maintenance;
it transports opaque tagged storage and does not grant ordinary access authority.

The initial implementation is on `prototype/copyable-only` in the superproject
and QEMU. It starts from superproject aeabf0091287 and QEMU 0542c5556c2d.
This is an experimental ABI fork, not a replacement for the verified baseline.

## Implementation gates

1. **Instruction boundary (implemented and tested).** Remove decoder entries for
   REVOKE, MREV, SPLIT, DELIN, INIT, DROP, SEAL, CALL, RETURN, legacy CAPENTER,
   debug mint/cap-memory-entry/tag-map-clear, and sealed-domain supervise/GC.
   Reject linear CSMINT descriptors before publication or node allocation.
   The QEMU copyable-only contract tests pass 15 exact illegal-instruction checks
   at each of M/S/U and a rejected-linear/successful-copyable mint pair.
   An older legacy-capable QEMU fails at REVOKE with a capability-operand fault,
   demonstrating that the rejection fixture distinguishes the old behavior.
2. **State and bootstrap audit.** Remove unreachable legacy helpers/types and
   reject legacy tagged state at context entry/restore and other ingress points.
   Audit reset, capability CSRs, trap handling and the compiler-generated
   transfer paths. CSRUNV remains the prototype execution boundary; it must not
   rely on sealed/linear tokens. Ordinary scalar Linux boot must work without
   legacy entry instructions. No legacy firmware fallback is permitted.
3. **Grant and runtime migration.** Version the adapter ABI and remove the
   linear field from the new interface; reject old adapters. Change the
   launcher, module, libc mapping helpers and thread-frame setup together.
   Remove `vm.h`'s DELIN. Preserve lazy admission and distinguish requested
   capability authority from current VMA protection: PROT_NONE reservations
   and later mprotect must work without using linearity as their discriminator.
   Replace/refuse the old linear-only CV_REMAP path explicitly; retain the
   existing allocate/copy/retire public remap semantics. Keep pins for this gate.
4. **Pinned safety gate.** Run mapping/protection, allocator, nested lifetime,
   stale-pointer, bounds, thread/context, and same-address-reuse tests. Check
   lifetime-ID reuse cannot revive pointers. Replace linear-only test fixtures;
   do not weaken safety assertions to make them pass. Test ordinary capability
   loads/stores preserve both source and destination, including across traps.
5. **Unpinned integration.** Reconcile the existing Linux 6.1 unpinned patch and
   raw tag moves with this ABI. Preserve tags across supported copies/migration;
   clear before reuse by any owner. Keep supervisor-frame pins initially.
   The first supported scope is one hart, no KSM/THP, no tagged swap, and no
   claim of protected fork support from the existing COW stimulus alone.
6. **Paired measurements.** Current pinned / copyable-only pinned /
   copyable-only unpinned. Use the same compiler, workload, logical markers,
   kernel configuration and CPU settings wherever compatible; record every
   required binary/ABI difference. The ISA change requires distinct baseline
   and prototype QEMU binaries, built from the same base. Do not regain legacy
   instructions with a compatibility flag just to share one binary.

## Measurement acceptance

Use the existing allocation workload from `inspect/musl-differences` as a frozen
input. Count program, Linux, module, launcher and firmware instructions separately;
also count pin operations, faults, page copies, tag clearing, ioctls, resident
pages and lifetime bookkeeping. Run repeated alternating arms. Include movement
and physical-page-reuse tests so removed pin work is not compared while ignoring
its replacement. QEMU instruction counts do not establish hardware latency,
bandwidth or coherence costs. Do not attribute the whole baseline boundary delta
to linearity, and do not publish performance numbers before safety gates pass.

## Current qualification boundary

The pinned and unpinned Linux prototypes now run with ordinary OpenSBI. The
module discovers a versioned copyable profile through the device tree, creates
only copyable grants, and rejects the old grant ioctl. Legacy ownership/control
encodings have no decoder entries at any privilege. The old remap transaction is
refused; public remapping uses allocate/copy/retire. CSRUNV rejects non-copyable
tagged register state on entry/restore. Device interrupts use ordinary MIP;
pending interrupts are not overwritten by context restoration.

Both modes pass 22 runtime checks on the same patched kernel, including expected
fault causes, ABI refusal, timer wakeup and module unload. Additional tag-copy,
physical reuse, tagged-swap refusal, firmware-interrupted-copy and COW movement
checks pass. Complete deletion of unreachable legacy helper/type code and broader
ISA/OS qualification remain work; there is no legacy instruction fallback.

Three matched allocation runs per mode give 226,118,941 -> 198,791,518 median
instruction attempts (-12.085%). Page-pin entry attempts fall from 14,515 to zero.
This measures unpinning inside the new copyable-only stack, not the independent
benefit of removing linearity. See the [build/run instructions and qualification](../../runtime/virtual/tests/copyable-only/README.md)
and machine-readable results there. Single-hart QEMU only; supervisor-frame pins,
tagged-swap refusal and KSM/THP restrictions remain. No RTL or hardware-cycle
performance claim follows from these results.
