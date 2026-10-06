# Virtual Capstone: first QEMU/Linux prototype

Status: **M0 in progress**, 2026-10-06. This is the implementation contract
for the `virtual-capstone-prototype` branch. It narrows the earlier
[Linux execution-boundary proposal](../design/trusted-linux-execution-boundary.md)
to milestones M0--M3 below. Those milestone names are local to this plan;
the older application-compatibility plan uses a different numbering.

## Deliverable and baseline

One static capability C program runs as a Linux U-mode process. It has one
address space, one thread and one hart, uses private anonymous arenas, and
exercises `malloc`, `free` and a small allowlist of synchronous syscalls.
Successful accesses must work; out-of-bounds, revoked and insufficient-rights
accesses must fail, including after reuse of the same virtual address.

The starting superproject is `3638abe58a4c`; its QEMU pin is
`2e6d0ff145205743d863b45fca83567e32b3d080`. The existing
[Linux feasibility gate](../../tests/trusted-linux-feasibility/README.md)
preserves one tagged register. Its recorded results do not establish this
contract. In particular, its debug mint, debug mode selector, firmware-tree
swap and separate-mm fork are not the target ABI.

The compiler and QEMU capability encoding remain at this baseline initially.
This project includes no RTL changes or RTL encoding reconciliation. It
preserves the depth-ordered node list and conditional UNINIT behavior; parent
chains, generation-based reuse and automatic hardware trap capture are outside
this prototype. A later measurement without UNINIT is a separate variant.

## Ownership and lifetime contract

`A` denotes an address-space instance, not a PID or reusable TLB ASID. A
capability's lifetime identity is `(A, node_id)`. The capability stores its
existing local ID; trusted context selection supplies `A`. The kernel must
prevent tagged capabilities from crossing address-space instances.

`N_A` is abstractly a forest. A node is live only if its ID is in the selected
table's allocated range and its own valid bit is set. Null, missing,
out-of-range and retired IDs fail closed. The implementation must explicitly
invalidate descendants: checking only a parent's valid bit is insufficient.

| Operation | Required effect |
|---|---|
| SPLIT | Partition linear authority into disjoint bounds; the new node is a sibling of the operand's node. |
| MREV | Keep the linear operand's node and insert a new revocation node as its parent. |
| REVOKE(R) | Invalidate every live strict descendant of `node(R)` in `N_A`. Leave that node live and transform R using the existing return-type rule. |
| DELIN | Convert linear authority to copyable authority under the existing type/permission rules. |
| DROP | Consume the supplied handle; dropping a revocation handle does not implicitly revoke its descendants. |
| Reuse | Publish authority for a fresh, never-before-issued lifetime; old pointers remain dead even at identical addresses. |

Keep the existing REVOKE rule: the result is LIN if all invalidated
capabilities are non-linear or R lacks write permission; otherwise it is
UNINIT and requires initialization before readable reuse. The node-level
linearity summary and handle operations must be checked against that rule;
conservative initialization must not be described as exact handle tracking.

For every arena the kernel retains an inaccessible revocation ancestor. It
creates that ancestor with MREV before exposing the child to U mode. User
SPLIT/MREV therefore remain strictly below the kernel ancestor. The kernel
tracks the ancestor by scalar ID, not by a persistent tagged REV handle.
Whole-arena retirement walks and invalidates all descendants, then retires
the kernel ancestor itself. Complete retirement before unmapping or reusing
the virtual range or its physical frames.

## Execution and storage contract

| State | Contract |
|---|---|
| Protected-U enable | S-controlled task state, exposed by a privileged CSR interface. It applies only in U mode and cannot be disabled by U code. Trap entry preserves enough state for S to restore it. |
| Page root | Linux selects `satp` and performs the normal translation invalidations. Capability accesses use virtual addresses and obey PTE permissions. |
| Revocation root | Privileged table descriptor: physical base, capacity and per-table monotone allocation state. No emulator host pointer is guest-visible. |
| Node storage | Pinned, U-inaccessible guest RAM reserved from ordinary Linux allocation. A defined serialized layout replaces QEMU host structs. |
| Node allocation | No ID reuse or wrap within A. Disable free-list reuse and the QEMU supervisor collector for this profile. Exhaustion raises a defined guest resource trap without partial instruction effects. |
| Capability storage | Existing QEMU 128-bit encoding plus one physical tag bit. Decode bounds from stored bits; do not preserve extra bounds beside a memory tag. |
| Context switch | Switch page root, lifetime root, mode and saved registers consistently before any U execution; flush or context-tag lifetime caches. |
| Context lifetime | Bind the table to the mm lifecycle; destroy at teardown after execution and saved authority can no longer resume. Reused backing must not expose old tags. |

An access checks tag, type, bounds, rights and node liveness, then translation
and PTE rights. The full interval must fit without arithmetic overflow.
Instruction fetch also needs PCC authority. Unsupported access forms trap;
there is no scalar-address fallback in protected U mode. Domain transitions,
CPMP writes, privileged capability CSRs and debug instructions are forbidden
to protected U code.

For Q-12, LDC consumes a non-copyable memory source and STC consumes a
non-copyable register source. Preflight every faulting part before committing
destination bytes/tags, source consumption or UNINIT cursor advancement.
A consuming LDC requires write permission on its source slot from both the
address capability and the PTE. Failure preserves retryable operand, memory
and tag state. Normal page-table accessed/dirty-bit effects are not a promise
to roll back all machine state.

The candidate already checks data permission bits in
`target/riscv/op_helper.c:_helper_access_with_cap`. Q-14 is a coverage and
fault-cause audit here, not an assumption that no check exists. Audit integer,
FP, atomic and capability accesses, fetch, and unsupported vector operations.

## Kernel boundary and firmware

Only the entry/return assembly uses tagged registers. Normal kernel C code
uses scalars. Save all GPR metadata, PCC and required control state before
scalar register saves or firmware calls can destroy them. A capability-width
scratch swap preserves user `tp` while acquiring the kernel context.

Choose one authoritative frame layout before implementing M2. Scalar edits,
including a syscall result in `a0`, must clear the corresponding capability
tag. Restore PCC using `epc` and its saved authority without widening bounds;
check executable type, rights, liveness and the resumed instruction range.
Frame slots are kernel-only, and clearing/reusing them must clear tags.

The three privileged authority operations are memory-to-memory mint, full
requested-span validation and whole-arena retirement. Context save/restore
and the wide scratch operation are additional privileged facilities. Mint
only registered arenas and bootstrap mappings, never an arbitrary user
address. Validation reuses the capability comparator, including liveness.

The fixed QEMU platform must delegate every supported U exception to S:
page faults, access faults, misaligned accesses, illegal instructions,
breakpoints, U ECALL and Capstone causes 24--29 plus resource exhaustion.
Check the writable delegation mask as well as firmware programming. Sstc
supplies supervisor timers. Disable other machine interrupt sources that
could enter an ordinary scalar firmware handler while user state is live.

The critical interval includes U execution, incomplete S entry saves and
partially completed S return restores. Clearing SIE alone cannot protect it
against M interrupts. Pinned context frames and entry code avoid faults while
unsaved state is exposed. Unexpected M entry during this interval must fail
the prototype gate; it must never resume with damaged authority. Resumable
NMIs are outside this fixed platform profile. Explicit SBI calls occur only
after complete saving and before restoration begins.

## Linux, libc and exclusions

Linux supplies initial authority for code, data, stack and TLS. CRT and the
allocator derive from it, set representable object bounds and revoke before
reuse. Check compressed-bound rounding against neighboring objects; removing
the fat-bounds side store can expose bugs in existing allocator assumptions.

The syscall allowlist checks original tagged arguments, full requested
lengths and required rights before copying. Bad buffers return EFAULT.
Only whole registered arenas may be unmapped. Scrub physical frames and
their tags before handing them to a new owner. Both metadata and context
frames remain protected from user writes.

Reject all fork/clone variants for protected tasks, shared tagged mappings,
additional user aliases of private mutable backing, swap, migration, signal
handlers, ptrace register writes and asynchronous I/O. Ordinary unrelated
Linux tasks may still run. Keep an unprotected Linux boot/control gate.

## Milestones and acceptance

| Milestone | Required evidence |
|---|---|
| M0: contract | Forest/list comparison, exact mode/instruction policy, guest table and CSR ABI, trap/context layout, exhaustion and retirement semantics. |
| M1: QEMU | Positive and negative bounds/rights tests; precise linear moves and fault/retry; stored-bit decoding; equal numeric IDs in two tables remain independent; full table traps without partial changes; collector and free-list reuse disabled. |
| M2: Linux context | All GPRs and PCC survive syscall, timer, page-fault retry and scheduling; `a0` scalar edit clears its tag; forbidden U operations and capability faults arrive in S; no unsafe M entry during save/restore. |
| M3: C process | Static C program with malloc/free and checked syscalls; old pointers fail after free and same-address reuse and after whole-arena unmap; fresh pointers and legal accesses succeed. |

Each fault test names the expected instruction, cause and destination
privilege. Include positive controls so denying everything cannot pass.
Counterexample probes must force the actual machine access; optimized C
undefined behavior is not an adequate stale-pointer oracle. Record binary,
source and guest-image hashes for runtime gates. QEMU wall time is not a
processor performance result.

## First implementation sequence

1. Compare the candidate's C node list with an independent forest model.
   Check nested MREV/SPLIT, REVOKE, dropped revocation handles, arena
   containment and two tables with identical local IDs. This is the first M0
   artifact, not proof of the complete ISA or a completed M1.
2. Fix the M0 ABI details still open: serialized node/header layout and CSR
   numbers, mode transitions, privileged-operation encodings, frame layout,
   exhaustion recovery and the precise supported instruction/exception set.
   **Drafted** in the [prototype ABI](virtual-capstone-abi.md), which also
   records three findings in the candidate: capability causes cannot be
   delegated, four Capstone CSRs are reachable from U, and protected U has
   no PCC fetch check.
3. Add Q-12 fault-and-retry probes and implement transfers against those
   probes. Preserve the existing U-mode and ordinary Linux controls.
   **Done** for protected U and the S context path: 25/25 in the reviewed
   [M1 gate](../../capstone-qemu/tests/virtual-capstone-m1/README.md) and
   74/74 in the existing U-mode suite on the same binary. The
   [combined acceptance runner](../../tests/trusted-linux-feasibility/run-q12.sh)
   also requires the bounded Linux process and its stripped-tag control.
   The kernel saves scalar `s2` before consuming STC; the full ABI keeps
   kernel `tp` until the final restore and excludes PCC from GPR cursor edits.
4. Replace the selector/host table with the guest-table context, disable ID
   reuse, and remove fat bounds from protected memory tags. Complete M1
   before extending the Linux patch to full contexts.

The [source/model entry point](../../capstone-qemu/tests/virtual-capstone-model/README.md)
records 8,949 generated prefixes, six directed cases and three detected
controls against the compiled C list at the initial bounds. Step 1 has that
bounded evidence; the remaining M0 ABI work and all guest milestones remain
open. No FPGA or application benchmark is a prerequisite for this start.
