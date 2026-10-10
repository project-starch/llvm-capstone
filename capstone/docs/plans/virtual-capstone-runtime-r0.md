# R0: virtual C execution through a trusted adapter

Status: implemented QEMU execution experiment, 2026-10-06. Qualification and
input hashes live with the [Linux gate](../../tests/virtual-runtime-r0/README.md)
and [instruction gate](../../capstone-qemu/tests/virtual-capstone-runtime/README.md).

Follow-on implementation: the [virtual application adapter](../../runtime/virtual/README.md)
now connects the loader, common libc and delegated-service loop. It adds
Linux-backed heap growth, first-touch fault resumption, arena retirement and
SQLite/mruby gates. The text below records the original R0 boundary; its
remaining integration steps are completed for that bounded application profile.

This implements the first execution experiment in
[virtual Capstone through the existing runtime](../design/virtual-capstone-runtime.md).
A small Capstone-compiled C function executes at ordinary Linux virtual
addresses. Linux supplies its mappings and resident pages. A loadable module
enters C mode and receives a checked execution event. The existing kernel and
firmware binaries are used without additional patches.

The normal `.dom` loader and delegated syscall loop are not connected to this
entry yet. The experiment reuses the QEMU supervisor's context snapshots,
quantum timer, event delivery and restoration code. It adds a trusted entry
form alongside physical CALL/RETURN. This establishes a usable execution
boundary without first changing the OpenSBI monitor's service ABI.

## What is reused

The branch starts from the virtual prototype's QEMU `dc8878185d` and the
superproject's runtime design `bd13d9348b75`. It carries the developer's
bounded guest node tables, CSMINT, physical tags, precise linear transfers,
permission checks and explicit capability fault values. It does not fork the
node-list algorithm or replace the physical monitor.

A context flag selects virtual C execution. Its data and instruction accesses
use the ordinary user MMU index, the selected `satp`, and user PTE permissions.
Capability addressing still applies because execution remains in C mode.
The lifetime helpers select that context's guest table. Physical C execution
continues to use the existing addressing and lifetime path.

## Experimental entry contract

`CSRUNV rd, frame_pa, action` uses custom opcode `0x5b`, funct3 `1`, funct7
`0x24`. This is a QEMU platform experiment, not an allocated or frozen ISA
encoding. It is available with the existing `x-capstone-u-mode=true` property
on one hart. Trusted S or scalar M code may invoke it. Existing C domains
and supervised applications may not invoke this physical-frame interface.

The caller owns a resident, 16-byte-aligned physical RAM frame. The frame and
node table must remain outside application authority until the context is
discarded. Their physical addresses belong to the trusted adapter, not to
application pointers.

| Byte offset | Contents |
|---|---|
| 0 | Sv39 `satp` |
| 8 | Physical guest node-table root |
| 16, 24, 32, 40 | Event kind, cause, PC and fault address |
| 48 | Scalar `a0` result |
| 64 | Tagged PCC, 16 bytes |
| 80–575 | Tagged initial x1–x31, 16 bytes each |

Action 0 enters a new context or resumes the paused context associated with
that frame. Initial capability slots are consumed when execution is accepted.
Action 1 discards the association. A terminal context must be discarded before
the frame is reused. The trusted caller controls frame allocation and lifetime;
a physical frame address is not proposed as a permanent process identity.

A new application inherits no kernel registers or capabilities. The supervisor
saves the trusted caller, installs the application's registers, translation
and lifetime roots, and restores the caller on escape. It flushes translations
when switching between physical and virtual execution. A paused context retains
its original roots; rewriting the input frame does not replace them on resume.

Kind 1 is a resumable quantum/interrupt event. Kind 2 is a terminal synchronous
fault. The small programs deliberately finish with ECALL, reported as cause 11
because they execute in C mode. This is a test completion convention, not the
application syscall ABI. Missing pages also terminate this first experiment;
recoverable page faults remain R2 work.

## Enforcement and bounded adapter

Every virtual C instruction has a runtime PCC check. Translation also checks
the requested instruction span before reading it. A block contains one guest
instruction, avoiding speculative translation faults from a later instruction.
The gate revokes a paused PCC and resumes an already translated loop, checking
that cached code cannot outlive its authority.

Data accesses use the prototype's rights, bounds and liveness checks and
physical tag tracking. LDC/STC use its consuming path. Privileged Capstone
operations, domain transitions and privileged system controls are unavailable
to the virtual application. The integer-only gate does not qualify floating
point or vector execution. Compact-bounds conformance beyond this gate remains
part of the prototype's encoding work.

The module is a bounded test adapter with a root-only device. It accepts one
host thread, private resident anonymous mappings, distinct mapped pages and
nonoverlapping code/data/stack/TLS ranges. It pins those pages, mints initial
authority, runs bounded steps and releases the pages after discarding the
context. Linux remains responsible for allocation and scheduling.

This adapter creates one lifetime namespace per invocation. On teardown it
preserves output bytes while clearing all tags on the registered pages with
scalar stores, then clears kernel frames and drops the table. Without this,
a subsequent invocation could reuse an identity while stale tagged pointers
survive in user memory. The Linux gate retains old pointer bytes and checks
that this cannot restore authority. A long-lived application will instead
keep its namespace across service calls and arena growth.

The CPU is explicitly configured for Sv39. A wider Linux VA layout is outside
this entry contract; it must not be silently treated as an Sv39 process.

## Gates and next integration step

The instruction gate covers scattered backing, capability/PTE denial, PCC
rights and bounds, revocation including cached code, linear consumption,
forbidden controls, context-root restoration and actual quantum resumption.
The Linux gate requires exact causes and PCs, real nonadjacent physical frames,
code/stack/TLS accesses, cleanup, stale-slot rejection and a Capstone-compiled
C function. Existing M1/U-access suites and physical mruby applications remain
regression gates. This is QEMU evidence; no RTL claim follows from it.

Next connect this entry to the existing loader and delegated-service loop:

1. Load code, globals, TLS and stack into owning-process mappings; perform the
   existing capability relocation and CRT initialization at virtual addresses.
2. Retain one context and lifetime namespace across delegated service events.
   Define which event resumes after a call, and which terminates the process.
3. Reuse the current checked buffer transport and Linux syscall implementations.
4. Grow arenas through Linux mmap; revoke before free or whole-arena retirement,
   then qualify retained stale pointers after reuse at the same virtual address.

Keep recoverable faults, general threads, fork, shared tagged pages and
swap/migration outside that first application integration. Their contracts
remain necessary before claiming the broader design complete.
