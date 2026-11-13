# The kernel as the outermost allocator

[Guide](README.md) · Related: [ISA](isa.md) · [Runtime and Linux](runtime.md)

Status: design decided on 2026-11-12, on the branch `address-space-root` of
this repository and of capstone-qemu (both from `one-operand-crevoke-crenew`).
The instruction rules are implemented and qualified in QEMU. The Linux
adapter, the libc and the RTL still use the earlier design, described under
"What runs today".

## Decision

The abstract machine starts with one object, R, and one reference to it.
Every other object is a child that some allocator made inside its block.

The kernel is the allocator of R. One reference covers the address space of a
process, and the kernel holds it. A mapping is a child of R:

| Kernel action | Operation of the abstract machine | Instruction |
|---|---|---|
| create the table and R at `exec` | the initial state | `CSMINT`, once |
| `mmap` | `alloc(R, offset, size)` | `CDERIVE` |
| `munmap` | `release(r)` | `CREVOKE` |
| free the table at `exit` | the end of the trace | none |

The system allocator makes its objects inside a mapping, and a nested
allocator makes its objects inside those. The kernel therefore does what
every allocator does: it keeps the reference to its block and hands out
children. Creating R is the only step that makes a reference from numbers,
and the only step reserved for the kernel.

## Mappings

Decided on 2026-11-12, after review; the reservation record was added the
same day after the review of stage 2. Each `mmap` creates a reservation in
two steps. The kernel derives a reservation record M from R and keeps the
only reference to M, as an allocator keeps the reference to its pool. It
then derives the program's mapping reference P from M, with the same bounds,
and hands P out. A kernel object holds the reservation's extent (padding
included), its visible length, the reference to M and the number of mapped
pages. Linux splits and merges VMAs for its own reasons, so the reservation
object, not a VMA, identifies the mapping, and VMAs of different
reservations never merge.

The program can renew or end P and anything it derives, but it cannot reach
M. A `CREVOKE` of M therefore always ends the whole reservation, including
references the program renewed. This removes the failure that the first
version had: there the kernel held a copy of the program's own reference,
and after a `CRENEW` by the program that copy ended nothing while the kernel
still released the range.

Every request of the program to end a mapping comes with its reference:
`munmap`, a park in the mapping cache, the unmap of a thread's stack at
thread exit, and the hand-over of a thread's frame at thread creation. The
kernel accepts the reference only if all of these hold:

1. The operand is a reference.
2. Its lifetime is live.
3. Its record's parent is M, the reservation's record.
4. Its bounds cover the reservation's visible length, and the call names
   that whole length.

Then the kernel executes `CREVOKE` on M. Condition 3 separates the mapping
from an object that an allocator made over its whole range. Condition 4
keeps a narrowed reference to part of a mapping from ending the whole
mapping. A refused operand never reaches an instruction that can trap, and
the kernel releases no range on a refused request. Only the teardown of the
process and the rollback of a failed grant end M without a request of the
program.

| Call | Rule |
|---|---|
| `mmap` | a new reservation, as above |
| `munmap` | only the whole reservation, with an accepted reference: `CREVOKE` of M, then the address range becomes free. A partial `munmap` is refused. |
| `MAP_FIXED` | into free address space: a new reservation. Inside a reservation: only with an accepted reference and only while the program derived no object below P (the record after P's in the preorder list is not deeper, one step) |
| `mremap` | `MREMAP_MAYMOVE` only, atomic: a new reservation, the pages move with their tags, the old reservation ends. On failure nothing changes. In-place resizing, `MREMAP_FIXED` and `MREMAP_DONTUNMAP` are refused. |
| `mprotect` | with a reference that passes conditions 1 to 3 and covers the named page range, which may be part of the reservation. It ends nothing. Page permissions never grant a right that the capability lacks. |
| kinds | anonymous mappings only. File-backed mappings, mappings shared between processes, `brk` and `userfaultfd` are refused. |
| shared region | the system-call region belongs to the kernel, is never unmapped, and its address range is never reused |

An address range returns to use only after the lifetime of its whole
reservation has ended, so no stale reference reaches a new mapping.
Whole-reservation `munmap` keeps the abstract machine exact: a partial unmap
would leave objects with live lifetimes in pages that no longer exist. The
runtime of today already refuses partial unmaps, fixed mappings and in-place
`mremap`, and every evaluated program runs with these limits. Extending the
machine with page availability is the way to allow partial unmaps later.

The prototype's `mremap` (stage 4b of the adapter plan) is map, copy and
unmap in the libc, not one kernel step. Before it copies anything it asks the
kernel, without changing anything, whether `munmap` of the old mapping through
the same reference and length would be accepted, so a stale reference, an
object below P or a wrong length fails with nothing changed and never faults
in the copy. It moves only an anonymous mapping that is read-write over its
whole length, and refuses every other one (`ENOTSUP`). The old reservation
ends through P, as at `munmap`. The copy is not atomic for the program's
other threads: a write to the old mapping during the copy may be lost, as for
`realloc`.

`madvise(MADV_DONTNEED)` and `MADV_FREE` change backing, not lifetimes, and
need no revocation: a fresh page carries no tag.

A program that ends a mapping's lifetime itself and hands back the stale
reference is refused, and the range stays unusable until `exit`. That is a
leak in a faulty program, not a safety problem: every reference into the
range is already stale.

A program that narrows its mapping reference and then renews the narrowed
reference can no longer unmap that mapping. The renewal ends the lifetime
of P and of every copy of it, and gives M a new child with the narrowed
bounds. That child does not cover the visible length, so every request to
end the mapping is refused, and the mapping stays until teardown. That is a
leak, not a safety problem: M stays live, and the kernel releases nothing.

## Supervisor references

The kernel computes with plain addresses, and its own loads and stores pass
no capability check. To hold R and to check an operand it needs two rules,
enabled together by the QEMU property `x-capstone-supervisor-refs`:

- `LDC` and `STC` in S take a plain kernel address as their base and keep the
  tag. A reference moves between a kernel slot and a register unchanged.
- `LCC` selector 8 in S answers the id of the operand's lifetime, the integer
  that names its record. The kernel owns the table and reads the record.

Every other instruction treats the kernel's register as it treats a
program's. `CINCOFFSET` moves the address, and `CDERIVE` and `CREVOKE` act on
the table that `srevroot` selects.

The kernel reads the id with `LCC` and not from a stored reference. A plain
load of a tagged word reads the address, so in the 55-bit format the slot
index is not visible to it.

Without the property both rules are off: `LDC` and `STC` with a plain base
are refused with 24, as on the RTL, and selector 8 answers 0. This changes
the ruling Q-23 of 2026-10-18 ("trusted S performs no capability memory
access") for platforms that set the property.

## Entering and leaving the kernel

Decided on 2026-11-12. The kernel saves the registers of a program itself.

A trap leaves the program's registers in place and enters the kernel as any
RISC-V trap does. The kernel stores the registers with their tags into its
own memory with `STC` and loads them with `LDC` before it returns. On a
switch to another process it also writes the table root, which clears the
processor's lifetime caches. Each thread has its own save area in kernel
memory.

This needs no rule beyond supervisor references: a kernel that holds R can
save a register the same way. It replaces the supervised entry of today, in
which the processor saves all registers into a frame at every event
(`CSRUNV`, its frame format, event kinds, actions and context slots). Three
conditions hold in both forms: a paused program cannot reach its saved
registers, the registers keep their tags, and a change of process changes
the table root.

Not implemented. QEMU has the trap path as a test mode (protected U, which
the address-space root gate runs in). The adapter and the RTL use `CSRUNV`.

## The program counter

Decided on 2026-11-12: unchanged from today. One reference covers the
program's code, and the processor checks every fetch against its bounds, its
execute right and its lifetime. Jump targets and return addresses are plain
addresses inside it. A fetch is thus a read through a reference, like every
other access, and when the lifetime of a code mapping ends, no instruction
executes from it. With kernel-saved registers the code reference is one more
register to save, and the trap's saved pc stays an address. The design does
not constrain where a program jumps inside its own code.

## System-call buffers

Decided on 2026-11-12. The program's libc copies a buffer through the
caller's reference into a region it shares with the kernel, and the kernel
accesses only that region. Every access to an object of the program thus
happens in the program, through the reference the caller passed, and a stale
buffer faults in the copy before the kernel acts. The libc also converts
structures that hold pointers into the kernel's layout, since a user pointer
is 16 bytes and a kernel pointer 8. The kernel refuses access to any other
user address of a protected process, including through pinned pages.

Frequent small calls (read, write, pread, pwrite) take a direct path: the
program passes the reference in a register, and the kernel code of these
calls accesses the buffer through exactly that reference.

Rejected after review: a search of the saved registers for a covering
capability (a stale argument is authorized by another live register), and
handles in the high address bits (integers read from user memory forge
them, and every structure with pointers changes layout). The general
zero-copy design is Morello's PCuABI, in which `__user` pointers are
capabilities in the kernel. It needs a compiler with hybrid capability
types, which llvm-capstone does not have.

## Evidence

`capstone-qemu/tests/address-space-root` runs a small kernel in S that
creates R, maps two pages and serves `map` and `unmap` for a program in
protected U. Eight cases and three controls pass on the narrow id format and
on the 55-bit format, 22 of 22. The controls show that an unmap without
`CREVOKE` lets the stale access through, and that a processor without the
property refuses the kernel's first `LDC`.

Unchanged by the rules: `sublet-lifetimes` 151 of 151, `virtual-capstone-m1`
74 of 74 and its proof gate 8 of 8, `capstone-raw-moves/ldc.sh`.

The gate uses a flat table and protected U. It does not run the Linux
adapter, `CSRUNV` contexts or a paged table.

## What runs today

The adapter creates one root per mapping with `CSMINT`, which makes a private
parent record and a child, and it ends a mapping with `CSRETIRE` on the
parent's number. The kernel never holds a reference. `munmap` names the
mapping by its registered range and not by a reference.

## Work that follows from the decision

1. Adapter: create R when it creates the table, make every registered range
   (image, stack, startup data, exchange, `mmap`) a child of R, check and end
   the operand at `munmap`, carry `mremap` over, and drop the private parent
   records and their collection paths.
2. libc: `munmap` and `mremap` hand the reference to the kernel.
3. Model and contract tests for the adapter path, then the landing gates, the
   port gates and the bug corpora.
4. RTL: neither rule exists there. Until it follows, a kit runs on QEMU or on
   the board and not on both, as with the one-operand `CREVOKE`.

## Further decisions of 2026-11-12

- Signals. The kernel writes the signal frame with tagged registers through
  the saved stack reference, an exact reference like the direct path of
  read and write. `sigreturn` restores capabilities only from that frame and
  validates the restored execution state.
- Long releases. Above a threshold of records, `CREVOKE` traps to the
  kernel, which finishes the walk preemptibly. No thread of the process runs
  until the walk is done, and the release takes effect when every
  generation is advanced and the lifetime caches are invalidated, before any
  slot or memory of the released objects is reused.
- Resource limits. Allocation can fail, and the abstract machine gets that
  outcome: `CDERIVE` reports failure and `malloc` returns NULL, no fault. A
  slot at its last generation retires. `CRENEW` on such a slot ends the old
  lifetime, and only the new one fails when no free slot is left.
- `fork`. The kernel copies the lifetime table at `fork`, with the same slots
  and generations, since the processor writes the table through physical
  addresses and Linux's copy-on-write does not cover it. Out of scope for
  the prototype.
- More than one hart, swap of tagged pages, lifetimes for stack objects: out
  of scope. The claim covers heap objects and nested allocations of
  statically linked programs.
- Allocator trust. The processor checks that a child lies inside its parent.
  That live siblings do not overlap is the allocator's duty. A client holds
  only its object's reference and cannot create an overlapping sibling.
- Machine-mode traps. Firmware preserves register tags across every trap
  into M: it saves and restores the interrupted registers with capability
  stores and loads. Found in stage 1: with `sstc=false` the firmware takes
  the timer in M and restores the S registers with plain loads, which strips
  their tags, and program registers saved by the kernel meet the same
  problem. Until the firmware does this, the adapter keeps every use of a
  reference in one straight-line run, which holds on QEMU only (it takes
  interrupts between translation blocks). The rejected alternative, no M
  trap under S or U, breaks with any SBI call or emulation trap.
  Implemented on 2026-11-12 for the QEMU platform: the monitor's trap entry
  (caplifive-sbi `664dec6`, wrapper caplifive-opensbi `ad308f8`, build option
  `CAPSTONE_TRAP_KEEP_TAGS`) keeps the tags, and a kernel reference held
  across timer traps survives 8 of 8 holds (0 of 8 on the stock firmware).
  The monitor runs in legacy C mode and saves through its stack capability.
  QEMU's rule for scalar M (capstone-qemu `f8aeb15905`) covers OpenSBI's
  own `_trap_handler`, which handles traps only once protected U is the
  program mode. That handler is not patched yet: it needs a register it can
  free without losing a tag, and CSCRATCH holds the monitor's stack.

## Open design points

- Exact bounds.
