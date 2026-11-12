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

## What the kernel must do at an unmap

A program can end or renew the lifetime of anything it holds a reference to.
The kernel therefore does not remember an identity for a mapping. It takes
the reference that the program hands back and ends that lifetime only if all
four conditions hold:

1. The operand is a reference.
2. Its range is one whole mapping.
3. Its lifetime is live.
4. Its record hangs directly under R.

Condition 4 separates a mapping from an object that an allocator made over
the whole range of a mapping. Ending that object would leave the mapping
live, and the kernel would hand out an address range that live references
still cover.

A renewed mapping keeps its place under R, so its renewed reference passes
the check. A program that ends its mapping itself and hands back the stale
reference is refused, and the address range stays unusable until `exit`. That
is a leak in a faulty program and not a safety problem: every reference into
the range is already stale.

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

## Open design points

Each is a decision about the design and not yet about code.

- Machine-mode traps. Found in stage 1 (2026-11-12): with `sstc=false` the
  firmware takes the timer in M and saves and restores the S registers with
  plain stores and loads, which strips their tags. A kernel that holds a
  reference across such a trap loses it. On QEMU the adapter keeps every use
  of a reference in one straight-line run without a branch, and QEMU takes
  interrupts only between translation blocks. That does not hold on hardware
  that traps between any two instructions. Kernel-saved program registers
  meet the same problem. The design must require firmware that preserves
  register tags across every M trap, or a platform on which no M trap
  interrupts S or U (Sstc and no emulation traps). With `sstc=true` the
  current QEMU guest stalls during boot.

- Partial `munmap`, `mremap` and a fixed mapping over a live one. The abstract
  machine has no partial release. Proposal: the kernel frees the pages and
  keeps the address range reserved until the whole mapping ends.
- Who checks the buffer of a system call. Proposal: the kernel accesses it
  through the reference the program passed, at every access.
- Signals, `fork`, more than one hart, lifetimes for stack objects, what a
  program sees at a resource limit, exact bounds, and swap of tagged pages.
