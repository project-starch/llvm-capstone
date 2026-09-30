# Capstone memory with a trusted Linux kernel

Status: NEW DESIGN DIRECTION, 2026-09-30. This branch starts from `dev`
(`4439dd0a55f9`), independently of the caplified mapping implementation lanes.
This document specifies the intended system. It does not claim an implemented
Linux capability ABI, working `fork()`, or measured hardware savings.

## 1. What we want

Applications should use the same operating-system functions as ordinary Linux
processes, through familiar libc interfaces. Linux manages virtual memory,
files, networking, scheduling and processes. Our contribution enforces the
authority and lifetime of pointers to objects inside that memory. Missing IPv6,
file mappings, threads, `fork()` or dynamic loading are integration gaps to
close, not accepted properties of the protection model.

Additional application changes must follow from capability representation,
object boundaries or lifetimes. This is source and functional compatibility,
not binary compatibility with existing 64-bit libraries. The
[application compatibility milestones](../plans/trusted-linux-application-compatibility.md)
turn this goal into shared platform work and acceptance gates for every port.

Linux is trusted for confidentiality, integrity and isolation between processes.
Firmware that can change execution state or memory ownership is trusted too.
Neither is an adversary in this design. Applications and their inputs may be
malicious or contain memory errors. The compiler, libc and allocator components
that establish object boundaries and lifetime rules are part of the relevant
memory-safety TCB, together with the hardware and privileged software.

The intended guarantees are bounded access through genuine capabilities,
revocation of stale object pointers before storage reuse, and containment between
components with separately delegated authority. An in-bounds logic error or
misuse of an accessible genuine capability is outside those guarantees. Bounds
compression must not expose another allocation; exact requested-size bounds
require an explicit representability policy. Heap coverage does not imply
complete stack or global-object lifetime coverage.

This is a deliberate change of threat model. A compromised kernel or monitor can
read or corrupt application memory. Linux syscall implementations are trusted;
file contents, network data and other applications' messages remain untrusted
input. Availability, authenticated launch and side-channel resistance are not
new guarantees supplied by this design.

## 2. Three responsibilities

| Responsibility | Owner | State |
|---|---|---|
| Map virtual pages to physical frames; handle faults, page permissions and backing | Linux and the ordinary MMU | VM areas, page tables, TLB and physical-page accounting |
| Divide memory into objects; manage nested lifetimes and reuse | libc and the allocator | Allocation records and lifetime-control authority |
| Enforce pointer authority on each access | Capstone hardware | Capability bits and tags, protected object-lifetime metadata |

One process address space has one page-table root covering all its mappings.
The heap, stack, libraries and separately allocated mappings can coexist under
that root. There is no need to switch roots for each pointer or each `malloc()`.
Linux's page tables already provide this address-space structure.
[Linux page-table documentation](https://docs.kernel.org/mm/page_tables.html)

A capability-enabled user process scheduled by Linux is the preferred
integration direction to investigate. The final execution mode and hardware
mechanism remain open until the milestone plan's M1 decision. Current Caplifive
capability execution uses C-mode; trusting Linux does not change that mode into
Linux user mode or supply ordinary translation to its accesses.
[Caplifive modes](https://capstone.kisp-lab.org/specs-caplifive/)

The existing module, monitor and delegated launcher can remain a migration
bridge. Their removal is not a prerequisite for restoring OS functionality.
Integrating capability execution with user privilege, faults and Linux context
switches is work to specify and implement, not an existing feature asserted
here. Capstone is the current research platform, not a settled processor choice
for every implementation of this goal.

## 3. Follow one access

Let **c** name a process address-space instance, not a numeric PID or a reusable
TLB ASID. Threads sharing an address space share **c**. The kernel installs both
its translation context and its object-lifetime context when scheduling it.

| Symbol | Meaning | Where it comes from |
|---|---|---|
| p | A 128-bit capability representation | Register or memory, with an out-of-band validity tag |
| a, [b,e), pi, nu | Cursor, decoded bounds, rights and lifetime identifier | Decoding p; these are semantic projections, not independent uncompressed bit fields |
| N_c | Protected object-lifetime state for address space c | Selected by privileged context state; inaccessible to ordinary application stores |
| T_c | Virtual-to-physical translation and page permissions | Ordinary page tables selected by the current process context |
| F, delta | Physical frame base and offset within it | The MMU/TLB result |

For an access of width **w > 0** and operation **op**, the intended condition is:

```text
cap_ok(c, p, op, w) =
    tagged(p) and usable_type(p) and live(N_c, nu(p))
    and op is permitted by pi(p)
    and [a(p), a(p) + w) is within [b(p), e(p)) without overflow

access(c, p, op, w):
    cap_ok(c, p, op, w)
        -> ordinary MMU translation T_c over every page touched
        -> page permissions permit op
        -> physical access, otherwise fault
```

For a one-byte read, `T_c(a) = (F, delta, permissions)` supplies the physical
address `F + delta`. No mapping identity `(id, generation)` selects a separate
registry and root. The lifetime identifier still protects the object; a valid
PTE alone never makes a revoked pointer valid. This is the logical condition,
not a required serial pipeline or a count of memory reads.

```text
                  current process c
                    /           \
           lifetime state N_c   ordinary page-table root
                    |                     |
128-bit p --nu--> object check   a --> MMU/TLB --> frame
    |               |                     |
    +--tag/bounds/rights-------- both permit access --> data
```

Two processes may use the same virtual address for different private objects.
The same capability bits are interpreted in the installed address-space
context. Within one address space, capability delegation preserves meaning.
Across processes, raw capability transfer is not an import operation: ordinary
IPC transports data or kernel-mediated handles. Shared tagged mappings require
a separate namespace/import contract and are excluded from the first prototype.
The kernel must enforce that exclusion; it cannot be a convention applications
are expected to obey.

**N_c is the preferred mechanism to investigate, not an existing ISA facility.**
The current global revocation-node implementation cannot simply be declared
process-local. A privileged namespace selector, lifetime-cache tagging or
flushing, node-generation rules and context-switch semantics need a concrete
design. This cost remains even if the mapping registry disappears.

## 4. Dynamic malloc and free

Dynamic allocation is a direct goal. The allocator can use an existing arena
without entering the kernel. When it needs more space, libc requests another
anonymous mapping. Linux chooses backing and can supply it on demand. Anonymous
memory is initially zero; `malloc()` itself does not promise zeroed contents.
[mmap semantics](https://man7.org/linux/man-pages/man2/mmap.2.html),
[malloc semantics](https://man7.org/linux/man-pages/man3/malloc.3.html)

The following are semantic operations, not proposed instruction encodings:

| Operation | Owner and transition |
|---|---|
| `mmap(length, rights) -> arena capability or error` | Linux reserves a process-local virtual range and establishes the mapping; the trusted capability ABI supplies authority for that range to libc. An integer address alone is not a tagged capability. |
| `malloc(n) -> p or NULL` | libc selects a slot, creates a fresh lifetime in N_c under the arena, and returns appropriately bounded data authority. If needed, it obtains another arena first. |
| `free(p) -> void` | For a valid allocation, libc retires its lifetime and descendants, completes the required access synchronization, then makes the slot reusable. This need not change a PTE. |
| `realloc(p,n) -> q or NULL` | libc resizes safely or allocates and copies before retiring the old object. For nonzero n, failure preserves the old allocation. Bounds, tags and lifetime semantics need qualification. |
| `munmap(range) -> success or error` | The trusted VM/lifetime interface retires authority for the removed allocation, completes access synchronization and removes the mapping before unrelated reuse. Partial overlaps require an explicit retirement rule. |

For example, `p = malloc(64)` followed by `q = p; free(p)` leaves **q** invalid
even if another allocation reuses the same virtual address and the same PTE.
The replacement receives a fresh lifetime identity. Freeing a parent allocation
also ends the subordinate lifetimes owned by that allocation.

Growing the heap adds arenas or extends mappings where possible. It does not
move an existing object merely to add capacity. Allocation may fail because of
memory, address-space or lifetime-metadata limits; no unbounded-memory promise
is made. There is no inherited 256-MiB mapping limit or high logical-address
partition from the other design; the selected MMU mode and ABI determine the
address geometry.

Mapping replacement must not bypass object retirement. In particular, direct
`munmap`, `MAP_FIXED`, shrinking backing and later address reuse need one
coherent kernel/libc contract. A page-table update alone cannot implement
temporal safety. By contrast, moving the backing of the *same* live allocation
may preserve its lifetime if bytes, tags and authority remain equivalent.

## 5. Fork: possible, with independent lifetimes

The new address-space semantics remove an obstacle to `fork()`: parent and
child can retain the same virtual addresses while using different backing.
Linux defines separate process memory spaces and implements private-page
copying with copy-on-write. [fork semantics](https://man7.org/linux/man-pages/man2/fork.2.html)

For Capstone, copying bytes and page tables is insufficient. A private-memory
clone must include capability tags, capability registers, allocator state and
the associated lifetime graph. The required abstract transition is:

```text
fork(c_parent) -> c_child or error

private bytes and capability representations initially agree
virtual addresses agree; private backing is independently owned
N_child = independent clone of N_parent for inherited private lifetimes

(c_parent, nu) and (c_child, nu) identify different private lifetimes
free_parent(p) leaves live(N_child, nu(p)) unchanged
```

Using process-local identifiers permits inherited pointer bits to remain
unchanged. Aliases in each process still name the same local object. The clone
must preserve revoked/dead identities as dead and preserve derivation and
generation relationships; cloning only live nodes and recycling the rest could
revive stale pointers. Context identifiers and caches must not alias after
process exit or ASID reuse.

The proposed first implementation uses **eager copying of private pages and
lifetime metadata**, with a consistent snapshot and complete failure cleanup.
That establishes semantics before optimizing with COW. Capability-aware copying
needs an authorized kernel mechanism: ordinary byte copying may erase tags,
and ordinary linear-capability loads/stores move rather than duplicate authority.
Fork creates distinct private objects, so privileged cloning must explicitly
rebind their linear authority to the child's namespace. It must not grant user
code a way to duplicate a linear resource in the same namespace.

External linear capabilities, sealed contexts and shared-resource authority
cannot be duplicated by cloning private metadata. The first contract must
identify and reject unsupported inheritance, or apply an explicit resource
inheritance rule, before publishing the child. Ordinary file descriptors follow
Linux's descriptor inheritance rules; they are not evidence that arbitrary
Capstone handles can be copied. A failed clone leaves the parent's authority
unchanged and publishes no partial child.

**COW is a later optimization with an additional Capstone condition.** Loading
a linear capability can clear its source slot/tag, making an apparent load a
memory mutation. Private sharing must split before that mutation as well as
before stores. COW faults must be restartable without consuming authority;
page copies must preserve tags; mutable lifetime metadata must not remain
shared between independent processes. A prototype may eagerly copy pages that
contain linear capabilities, but detecting and maintaining that classification
is itself part of the contract.

Start qualification with a single-threaded caller. Threads in one address space
share N_c; a later multi-threaded fork needs a consistent lifetime snapshot and
libc lock/handler rules. The standard restrictions on the child after a
multi-threaded fork still apply. This document does not equate the existing
spawn/exec transport with a working memory-cloning `fork()`.

## 6. What hardware remains

| Keep or integrate | Reason |
|---|---|
| 128-bit capabilities, tags, bounds, rights and type checks | Fine-grained pointer authority |
| Object derivation and revocation | Allocation and nested lifetime safety |
| Lifetime context selection and cache discipline | Process isolation and independent fork lifetimes |
| Capability accesses through the ordinary MMU | Linux virtual memory, page permissions and restartable faults |
| Tag-preserving privileged save/restore and page copying | Scheduling, signals, fork and later paging |
| Completion rules for revocation and mapping changes | No old access may corrupt storage after its reuse |

The target omits capability-bearing PTEs, the per-mapping binding registry,
mapping-specific CREATE/POPULATE/DETACH/DESTROY instructions and hardware that
excludes a hostile Linux from its own backing pages. Linux owns ordinary PTEs
and coordinates page initialization, reclamation and TLB synchronization.
These responsibilities do not disappear; privileged software becomes trusted
to perform them correctly.

All capability accesses, including FP, atomics, capability transfers and
instruction fetch where applicable, must obey the selected translation and
permission rules. A linear-capability load that clears its source also needs
write permission on that source mapping; it must fault before mutation when
the page is read-only, allowing the kernel to resolve COW where appropriate.
Kernel copies, signal frames and context switching need an
ABI that handles tags and authority correctly. Stock Linux on the current
Capstone core is therefore not automatically sufficient. The expected
simplification concerns the mapping security mechanism; remaining hardware and
kernel costs must be measured separately.

## 7. Contracts to validate before implementation

| Property | Required example and counterexample |
|---|---|
| Object bounds and rights | In-bounds permitted access succeeds; overflow, forbidden writes and forged pointers fail. |
| Temporal safety | Fresh allocation succeeds at a reused address; all stale aliases still fail, including after unmap/remap and node reuse. |
| Dynamic growth | Retain old objects while adding mappings backed by scattered frames; force memory/node exhaustion and recover without leaks or authority changes. |
| Fork independence | Inherit aliases in registers and memory; free or write in either process; the other's private copy remains valid and unchanged. Inherited stale pointers stay invalid. |
| Context isolation | Equal addresses and node indices in different processes cannot import foreign authority; switching and reusing context IDs cannot reuse stale cached authorization. |
| Linear ownership | Clone private ownership into distinct namespaces; reject unsupported external handles. COW never duplicates or loses authority through an implicit write. |
| Completion | Delay an already checked access across free/unmap on another hart; reuse is blocked until the old access completes or is cancelled. |

Use the [milestone plan](../plans/trusted-linux-application-compatibility.md)
to order this work. Close the execution/ABI decision and process-local lifetime,
privileged cloning and VM-retirement contracts before implementing their new
mechanisms. Qualify ordinary translation and a growing heap before eager-copy
fork, including tags, registers, stale aliases and failure rollback. COW,
shared tagged memory, swap and multi-threaded cloning need their own contracts;
first-prototype exclusions do not waive the final application compatibility
gates. Existing I/O and thread integration can progress in parallel.

The earlier mapping model and emulator results concern a different threat model
and context semantics. They remain useful evidence for those implementations;
they do not establish these new properties. This branch changes documentation
only and does not import their mapping implementation or submodule pins.
