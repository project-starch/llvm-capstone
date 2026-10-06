# Virtual Capstone through the existing runtime

Status: agreed integration direction; implementation proposal, 2026-10-06.

The [R0 execution experiment](../plans/virtual-capstone-runtime-r0.md) now runs
virtual C code through a loadable Linux adapter using the existing QEMU
supervisor. Connecting the normal application loader and service loop remains
work beyond that bounded gate.

Capstone should use the virtual address space and services of an existing
operating system through the runtime we already have. The first target is a
growing protected application backed by ordinary Linux mappings. Reusing the
launcher, delegated syscalls and resumable execution should make this a
smaller project than first making Linux itself understand every capability
register. The amount of work depends chiefly on fault delivery and resumption;
a working vertical slice must establish that cost.

## Processor philosophy and trust

Design the processor once, with sufficiently general execution and memory
interfaces to support small adapters for different operating systems. Prefer
a module or driver and a runtime library. Change an OS core only where a
concrete integration experiment establishes that an existing extension point
is insufficient, and keep that change minimal.

The OS and the relevant firmware, module and runtime are trusted. The OS may
create initial authority, inspect saved state and move application memory.
Capstone enforces application object bounds, permissions and lifetimes under
that trusted integration contract. It does not protect applications from a
malicious kernel. The existing
[trusted Linux memory model](trusted-linux-memory.md) defines the scope.

| Responsibility | Owner |
|---|---|
| Scheduling, threads and processes | Existing OS |
| Virtual mappings, physical frames, page faults and page permissions | Existing OS and its MMU |
| Files, sockets and other system services | Existing OS, reached through the runtime |
| Object subdivision and lifetime operations | Capability libc and allocators |
| Bounds, permissions, linear transfers and liveness checks | Capstone processor |
| Capability execution context and OS adaptation | Runtime, module and processor interface |

The processor interface must describe architectural state and operations. It
must not depend on Linux structure layouts, syscall numbers or a particular
kernel entry sequence.

## Start from the working runtime

The [current application runtime](../../runtime/applications.md) already has a
Linux launcher, managed ownership, supervised execution, saved continuations,
termination and delegated services. The
[delegation ABI](../plans/delegation-abi.md) runs real Linux syscalls in the
owning launcher task. These are the starting components.

The current memory path still carves physical authority, retains a physical
pool and copies buffers through exchange storage. Its documented memory
limits and some missing mapping functions follow from that integration.
The supervised execution support is qualified on a bounded QEMU platform;
its documentation does not establish an FPGA implementation or general
multicore support.

The next implementation reuses that execution path while giving the
application Linux virtual addresses. It does not require removing the
monitor or delegating directly from capability registers into every Linux
syscall handler.

## One virtual address space

Application code, data, stack, TLS and heap use mappings in the owning Linux
process. The launcher and capability application can refer to the same
registered buffer by the same virtual address. The module grants application
authority only for registered mappings; sharing an address space does not
grant the application authority over all launcher memory.

```text
capability cursor and bounds are virtual
                 |
       bounds, rights and lifetime check
                 |
       owning process page tables
                 |
       translated physical frame and page rights
```

Instruction fetch and every supported data access must use the selected
process translation context and user access permissions. Retaining a
privileged execution mechanism must not accidentally bypass translation or
PTE permissions. Adding translation to ordinary loads alone is insufficient.

An address-space instance A owns its lifetime namespace N_A. A thread has a
separate saved execution context C. Threads in A share its mappings and
lifetimes, but each has its own registers and PCC. A reusable ASID is not a
lifetime identity, and a page-table root alone is not a thread identity.

When the heap needs space, the runtime obtains a Linux mapping and the module
creates an arena capability for its virtual range. The allocator derives
object capabilities and fresh lifetimes within it. A large virtual arena may
span physically scattered pages. On whole-arena removal, the integration
retires its authority and completes outstanding accesses before unmapping and
unrelated reuse. Object free normally leaves the Linux mapping in place.
Moving the backing of the same live object preserves its lifetime when bytes
and tags remain equivalent; replacing it with an unrelated object does not.

## Reuse explicit execution and resumption

Extend the existing supervised entry and continuation mechanism to carry the
application's translation context and lifetime root. Keep execution state in
an explicit context; do not infer it from Linux register-save patterns.

The architectural contract must cover:

- Enter or resume C with its associated address space, lifetime namespace and
  complete capability register state, including PCC.
- Suspend it on a service request, execution quantum, fault or termination
  request before ordinary OS or firmware code can destroy live metadata.
- Prevent simultaneous execution of the same saved context on two harts and
  prevent restoration of a destroyed context.
- Resume a failed access at its original instruction after a recoverable
  page fault, with no partial linear transfer.

A missing page becomes a recoverable event with the virtual address, access
kind and saved faulting PC. The adapter asks the OS to resolve the mapping
through its VM interfaces and resumes the instruction. An invalid capability
remains a capability fault. The adapter must distinguish the two without
implementing its own pager.

The runtime must return control to Linux even when an application never makes
a service call. Linux schedules the owning host thread; the runtime resumes
its saved capability context when that thread runs again. Later threading
work should map application threads onto OS threads rather than add a second
scheduler.

These are required extensions to the existing execution path, not claims that
it already handles virtual faults. A small experiment must select and verify
the concrete entry, exit and fault mechanism before its ISA is frozen.

## System services and memory tags

Keep the current syscall shape checks and delegated service implementations.
First make virtual memory work with the existing buffer transport. Then allow
direct buffers where the original capability covers the complete requested
range and its lifetime is protected for every OS access. Concurrent free or
unmap must not turn a delayed output into a write to a replacement object.

Common virtual addresses can remove exchange copies for suitable buffers.
Structures containing capability pointers, such as capability iovecs, still
need conversion to the OS ABI. Additional syscall and library coverage remains
software work; virtual addressing does not provide full POSIX compatibility
by itself.

Virtual capabilities do not require virtual tags. Keep physical tags for the
first experiment. Resident pages may be pinned after allocation by Linux;
they need not be physically contiguous. Account for and release those pins.
This permits heap growth without claiming transparent swap or migration.

Before freezing the long-term memory interface, specify how tags survive a
trusted page move or export and restore. A byte copy alone does not preserve
capability tags. Restoring metadata onto new or modified contents must not
fabricate authority. The trusted OS may perform these operations; determine
which existing hooks the adapter can use and identify any genuinely necessary
core hook. The OS continues to choose pages and backing storage.

## Smallest implementation and acceptance

The first application is static, on one hart and one thread, with private
mappings and whole-arena retirement. Its heap grows through Linux mappings;
its code, stack and TLS also execute under the selected page tables. Resident
backing is acceptable for the first memory demonstration. Recoverable faults
are a separate required gate before claiming demand paging.
Fork, shared tagged mappings and application signal-register editing need
separate contracts and are outside this first slice.

| Step | Work and acceptance |
|---|---|
| R0 Execution slice | Reuse the launcher and supervised continuation to run a small capability program at Linux virtual addresses. One object crosses pages deliberately backed by nonadjacent frames. Capability denial and PTE denial independently fail; legal accesses succeed. Record every required change by layer. |
| R1 Growing heap | Connect arena creation to Linux mmap and retirement to munmap. Grow beyond the initial heap reservation, use the added memory, and return it. Old pointers fail after free and after unmap followed by reuse of the same virtual address; fresh pointers succeed. |
| R2 Faults and scheduling | Resolve a deliberately absent page through Linux and retry exactly once. Exercise consuming LDC and STC faults, timer suspension, two independent application processes, and termination of code that never yields. Bound and release retained resources. |
| R3 Application reuse | Rebuild representative existing applications with the common SDK. Preserve output and memory-safety gates. Measure launch cost, service transitions, copied bytes, metadata, pinned pages and memory returned to Linux. Expand file mappings and threads from observed application needs. |

R0 is the next bounded experiment. It determines whether the current
execution bridge needs a small extension or a different entry mechanism.
Do not make direct-buffer I/O, automatic trap capture or full Linux register
context support prerequisites for this first result.

The initial platform target is an unchanged Linux core plus the module and
runtime, with processor and firmware changes allowed. Any required kernel
patch must name the unavailable extension point and demonstrate the smallest
working change. Before committing the processor interface to RTL, exercise
the same context and VM contracts with a second OS adapter, including two
threads in one address space and the proposed metadata transfer operations.

## Existing work and effort

Reuse the [virtual prototype](../plans/virtual-capstone-prototype.md) and its
[ABI work](../plans/virtual-capstone-abi.md): precise linear transfers,
permissions, fault values, representability, guest lifetime tables and
privileged authority operations remain useful. Their integration with the
supervised runtime is additional work. The Linux entry and return patches
remain an alternative integration experiment, rather than a prerequisite for
the runtime route. This proposal changes no implemented encoding or opcode.

The expected saving comes from reusing the runtime and Linux VM instead of
expanding physical-pool management or first adapting all kernel contexts.
The work concentrates in translation, fault and continuation handling, the
loader and arena allocator. Node exhaustion and capability ABI restrictions
remain independent limits to measure.

This is a credible short path to a useful virtual-memory prototype. A calendar
estimate should follow R0; general threads, paging, file mappings and broad
application compatibility are subsequent milestones, not a small translation
patch. Keep kernel changes, platform-specific code and duplicated OS state
as explicit costs when comparing designs.
