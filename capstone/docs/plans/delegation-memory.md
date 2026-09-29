# Delegated memory: Linux backing, monitor grants, domain pointers

Status: DESIGN PROPOSAL, 2026-09-30. No runtime changes or new qualification
results are claimed by this document. It records the memory design discussion,
the recommended first implementation, and the contracts still to be proved.

Base: `delegation-cheap-rows` at
`f0b6f36395df4443a44f47d402f03578798e4b28`; work branch `delegation-memory`.
This is the current delegated application stack, including signals, the
round-cost work and the plain syscall rows. At branch creation, `origin/dev`
at `330014ead628` did not yet contain that stack. The sockets and threads
branches are parallel work, not prerequisites or evidence for this proposal.

This develops memory step 5 of [the delegation plan](delegation-abi.md).
The [signal plan](delegation-signals.md) supplies the existing round, restart
and handler-delivery contract that memory operations must preserve.
The [architectural alternatives](delegation-memory-options.md) explore protected
views, permission changes, translation and file-blind paging beyond this baseline.

## 1. Purpose and philosophy

An application obtains memory from its operating system. Linux supplies the
backing storage; the monitor establishes and withdraws authority over regions;
the domain derives its own pointers and allocates objects inside those regions.
The monitor does not learn filesystems, file descriptors, Linux tasks or mmap
flags. The domain does not reproduce the operating system's storage services.

**Physical addressing in the current runtime:** the domain receives a
capability whose bounds refer to a physically contiguous RAM extent. Its data
accesses do not use the launcher's Linux virtual addresses or Linux page
tables. The module currently calls `dma_alloc_pages` (using CMA for large
extents), obtains the physical base with `page_to_phys`, and passes that base
to the monitor to create a region. Linux therefore chooses and owns the RAM
allocation; the monitor decides which domain may access that physical extent.
These are compatible responsibilities. The launcher may have a separate Linux
virtual mapping of the same pages when preparing or sharing data. For a private
grant, no Linux *user* mapping may retain access after exclusive transfer; the
kernel's ordinary mapping and the module's bookkeeping are not the domain's
pointer and do not disappear merely because the user mapping is removed.
Their existence is not permission to access the transferred memory: Linux's
accesses must still pass the CPMP capability check, and obsolete authority must
be removed on transfer. Kernel privilege or a surviving direct-map PTE cannot
by itself restore the domain's exclusive physical authority.

Today an application receives its data, stack, arena and exchange regions at
startup. Anonymous mmap is allocated inside the arena; file mmap is refused.
The proposal makes region delivery repeatable during execution, so an allocator
can obtain another pool instead of requiring a guessed maximum arena per port.

The first objective is dynamic private memory with a complete lifetime contract.
The next objectives are private file mappings and actual shared storage.
This is not a promise of Linux virtual-memory semantics: a capability over a
physical extent does not supply demand paging, address remapping or page tables.

## 2. Three layers and their responsibilities

| Layer | Responsibility | State it needs |
|---|---|---|
| Linux, including the module and launcher | Allocate and retain physical backing; perform file I/O under the owning task's credentials; expose supported shared objects; enforce device-file ownership and Linux-side limits | Backing allocations, Linux mappings and references, descriptors, file offsets, shared-object names, accounting |
| Monitor | Establish authorized region grants; retain superior revocation authority; deliver capabilities; revoke descendants; reclaim and scrub safely | Physical extents, domain identities, grant identities and lifetimes, exclusive/shared authority, capability permissions, representability, protected continuations |
| Domain libc and allocators | Associate replies with delivered capabilities; implement pointer-returning C interfaces; allocate and free objects; grow and retire pools | Tagged capability slots, local mapping records, pool metadata, object lifetimes |

Exclusive/shared mode and capability permissions belong in the monitor: they
are properties of authority, even though MAP_PRIVATE, MAP_SHARED, PROT_* and
file access modes remain outside it. Linux-side ownership checks belong to the
module; the monitor checks its own domain and authority relationships. An
integer supplied by the launcher or domain is never sufficient to mint authority
over arbitrary physical memory.

The overlap invariant is **no independent exclusive authority over the same
bytes**. A rule forbidding overlap with every live capability would forbid both
normal derivation and shared memory. Shared grants must derive from an explicitly
authorized common backing object, with lifetimes that can be withdrawn safely.

## 3. Existing mechanisms and limits

The following are source-backed starting points, not evidence that the new
protocol already works:

- [mmap_shm_level0.c](../../ports/musl-capstone/runtime/mmap_shm_level0.c)
  implements anonymous mmap and local System V shared-memory emulation using
  malloc. It returns ENODEV for file mmap and rejects partial munmap. Its
  pointer-returning wrappers are necessary because a scalar syscall result
  cannot reconstruct a tagged capability.
- [hostcall.c](../../ports/musl-capstone/runtime/hostcall.c) receives program
  regions at startup and supplies them through `__capstone_region`.
  [start-musl.S](../../ports/musl-capstone/runtime/start-musl.S) has the
  suspended computation's resume path. A new delivery must fit that path without
  overwriting saved state or losing a linear capability.
- [The runtime ownership contract](../../runtime/applications.md#ownership-reuse-and-limits)
  retains monitor revocation roots above transferred storage. Linux mappings
  prevent linear transfer; transferred storage cannot subsequently be mapped
  by Linux through the managed API.
- The same contract distinguishes live storage from retained cache storage.
  Returning an extent to the process cache **does not return its RAM to the
  Linux page allocator**: the monitor still owns the original carved authority.
  True release to Linux is a separate lifecycle operation to implement and prove.
- The documented defaults include a 384 MiB retained-storage budget and 96
  monitor region slots, shared with monitor bookkeeping. Dynamic grants must
  report capacity exhaustion and reclaim usable slots. Removing per-port arena
  guesses does not remove physical, table, revocation-node or cache limits.
- [process-test.c](../../runtime/tests/application/process-test.c) checks
  managed ownership, dup/fork/VMA lifetime, reuse scrubbing and rollback. Its
  `mmap lifetime, scrub` result concerns Linux mappings of managed regions;
  it does not prove runtime delivery or stale domain-pointer rejection.
- [sublet_heap.c](../../ports/musl-capstone/runtime/sublet_heap.c) currently
  implements one buddy pool with static metadata. Sublet's primitives support
  the proposed composition, but this adapter still needs a multi-pool design.

Code and the current headers are authoritative for the implemented ABI. Older
sections of the delegation plan describe earlier wire sizes and error choices.

Implementation entry points in this checkout:

| Area | Source |
|---|---|
| Row format and exception classification | [delegate.h](../../runtime/include/capstone/delegate.h), [common/delegate.c](../../runtime/common/delegate.c) |
| Launcher and Linux service dispatch | [exec.c](../../runtime/linux/exec.c), [delegate-service.c](../../runtime/linux/delegate-service.c) |
| Libc round and signal integration | [runtime/delegate.c](../../ports/musl-capstone/runtime/delegate.c) |
| Mapping wrappers and pool adapter | `mmap_shm_level0.c`, `sublet_heap.c`, linked above |
| Module and monitor | The Buildroot/OpenSBI submodule revisions pinned by this branch; make required changes in their own work branches and record tested revisions |

## 4. Backing, grants and pointers have different lifetimes

Keep these identities separate:

1. **Backing extent/object:** physically stable storage retained by Linux/module
   bookkeeping and represented by the monitor's authority.
2. **Grant:** a particular domain's authority over that backing or an authorized
   subrange, with a generation and a retained revocation ancestor.
3. **Derived pointers:** capabilities produced by the domain, including allocator
   subtrees, aliases saved elsewhere, and capabilities in paused contexts.

For a private region there is one exclusive recipient. The monitor keeps the
superior revocation handle while the domain receives the transferable linear
capability. The domain may split, further derive or delinearize its authority.
Consequently, release cannot require reconstructing the original whole linear
capability from every allocator fragment. Release requests termination of the
grant; the monitor revokes through its retained ancestor.

For shared storage, each participant needs an independently withdrawable grant.
Unmapping in D1 must invalidate D1's derived pointers while D2 and Linux retain
their permitted access. Merely distributing copies of one non-linear capability
does not establish that property. Constructing and testing the required grant
relationships is the second architecture proof, after private transfer.

Do not scrub shared backing when one participant detaches. Scrub on final
reclamation after all relevant grants and Linux references have been handled.
The Linux reference policy must also preserve an object's contents if an open
descriptor still keeps it alive without a current domain mapping.

### Revocation and scrub are separate obligations

Revocation invalidates authority derived from the grant, including a stale
pointer stored outside the returned extent. Scrub removes residual bytes and
embedded capabilities inside the reclaimed storage before another owner can
receive it. Clearing tags inside the extent alone does not invalidate aliases
stored elsewhere; revoking aliases alone does not sanitize the next owner's data.

A reclaimed extent becomes reusable only after both obligations complete and
the platform's saved-context/CPMP handling cannot resurrect old authority. If
reclamation cannot finish, retain or quarantine the extent; never report it as
available. Node-ID reuse must obey the platform's existing stale-tag discipline.

### Geometry

Check physical contiguity, overflow, page alignment and capability-bound
representability before publication. Allocate all padding covered by the actual
representable capability; never round authority outward into somebody else's
allocation. Record the requested mapping length separately from backing size.

A power-of-two block aligned to itself is a useful sufficient rule for buddy
pools, not the general ISA requirement for every grant. The platform's actual
compression granule determines valid bounds. QEMU's full-precision bounds are
not evidence that a geometry is exact on silicon.

## 5. Scalar rows with capability delivery at resume

mmap, munmap and supported msync operations use the existing request/reply
transport. Linux-facing arguments remain integers and buffer offsets. A region
result travels through a monitor-controlled capability delivery path, never
through `(void *)result` or an integer domain address.

Proposed logical fields, with layout to be fixed during implementation:

- Request: operation, request sequence, length/alignment, mapping options and
  file descriptor/offset when applicable.
- Existing mapping reference: owner-scoped grant ID, generation, offset and
  length. Validate overflow, range, state and ownership at the responsible layer.
- Completion: request sequence, status, grant ID/generation, usable and backing
  lengths as needed, plus a designated capability slot or register at resume.

The libc translates a mapping pointer to its local grant record. An ID identifies
an object; it does not authorize access by itself. No domain pointer value is
passed to Linux as a Linux virtual address. Existing exchange offsets continue
to identify exchange buffers, and must not be confused with offsets in a grant.

The memory rows are compound services: Linux allocation/I/O plus monitor grant
operations. A raw `syscall(SYS_mmap, ...)` in the launcher would only create a
Linux mapping. Reuse the row transport and error conventions, while explicitly
describing this additional completion type. Remove supported operations from
the blanket MEMORY exception only when their contracts are implemented; do not
silently delegate the remaining unsupported VM operations.

### Acquisition transaction

1. Validate the requested operation and reserve tracking capacity. Allocate
   suitable backing, zero anonymous storage, and populate a private file copy
   before making the grant visible.
2. Remove temporary Linux user mappings before an exclusive transfer. The module's
   mapping/reference checks must prevent retained or newly created user aliases
   through dup, fork or concurrent mapping operations from bypassing exclusivity.
3. Have the monitor validate authority and geometry, retain the revocation root,
   and prepare delivery to the intended suspended domain/context.
4. Resume with an unambiguous capability completion. Libc moves the capability
   into domain-local mapping/pool state exactly once and clears the delivery
   slot before exposing the result or delivering application signal handlers.
5. Finish ownership bookkeeping. Success to the caller means the corresponding
   capability was received; a scalar success with an absent/stale slot is not
   a successful mapping.

Treat preparation, publication and consumption as explicit states even if the
first implementation combines some transitions. Couple the scalar completion
and capability to the same request. Decide whether an explicit acknowledgement
is needed from the actual delivery mechanism; do not add a round without cause.

Signal delivery may initiate nested syscalls or leave via longjmp. Pending
delivery must therefore be parked before callbacks and must not be overwritten
by a nested request. A committed allocation cannot be repeated merely because
a signal, EINTR or the existing RETRY protocol interrupted completion. Retry
must resume the same operation or return its recorded result.

Every failure/termination edge has an owner: free ungranted backing, or revoke
and reclaim a published grant. Device-file destruction must collect a grant
even if the domain never consumed its delivery. SIGKILL cannot depend on a libc
acknowledgement. Generation reuse must not admit stale completions.

### Release transaction

1. Libc identifies the grant and validates the supported range shape. Version 1
   supports a whole mapping only, with page-rounded length recorded at creation.
2. Serialize the release against acquisition, other operations and relevant
   execution contexts. Prevent new derivations/entries where the platform needs
   that to guarantee revocation completion.
3. Monitor revokes the grant's descendants. For private final reclamation, scrub
   and remove obsolete associations; for shared detach, preserve other grants.
4. Make the extent reusable only when its full reclamation contract permits it.
   Initially this may mean the retained pool, not Linux's page allocator.
5. Publish one completion and retire the local mapping record. Duplicate or late
   protocol messages cannot release a new generation occupying the same ID.

Before irreversible revocation, failure leaves the mapping live. Afterwards,
an error cannot be reported as though the old mapping were still usable; finish
cleanup under runtime ownership or retain it in a documented failed state.
Specify this boundary before coding the user-visible error path.

## 6. Mapping semantics and explicit limits

| Operation | Proposed behavior | Initial scope |
|---|---|---|
| MAP_PRIVATE + MAP_ANONYMOUS | Zeroed private region, capability-derived result | First mmap implementation |
| MAP_PRIVATE + file | Eager private copy, no writeback | After private grants and pool growth |
| MAP_SHARED + supported contiguous shared object | Same physical backing, explicit grants and Linux references | Separate sharing milestone |
| MAP_SHARED + ordinary memfd/shm object | Import only if backing meets the physical and lifetime requirements | No general support implied |
| MAP_SHARED + ordinary file | Refuse until a coherent backing/writeback contract exists | Deferred |
| munmap | Revoke a complete mapping; preserve unrelated grants | Partial ranges deferred |
| msync | Define per supported backing type and flag; no fake successful file sync | Implement when relevant backing exists |
| mprotect | Capability attenuation supplies a new view, not general mprotect | General range protection deferred |
| brk, mremap, demand paging, MAP_FIXED | No implementation promised by this plan | Explicitly unsupported |
| madvise and other VM flags | Audit and specify individually | Never acknowledge unsupported effects silently |

Choose and test refusal errnos per operation rather than treating all failures
as allocation exhaustion. Preserve the distinction between malformed requests,
unsupported semantics, permission failures and actual capacity exhaustion.
The existing implementation's ENODEV and EINVAL behavior is baseline evidence,
not a specification for every new case.

### Private file mappings

Eager copying preserves private-write isolation. It need not observe subsequent
file writes; it does not reproduce every Linux VM behavior. Define EOF and the
zero-filled final page, access beyond the file, truncation/growth, errors while
populating, and the point at which errors are returned. A sequence of pread calls
is not an atomic snapshot against concurrent writers.

Handle short reads and interruptions without publishing a partially initialized
mapping. Keep the source object alive until population is complete. File-backed
services that retain a dependency must retain the file object rather than reuse
an application fd number after close. Check requested rights against the source
and only issue capability permissions the target enforces.

The current delegated I/O path needs an ordinary-memory bounce buffer for some
backends because the exchange mapping cannot be pinned by the 9p transport.
Do not assume pread directly into a module/CMA mapping works for every backend;
qualify that path or copy from suitable Linux storage before exclusive transfer.

### Shared memory

Physical sharing supplies visibility of the same bytes, assuming supported
cache attributes. Applications still need synchronization. Cross-domain/Linux
atomics, futex identity, ordering and any required cache maintenance need their
own supported contract; coherent storage alone does not establish them.

Ordinary memfd/shm allocation does not guarantee a single contiguous physical
extent, and pinning does not make scattered pages contiguous. Candidate paths:

1. Export module-allocated contiguous storage as a Linux-mappable shared object.
2. Import an existing object only after proving its backing and lifetime fit.

These are not interchangeable with transparent support for every memfd.
`multiprocessing.shared_memory` additionally needs the POSIX name/open/size/
unlink lifecycle, attachment lifetime and interaction with native participants.
Names and Linux object semantics stay on the Linux side. Same-name objects must
resolve to the same backing, not separate private copies in each launcher.

Shared storage also needs a policy for capability-bearing data. The current
exchange region is a byte transport. Sharing arbitrary capability slots between
domains can transfer authority beyond ordinary byte sharing. State whether the
new object is a byte-oriented transport or permits capability transfer, and
prove the corresponding permissions and revocation behavior before exposing it.

### Why whole-buffer writeback is not ordinary MAP_SHARED

The initial idea of writing a copied region back on msync/munmap is deferred.
It loses coherence with native mappings and file writes. Writing unchanged bytes
can overwrite somebody else's newer data: a native writer changes a different
part of the file, then a domain flushes its older complete copy over that part.
Even a shared CMA copy among domains would still diverge from the ordinary
file's page cache and external mappings.

A separate explicit copy-and-writeback service could choose restricted ownership
rules. It must not silently acquire MAP_SHARED semantics by name. MS_SYNC also
needs completed synchronization to the backing file, not merely successful
buffered pwrite; supported MS_ASYNC/MS_INVALIDATE behavior and errors would need
definition. Do not rely on orderly munmap to preserve dirty data after process
termination.

### Why attenuation is not mprotect

A read-only derived capability restricts accesses through that capability.
Previously issued writable aliases remain writable. Revoking and issuing new
read-only pointers instead kills the application's old pointers; it does not
transparently change the permissions of those pointers as mprotect does.
Guard pages, PROT_NONE and RW-to-RX transitions remain separate problems.
Permission enforcement must be checked on the actual target, not inferred from
the existence of permission bits or from a fault with a different cause.

Other named constraints: backing must remain physically stable; allocation and
zeroing are paid up front; a free-RAM total does not guarantee a suitable extent;
there is no relocation promise. Partial munmap requires its own representability,
revocation and surviving-pointer design. A stale-pointer capability fault is the
hardware failure mode; Linux SIGSEGV delivery to an application handler is a
separate runtime contract, not implied by that fault.

## 7. Allocators consume pools

Put region acquisition below both malloc and public mmap:

```text
                        region acquire/release
                         /                 \
          allocator-owned private pools    application mappings
                 /           \
              level0        Sublet
```

Do not retain today's mmap -> malloc path when malloc starts requesting regions;
that creates a recursion. Application file/shared mappings do not automatically
enter malloc's pool inventory. Their contents and lifetimes belong to the caller.

For Sublet, use one buddy instance per suitable region, with an index that finds
the owning pool for a pointer. For level0, keep block allocation local while
obtaining and retiring its backing through the same region service. Level0 still
does not acquire per-object temporal safety merely by using revocable pools.

Metadata must grow with pools instead of preserving a compile-time whole-heap
ceiling. Specify a bounded bootstrap reservation and a recursion-free method for
allocating pool descriptors, capability slots and buddy tables. Handle metadata
failure by returning an acquired region without publishing a partial pool.
Return only idle allocator pools; live objects must not be invalidated merely
to reduce retained memory. Refill and release must account for nested signal
activity now and thread synchronization when that branch becomes a dependency.

Pool sizing and retention are allocator policy. Measure granule padding, buddy
slack, metadata, live grants, retained storage and allocation/reclaim cost
separately. Dynamic allocation removes per-port maximum heap guesses; it does
not eliminate bootstrap storage, configured budgets or startup stack sizing.

## 8. Implementation order and acceptance gates

These are planned gates, not results. Source the repository test environment
before builds/tests. Pin the module, monitor and emulator/RTL identities in
every result; a QEMU pass does not qualify silicon representability or timing.

### M1: private region acquisition and return during execution

Extend the module, monitor delivery/resume path, launcher and libc slot handling.
Use a small domain contract before introducing mmap or allocator policy.

- Yield, receive a region, write and read its endpoints, return it, then receive
  another region while the same application remains alive.
- Save aliases in other memory and suspended state; after return, those aliases
  must fault. Include a live-pointer control and verify the actual fault cause.
- Reuse the same physical extent for a new generation. Old pointers remain dead;
  the new owner sees sanitized storage, including no retained capabilities.
- Keep a neighboring region live and verify its data, permissions and aliases
  survive every operation on the returned region.
- Exercise owner mismatch, stale generation, duplicate release, wrong sequence,
  malformed/overflowing sizes, unsupported geometry and exhausted tables.
- Interrupt/terminate at preparation, publication and before consumption. Test
  signal-handler re-entry and retry without duplicate grants or slot overwrite.
- Repeat acquire/return beyond the slot count and verify bounded live resources.
  Inspect cache accounting separately; do not call retained memory a leak or
  claim it was returned to Linux.

### M2: anonymous mmap and whole-mapping munmap rows

Add the scalar row handling and capability-returning libc wrappers on M1.
Check zero fill, page-rounded lengths, actual grant bounds and permissions,
MAP_FAILED/errno, refusal paths and mapping-record exhaustion. Confirm that
munmap is effective revocation, including aliases outside the mapping. Keep
partial munmap and unsupported flags explicit.

### M3: growable level0 and Sublet pools

Replace fixed heap backing with independent pools and dynamic metadata. Force
growth beyond the old arena bound, verify older objects survive growth, force
metadata/allocation failures, and return idle pools. Run the affected allocator
and application contracts with both policies; preserve their distinct safety
claims. Record live and retained memory rather than a single arena-size number.

### M4: private file mappings

Implement population under Linux credentials and the chosen EOF/error contract.
Test offsets, short reads, empty/partial files, mapping permissions, failed I/O,
descriptor lifetime and private writes leaving the source unchanged. Compare
the supported subset with native behavior; record deviations explicitly.

### M5: shared backing and independently revocable grants

Prove Linux/D1/D2 share actual storage; detaching D1 kills its aliases without
changing D2's data or access. Exercise duplicate/open references, final release,
process death and the selected capability-data policy. Then add shared-object
names and suitable memfd/shm integration. Qualify native participants and
`multiprocessing.shared_memory` only after those complete contracts pass.

File MAP_SHARED, general mprotect, partial unmap and true return to the Linux
page allocator are separate follow-ons. They do not block M1 and must not be
implied by its success.

## 9. Decisions to close at the affected milestone

| Decision | Required before |
|---|---|
| Exact delivery slot/register, sequence binding and consume point; integration with signals and RETRY | M1 implementation |
| Retained revocation-root shape and release's irreversible boundary | M1 implementation |
| Geometry/permissions accepted on each target; overflow and capacity errors | M1 qualification |
| Cache return versus actual Linux page release, with distinct accounting | M1 result claims |
| Public mmap length/flag/range rules and refusal errnos | M2 |
| Pool metadata bootstrap, growth, lookup and retention policy | M3 |
| File population, EOF, truncation and I/O-error behavior | M4 |
| Shared backing provider, per-participant revocation and capability-data policy | M5 |
| POSIX shared-object lifecycle and native-process interoperability | memfd/shm compatibility claims |

The recommendation is to start M1 with one pending private delivery per context
and complete-region release, using existing retained backing. That bounds the
first protocol without making a promise about general shared mappings or Linux
page reclamation. The first concrete implementation task is to identify the
current module/monitor grant path and specify its prepare, publish, consume and
revoke transitions against the real resume assembly.

## 10. Alternatives and semantic references

Keeping file copies in the current static arena may be a temporary compatibility
step, but preserves the fixed-size problem and cannot provide coherent sharing.
Delegating page faults to the launcher would require a separate VM/fault design;
it is outside this initial proposal, but does not inherently require file
semantics in the monitor. Running domains with protected address translation
would also change the ISA/monitor boundary. Both are considered in the
[architectural alternatives](delegation-memory-options.md); neither is needed
to prove dynamic physical-region grants.

Primary references for the Linux semantics discussed here:

- [mmap and munmap](https://man7.org/linux/man-pages/man2/mmap.2.html): private
  writes, shared visibility, file extent behavior and unmapping.
- [msync](https://man7.org/linux/man-pages/man2/msync.2.html): synchronization
  completion and supported flag meanings.
- [mprotect](https://man7.org/linux/man-pages/man2/mprotect.2.html): protection
  applies to a mapped address range, not only to a newly returned pointer.
- [memfd_create](https://man7.org/linux/man-pages/man2/memfd_create.2.html):
  Linux object identity, backing and descriptor lifecycle.
- [Linux NOMMU mappings](https://docs.kernel.org/admin-guide/mm/nommu-mmap.html):
  a useful precedent for eager private file copies and the requirement for
  suitable contiguous backing for shared mappings. This is an analogy for the
  constraints, not a claim that this runtime is Linux's NOMMU implementation.
