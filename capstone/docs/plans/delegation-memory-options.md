# Memory beyond physical grants: views, translation and alternatives

Status: DESIGN EXPLORATION, 2026-09-30. These are candidate extensions and
reasoned consequences, not implemented instructions or measured costs. The
[physical-grant plan](delegation-memory.md) remains the executable starting
point. This document reopens its deferred architectural choices and states what
each choice must prove. No particular ISA extension has been selected.

## 1. What we want to preserve

Linux selects backing and performs file/storage operations. A domain obtains
authority by capability derivation, with object bounds and lifetimes enforced
at access time. The monitor understands ownership, permissions, extents and
protected continuations; file names, descriptors and writeback policy stay in
Linux. Whether the capability's cursor must always be a physical address is a
separate design choice, now explicitly open for exploration.

Capstone's existing physical authority model matters when making that choice:
Linux page tables must not let Linux read a private transferred region, redirect
a domain's old pointer into somebody else's allocation, or recreate exclusive
ownership while other authorized sharers still exist. Allocation bookkeeping is
not memory access authority. In the current architecture, Linux's ordinary
virtual mappings are also subject to the CPMP capability check. A surviving
kernel direct-map PTE alone must not authorize access after exclusive transfer.

The published Capstone model deliberately uses physical addressing to avoid
depending on MMU behavior for its security argument. Adding translation would
therefore need a new composition argument, even if capabilities remain central.
See the [Capstone model](https://arxiv.org/html/2302.13863v2), sections 2.3--2.4.

## 2. Three constraints that runtime wrappers cannot remove

**A contiguous C object needs a contiguous address interval.** If its pointer
arithmetic directly produces physical addresses, that interval must correspond
to contiguous physical storage. Several smaller pools help separate allocations,
but cannot turn one scattered 1 GiB mapping into an ordinary contiguous `char *`.
A vector of capabilities is useful, but it changes the access interface unless
hardware or the compiler performs the selection on every access.

**Changing a new pointer does not change old copies.** Transparent mprotect
requires access-time policy shared by existing aliases, or an equivalent global
rewrite mechanism. Revocation only answers whether a pointer is still valid;
it does not make an old writable pointer remain usable for reads alone.

**Sharing bytes does not separate grant lifetimes.** A and B can have the same
backing and still need different rights and different unmap events. Duplicating
one non-linear capability does not create independently revocable attachments.
The existing MREV accepts a linear source; merely allowing it on a shared source
would make its reclaim semantics unsafe if it returned exclusive authority while
other sharers survived.

The local ISA source describes this in `parts/cap-man-insn.adoc` under MREV and
REVOKE. The runtime's [Sublet primitives](../../runtime/include/sublet/sublet.h)
and [merge analysis](../design/capability-merge-primitive-proposal.md) also show
why resource ownership, alias lifetime and allocator policy must be distinguished.

## 3. Candidate A: keep physical pointers and improve the region service

This requires no new addressing model. Use repeatable grants, several allocator
pools, eager private file copies, supported contiguous shared backing, and a
precise return/reclaim operation. Linux-owned names and a suitable shared-object
provider can be implemented around module-allocated storage.

A dedicated provider could allocate large aligned extents at object creation
and export them through a Linux fd. This is easier than making every existing
memfd physically contiguous after unrelated processes have used it. It still
needs coherent object identity, rights, sizing, unlink and final-reference
semantics. Do not describe such a provider as transparent arbitrary memfd import.

Useful changes to the current implementation are a complete cached-extent
release protocol, dynamic monitor bookkeeping, efficient tag-aware scrubbing,
and allocator coalescing where the ISA permits it. These reduce limits and cost;
they do not implement transparent range protection or join scattered storage.
Scrub acceleration must retain confidentiality: forgetting tags alone is not
equivalent to clearing the bytes before a new owner can read them.

Best fit: large enough contiguous pools and a deliberately documented mmap
subset. It is a practical implementation target even if a later architecture
supports virtual mappings.

## 4. Candidate B: add a protected view with mutable access policy

A **view** is a participant's authority to access some backing. Backing identity,
view identity and allocator/object revocation identity are separate. This is a
proposed abstraction; the name does not designate an existing Capstone type.

Conceptually, a derived pointer carries:

```text
view identity + generation, cursor/bounds, maximum permissions, object lifetime
```

The view supplies its current allowed ranges and permissions. Every access must
pass all checks:

```text
valid pointer and object lifetime
AND live view/generation
AND access within pointer bounds
AND pointer maximum permissions permit the operation
AND current view range permissions permit the operation
```

Initially the cursor could remain physical and the view could cover one physical
extent. No page translation is necessary to explore view-local detach and rights
changes. A whole-view permission mask is enough for a first prototype; arbitrary
partial mprotect/unmap additionally needs range or page state.

### What the abstraction buys

- D1 and D2 can have distinct views over the same backing. Revoking D1's view
  kills its aliases while D2 remains live. Sharing D1's pointer onward carries
  D1's view lifetime; creating a genuinely independent view requires management
  authority over the backing.
- Changing a view from RW to R affects old pointers on their next access, even
  if their immutable maximum permissions still include W. A later RW setting
  can restore writes within that maximum. A pointer explicitly attenuated to R
  remains R. Initial mapping protection therefore needs to be distinguished from
  the maximum rights that management is authorized to restore.
- Object revocation still ends malloc lifetimes independently of mapping
  protection. A physical object reachable through several views needs a common
  object lifetime or explicitly defined sharing semantics; creating another
  view must not reset its revoked object identity.

Management uses a distinct control capability. An ordinary data capability must
not let its holder restore write permission or revive a revoked view. Killing a
view is irreversible for that generation; changing protection is reversible
within the control capability's ceiling.

Revoking one view does **not** return a linear capability to the backing. Final
reclamation must first eliminate every conflicting view and access path, then
perform the existing initialization/scrub obligations. Physical aliases that
bypass the view gate cannot coexist with a promise that the gate controls all
application access through that view.

This needs a distinct view-invalidation rule. Applying existing REVOKE unchanged
to overlapping authority is not sufficient: its exclusive-reclaim contract and
the architectural alias invalidation rule are precisely what must not revoke
the surviving sibling view. The original physical root remains responsible for
final ownership recovery; the new view operation only withdraws access.

### Why it is not one harmless extra opcode

The shared policy must be consulted on all relevant data and instruction paths,
including atomics and capabilities restored from a saved context. Cached policy
needs invalidation when it changes; updates need a completion boundary across
active harts and in-flight accesses. Backing reuse cannot happen before that
boundary. Instruction permission changes also need the appropriate instruction
fetch/cache synchronization.

Existing revocation metadata is a candidate place to associate a view, but a
revocation-node ID is not automatically a view ID: allocator derivations create
more nodes, identities are reused, and protection changes must preserve pointers
that revocation would kill. An inherited association/cache might avoid widening
the pointer representation; this needs an encoding and implementation proof.

Partial unmap adds a separate temporal issue. Marking a page absent and later
present in the same view can revive old pointers covering that page. Either
define that conventional mapping behavior explicitly, or add range-lifetime
identities/restrict reuse. Do not claim full stale-pointer protection from a
whole-view generation alone. Whole-view destruction and fresh generation reuse
are the simpler first contract.

## 5. Candidate C: put protected address translation behind a view

Let the pointer identify a logical interval, and let a protected map bind its
pages to backing capabilities:

```text
pointer -- bounds / lifetime / view rights --> logical page
logical page -- authorized map entry --> backing page + page offset
```

A map entry must be derived from real authority over its backing, not from a
physical address supplied as an integer. Every mapping mutation requires the
proper management capability. Exclusive physical backing is consumed/held by
the mapping object so that two virtual aliases cannot manufacture two supposed
exclusive owners of the same frame. Shared aliases require shared backing
authority. Logical linearity alone is insufficient for physical exclusivity.

This makes scattered pages usable as one contiguous application mapping. It also
allows per-page presence/permissions, stable logical addresses during authorized
relocation, and potential demand paging. Existing fine bounds still restrict
objects within those pages. A protected view identifier must survive derivation,
copying and spills; it cannot be selected by an untrusted current-domain setting
that reinterprets a received capability in another address space.

Two implementation choices deserve comparison:

1. Reuse the existing page-table walker/TLB, with protected mapping authority and
   defined capability checks before and after translation.
2. Use a view-indexed page/extent table, initially with large pages or a small
   number of extents and a translation cache.

The second is still address translation. Calling its entries capabilities or
scatter/gather segments does not eliminate table lookup, cache invalidation,
address-space identity or page-boundary handling. The first can reuse hardware,
but ordinary Linux PTEs alone cannot be the complete authority if Linux remains
outside the domain's confidentiality/integrity trust boundary.

CHERI demonstrates that fine-grained capabilities and conventional virtual
memory can coexist. It does not by itself establish the required Capstone
linearity, revocation or hostile-remapping guarantees. See the
[CHERI architectural goals](https://github.com/CTSRD-CHERI/cheri-specification/blob/main/chap-architecture.tex).

### Files, dirty state and movement remain real work

Mapping the actual pages of a Linux shared object can supply coherent bytes
without whole-buffer copies. Linux must retain stable references and prevent
unsupported migration/truncation while those mappings exist. Long-term pinning
has restrictions; arbitrary files/backends are not automatically eligible.
See the [kernel page-pinning contract](https://docs.kernel.org/core-api/pin_user_pages.html).

Writes through the domain's translation path do not automatically set the dirty
state of Linux's PTEs or page cache. Define dirty reporting/write protection and
a race-free clear/rearm protocol with Linux writeback. A hardware dirty bit
alone is not a complete msync implementation. Mapping shared cache pages plus
reporting dirty ranges is fundamentally different from copying an old buffer
over a file. The monitor only handles page authority; Linux owns file policy.

Moving a page containing linear capabilities cannot be an arbitrary memcpy:
copying and publishing tagged contents can duplicate authority. The trusted
move protocol must quiesce affected access, transfer tags/values correctly,
update the mapping and invalidate old translations before the old frame is
reused. Copy-on-write and fork of capability-bearing storage need explicit
semantics for linear values too. These are follow-ons, not automatic benefits
of translation. Keep physical storage resident in the first prototype.

## 6. Demand paging can keep the monitor file-blind

The earlier reason to exclude fault delegation was too broad. A monitor does
not need to know that a faulting page belongs to a file. With a suitable mapping
model it can deliver an opaque event such as:

```text
(view ID, generation, page index, access kind, request sequence)
```

Linux resolves the event against its own memory/file object and offers backing.
The monitor checks the supplied authority and the authorized mapping transition,
then resumes the protected continuation. A fault requesting an absent page must
remain distinguishable from an invalid capability, revoked view or prohibited
access; the latter cannot be fixed by granting more authority.

An OS that cannot be trusted with private contents cannot freely redirect or
copy private pages. Pager responses must be tied to the correct object, offset
and pending request, and map mutation must obey its authority contract. Swapping
private tagged memory through untrusted storage additionally needs a protection
and restoration design for bytes, tags, linear identities and freshness.

This needs blocking/wakeup, cancellation, restartability, nested-fault handling
and a defined concurrency protocol. Reserve resident code, stack and metadata
for the fault handler so it can make progress. The important architectural
point is that a file-blind monitor can broker a fault; that does not make the
rest of paging cheap or make it work with unchanged direct physical pointers.

## 7. Other approaches

| Approach | What it makes easier | Limit or price |
|---|---|---|
| Many small physical pools | Growing heaps despite lack of one huge free extent | Each ordinary object still needs contiguous storage |
| Chunked buffers / vectors of capabilities | Large files and shared data with explicit segment access | Changes application interfaces; not transparent mmap |
| Dedicated contiguous shared-object provider | Real shared bytes with the present addressing model | Linux integration and constrained object geometry |
| Compiler-managed handles with software translation | Prototype views/scattered backing without new silicon | Instruments accesses; atomics, assembly, JITs and foreign calls need an ABI; trusted checks must resist bypass |
| Existing MMU plus Capstone checks | Reuses paging machinery and Linux-compatible address intervals | Requires protected mapping authority and revised physical-exclusivity proof |
| Coalescing/merge instruction, bulk scrub, larger tables | Allocator policy, reclamation cost and capacity | Does not supply virtual addressing or retroactive permissions |

An IOMMU by itself does not translate the CPU's ordinary C-mode loads; its device
address translations are not a substitute for a CPU access-path change. A single
base/offset translation register also cannot join arbitrary scattered pages.
Faulting and emulating every access is another software experiment, with the
same semantic obligations and potentially substantial per-access overhead.

## 8. Proposed next experiments

Keep the physical-grant work as the implementation baseline. Explore views in
a separate model/emulator experiment before choosing an ISA representation:

1. Create two views of one contiguous backing object. Save old writable aliases
   in each. Change A to R: A's read succeeds, A's write faults, B's write works.
   Restore A's allowed W: its old non-attenuated pointer works again, while an
   explicitly R-only pointer still cannot write.
2. Revoke A and keep B live. Reuse A's identifier with a new generation; A's old
   pointers remain dead. Attempt exclusive reclamation while B is live and
   require rejection. Repeat with spilled pointers and paused contexts.
3. Bind three nonadjacent physical pages to one logical interval. Exercise normal
   pointer arithmetic across boundaries and atomic/capability access cases.
   Substitute an unauthorized page or stale map response and require rejection.
4. Change rights or withdraw backing while another context accesses it. Verify
   that completion means the prohibited accesses can no longer retire, including
   through cached translations and alternate physical paths.
5. Integrate one supported Linux shared backing object. Verify coherent native
   and domain accesses, dirty reporting, explicit sync and lifetime on process
   death before making claims about general file mappings.

Use positive controls for every denial. Measure metadata per view/page/object,
access-path lookup cost, cache misses, update/shootdown cost, switching, and
revocation separately. No size/cycle advantage is established by this document.

These experiments establish semantics; they do not require shipping a new
physical-view ISA before investigating the existing MMU. The long-term
recommendation below prioritizes that investigation.

## 9. Recommended long-term architecture

For general Linux applications, the recommended endpoint is **capability-authorized
virtual memory using the existing MMU and TLB machinery**, while retaining
Capstone's physical ownership and object-revocation guarantees. This is an
architectural recommendation, not an accepted ISA change or a claim that the
integration exists today. Physical grants remain the first implementation.

The useful abstraction from candidate B is the mapping's independent lifetime
and permission policy. It need not become a second general translation mechanism
beside the MMU. Candidate C, implemented through the existing walker and protected
mapping operations, is the preferred direction. Exact capability encoding,
translation-context binding and caching still require design and measurement.

### Responsibility boundaries

| Layer | Responsibility |
|---|---|
| Linux | Allocate backing, select file/page-cache objects, manage storage and writeback, service eligible missing-page requests |
| Monitor and hardware | Admit mappings from actual backing authority, protect translation state, enforce grant rights/lifetimes and physical exclusivity, complete invalidation before reuse |
| Domain | Choose application mappings within granted authority, derive bounded object pointers, implement malloc/free and object revocation |

Linux may propose a frame and a mapping; a scalar physical frame number is not
authority to install it. The monitor's operations remain generic: bind authorized
backing to a logical range, change permitted access, detach a grant, reclaim
backing. They do not interpret an fd, filename, MAP_SHARED or storage policy.
The access rules must be enforced by hardware and protected capability state;
software bookkeeping alone must not replace Capstone's physical guarantees.

An application can use a protected virtual address space with capability-isolated
domains inside it. A separate page table for every small domain is not inherently
required. Capability transfer across different address spaces needs a defined
binding or authorized rebinding protocol: interpreting a received pointer under
the recipient's arbitrary current page-table root is unsafe.

Conceptually, an access needs all of the following:

```text
live object capability and grant
AND bounds and capability permissions
AND current mapping permissions
AND authorized translation to the physical backing
```

Linux must neither retarget an old pointer into an unrelated object nor regain
access to exclusively transferred frames through its direct map. Two virtual
names for the same frame do not establish two exclusive physical owners.
Mapping-control authority must be distinct from ordinary data access authority.

### Why this is the preferred endpoint

A contiguous virtual interval can cover scattered physical pages, removing CMA
contiguity as the general application-size constraint. Page permissions provide
the mechanism needed for mprotect on existing pointers. Distinct mappings can
share actual Linux backing pages, and translation permits stable addresses across
authorized backing changes. Fine-grained bounds and allocator lifetimes remain
capability responsibilities. Large pages can amortize translation without making
object protection page-granular.

The established precedents support the choice of mechanisms:
[CHERI](https://github.com/CTSRD-CHERI/cheri-specification/blob/main/chap-architecture.tex)
composes fine-grained capabilities with conventional MMUs, while
[seL4's mapping interface](https://docs.sel4.systems/Tutorials/mapping.html)
uses frame and address-space capabilities to authorize hardware mappings.
Neither precedent supplies Capstone's complete proof against hostile remapping,
or proves this proposed implementation. Reusing paging machinery reduces the
amount of new machinery to qualify; it does not make the composition mature by
itself. The published Capstone physical-address model needs an explicit extension.

### Conditions before calling the integration mature

1. **Authority and lifetime:** prove physical exclusivity across aliases,
   independently detachable shared grants, address-space binding, and stale
   pointer rejection after address/identifier reuse. Specify partial-unmap
   semantics; a PTE being absent temporarily is not temporal safety.
2. **Completion:** define and test when protection changes and revocation have
   taken effect across harts, cached translations, instruction fetch and saved
   contexts. A backing frame cannot be reused before that boundary.
3. **Linux backing contract:** integrate references, invalidation/truncation,
   eligible pinning and dirty reporting. Native shared-file writeback requires
   this contract; a page-table walker alone does not implement msync.
4. **Tagged storage:** specify tag-preserving movement without duplicating linear
   authority. COW and fork need explicit capability semantics. Paging private
   contents through untrusted Linux additionally needs protected storage or a
   narrower no-swap contract; generic fault forwarding does not provide secrecy
   or integrity for stored page contents.

The practical sequence is dynamic physical grants first, then a model/emulator
prototype of protected mappings through the existing walker with resident pages.
Use the lifetime and remapping attacks above as acceptance tests. Follow with
one real Linux shared backing object and its invalidation/dirty protocol. Add
demand paging, page movement and capability-aware COW only with their own
contracts. Full Linux compatibility is not a prerequisite for the first useful
stage and is not claimed by choosing this endpoint.

If changes to Capstone's addressing model are ruled out, candidate A is the
recommended bounded product: dynamic physical pools with explicit mmap limits.
For the broader long-term endpoint requested here, a custom scatter/gather map
would inherit most MMU obligations while adding another implementation to mature.
It should displace the existing-MMU direction only on concrete evidence of a
better security, complexity or performance result.
