# Native grant transactions and fault classification, draft 1

`runtime/virtual/native-grant.h` owns the versioned fault/mapping interface.
The native context lane supplies authenticated events and the tag/page lane
supplies page-release clearing. These interfaces are not a frozen integration
ABI. The normal adapter keeps the mapping prototype compiled out and the
anonymous fault optimization disabled by default.

## Fault admission

Only a protected event of kind 4 with cause 12, 13 or 15 can request ordinary
paging. Capability tag, type, bounds, rights and liveness failures never enter
the pager. Classification is necessary but insufficient: the module checks
its owning address space, live grant covering the address, access permission
and acceptable VMA under the namespace lock and mmap lock. The namespace lock
stays held across a sleeping fault. A backend which drops it must revalidate
the namespace, grant identity/generation and authority on every retry.
Restart uses the protected event's original PC; scalar callers cannot replace
that PC or turn a capability failure into a page fault.

## Transaction contract

An operation identifies a protected mapping record by namespace, monotonically
allocated mapping ID, exact address and exact length. No address alone names
authority. Threads share a namespace; distinct processes never share its
mutable generations. Stale IDs and foreign namespaces fail before effects.
Transport holds, protected frames, active copies and in-flight faults must
exclude retirement. Lock order is namespace exclusion then mmap write lock.

Preparation may allocate and fail. Until the commit boundary, every error
leaves mapping, authority and descendants unchanged. After retirement succeeds,
unmap and record removal must complete without allocation, interruption or
an error return. Re-minting cannot roll back retirement. There is no injection
point after retirement which returns failure: such a hook would violate the
contract it purported to test. Simulated observers must see either the old
mapping/grant pair or neither, never a mismatched pair. Internal transitional
states are inaccessible while both locks are held.

The Linux 6.1 prototype exports `capstone_grant_unmap`. It requires one exact
private anonymous VMA, with no locked, special, huge, growing or userfaultfd
policy. It prepares the maple update and detached tree before invoking the
retire callback at the existing unmap point of no return. Callback failure
has no authority effects and frees only temporary allocation. Callback success
continues through the already non-failing PTE removal and VMA freeing while
retaining mmap write exclusion. The platform's tag-aware `arch_free_page`
remains mandatory; this helper adds no alternate page-release path.

The module prototype is deliberately restricted to unstarted contexts, so
the launcher's transport and mapping registries cannot be made incoherent.
It uses an input-only ioctl; no result copy can report failure after commit.
Enabling it in running native tasks requires moving all transport holds and
mapping consumers into the module registry. `MAP`, `PROTECT`, `REMAP`, partial
operations, fixed replacement, shared/file mappings and unknown requests are
unsupported and return an error without effects.

## Remaining map design

A map helper must reserve an unused VA, VMA, page-table policy, accounting and
maple insertion resources before minting; the new VMA stays unpublished. The
module must reserve its ID, arena record, lifetime nodes and protected reply
slot before minting. Mint failure releases only prepared resources; successful
mint is followed by non-failing insertion/publication under both locks.
Mapping first and rolling back with ordinary `vm_munmap` is invalid. A later
native map interface must include authenticated result delivery so a failed
copyout cannot strand an inaccessible grant. This map helper is not implemented
by the unmap prototype and remains an integration blocker.

## Evidence boundaries

The executable double models namespace exclusion, identity, lifetime and the
prepare/commit contract, including deliberately broken variants. It does not
prove hardware tag preservation or Linux MM behavior. The separate kernel
prototype must be built and exercised with preparation failures and failed
retirement, plus negative controls for retirement ordering and stale authority.
Cross-model review, both-host gates and matched census evidence are required
before any landing. Unsupported operations remain refused throughout.
