# Caplified mapping tables: translated domain memory without a trusted mapper

Status: DESIGN CANDIDATE, 2026-09-30. The translation mechanism is not implemented
or hardware-qualified. A bounded executable contract model is available (§10.1).
The document fixes a small, checkable first stage of translated
domain memory and states what later stages must prove. It is the outcome of a
review exchange over the [physical-grant plan](../plans/delegation-memory.md),
the [alternatives](../plans/delegation-memory-options.md) and the
[addressing design](delegated-memory-addressing.md); §11 lists what it changes
in those documents.

Names follow Linux where the meaning matches: mapping (a range with fixed
flags, as a VMA), PTE, table entry, root table, populate, unmap, destroy,
PRIVATE and SHARED, max protection (Linux's `VM_MAYREAD`/`VM_MAYWRITE`), and
the PTE states none, present and locked. Where Linux has no such concept the
name is new: frame handle, detach handle, teardown token, and the locked
state itself, which exists because this design does not reuse addresses.

## 1. Scope

The candidate gives a domain a logical address space over scattered physical
frames, with page tables whose entries are capabilities. The monitor drives
the mapping operations. Hardware rules limit what those operations can do, so
that the monitor gains no data authority beyond what a revocation-root holder
already has: it can withdraw access and it can stall; it cannot read delegated
contents, redirect live pointers or duplicate exclusive authority.

Stage 1 covers PRIVATE, resident mappings and five transitions. Shared
backing, repopulation after eviction, permission changes, partial unmap,
physical exports, demand paging and file mappings are later stages with their
own conditions (§9). Nothing in Stage 1 is claimed for them.

The idea of capability-bearing protection structures is Caplification's
Figure 1(d): "adapting existing memory protection data structures (e.g., page
tables) to be capability-based". CapliFive applied it to PMP; this candidate
applies it to a domain's page tables. The step to translated Capstone
capabilities with two revocation trees is additional work whose security the
paper does not establish.

### 1.1 Residency and fault assumptions

The current runtime uses physical grants; this candidate is not an implemented
demand pager. In Stage 1, populated private data pages, table pages and the
metadata required to check capabilities remain resident while in use. Linux
must not transparently swap or migrate their backing. Revocation under §5.2
can withdraw access, but does not preserve a page for later restoration.

CREATE reserves logical authority without data backing. POPULATE can supply
a PTE that is still none later, including after the domain has split the
mapping capability. Turning this into allocation on first access additionally
needs a fault-delivery, allocation and restart protocol. Stage 1 does not
supply that protocol. An absent page whose PTE is none must be distinguished
from a locked PTE, a revoked capability, a wrong binding or a permission
failure; those failures must not be repaired by allocating memory.

Unmapping a PRIVATE page locks its PTE for the generation. Neither DETACH nor
frame revocation is a swap-out operation, and POPULATE cannot restore that
PTE. Memory pressure therefore requires allocation failure or explicit
withdrawal, rather than implicit eviction and refill, in this stage. A future
pager also needs resident handler code, stack and essential metadata so that
handling a fault does not depend on paging in the handler itself (§9.6).

## 2. Three separate decisions

| Decision | State |
|---|---|
| D1 PTE representation: page-table entries are capabilities, checked by the walker | Taken for this candidate |
| D2 Table ownership: who writes the tables | Monitor, for the first prototype. The rules in §6 make its power equal to revocation. Domain-managed tables would need the same hardware rules plus an argument that the component holding the root table is not in every other component's trusted base; not pursued here |
| D3 Context semantics: what a logical capability means in a context other than the one it was created in | Taken: capability-selected root, §9.4. The capability's binding (id, gen) selects the registry entry and through it the table; the executing context is irrelevant. The ambient alternative is recorded there with the reason it was rejected |

These are independent. D1 without D2 is possible; D3 does not follow from
either.

## 3. Security goal and trust

**Goal.** Even a malicious mapping monitor, holding no matching data authority,
can neither read delegated contents, nor redirect live pointers of a PRIVATE
mapping to other contents, nor duplicate exclusive authority. It can withdraw
access or prevent progress. A SHARED mapping (§9.1) promises a view on a Linux
object whose bytes may change; the redirection clause does not apply to it, the
other two do.

| Party | Trusted for | Not trusted for |
|---|---|---|
| Hardware | The invariants of §6 and the completion protocol of §8 | — |
| Monitor | Availability: it may revoke and stall | Confidentiality or integrity of mapped memory |
| Linux, module, launcher | Frames, files, names, storage policy | Anything inside a domain's private memory |
| Domain libc | Its own pointers, pools and logical address policy; its round code holds the META and exchange capabilities and is trusted by the domain's components for syscalls, as today | Other domains' memory |

Two qualification boundaries stand apart from this design. capstone-qemu keeps
a granule's tag across a Linux-side store, so any statement about shared,
possibly tagged memory waits for that fix. R-44 leaves the S/U-mode CPMP
tracker adopting unseen node ids, so any statement about Linux-side isolation
on the affected RTL waits for R-44. Neither blocks modelling the mapping
invariants.

## 4. Objects and capability kinds

**Physical data capability.** Today's capability: cursor and bounds are
physical addresses.

**Logical data capability.** Cursor and bounds are addresses in the logical
space of one mapping. It carries the mapping id and generation (id, gen). The
physical/logical distinction is unforgeable and inherited by every
derivation. Whether it is an inline bit or protected metadata is an encoding
question to settle after the semantics; nothing below depends on it.

**Mapping capability.** The logical data capability CREATE delivers to the
domain: linear, bounds equal to the mapping's range, rights equal to its max
protection, bound to (id, gen). It is what `mmap` returns in this runtime, as
in CheriBSD, and the domain derives every pointer into the mapping from it.

**Page-table capability.** Linear. Names one table page. Not dereferenceable by
ordinary loads and stores; used in place by the walker and by the mapping
instructions only. It is produced only inside CREATE and POPULATE, from an
exclusive physical page the monitor supplies, linear or UNINIT, with write
permission retained for initialization and later table updates: the instruction
consumes the page, writes every slot to none, which clears any capability tag
the page carried, and only then publishes the page as part of the table.
Requiring UNINIT pages would not be enough. UNINIT forbids the holder's reads
before its writes and says nothing about what the page contains (Capstone
§4.3), so a capability the monitor stored there earlier, or one a previous
owner left, would survive as a slot content and translate without ever passing
POPULATE's class check. The conversion is the same for the root page CREATE
takes and for the pages POPULATE takes later; CREATE takes no other page, and
every further table page enters through POPULATE (§5.1).

A table entry, the entry of a non-leaf level, is a page-table capability; it is
linear too, so a table page has exactly one parent slot. The root table is
linear as well and is held in exactly one place: the protected registry entry
for its id, which CREATE fills and which the walker reaches through a logical
capability's binding (D3, §9.4). No operation installs, moves or uninstalls it;
the entry holds it until DESTROY frees the entry, and a revocation of its page
(§5.2) invalidates it in place. The executing context is irrelevant to
translation. Table ownership (the monitor), the translation context the
capability selects and the executing domain context are three different things;
one holder of the root table does not imply one executing context.

**PTE.** A physical frame capability stored in a leaf-level slot. Used in
place by the walker. It leaves the slot only through UNMAP or through
revocation from above (§5.2); it cannot be loaded.

**Frame handle.** The senior revocation handle the monitor keeps over every
frame it received from the module, as the runtime ownership contract already
requires. Revoking it withdraws one page (§5.2).

**Table-page handle.** The senior revocation handle the monitor keeps over the
page a table page is made from, created before CREATE or POPULATE converts the
page; the same kind of object as the frame handle. Revoking it is the
environment transition of §5.2 and the only way a table page ever returns.
DESTROY returns nothing (§5.1).

**Detach handle.** Created by CREATE, senior to the mapping capability, held by
the monitor. Its operations are POPULATE, which uses it to name and authorize
the mapping and leaves it in place, and DETACH, which consumes it. It is not a
general revocation handle: a plain REVOKE on it would return a readable LINEAR
capability when the domain had delinearized the mapping capability (Capstone
§4.3), and the monitor could then read through PTEs that still translate.

**Teardown token.** Produced only by DETACH. Bound to (id, gen). Grants no
data access and cannot reactivate the mapping. An UNMAP with another
mapping's token is rejected by the binding; a repeated UNMAP of one PTE by the
PTE state (locked). The token is linear so that it has one holder, which is a
convenience, not what enforces either rejection. Consumed by DESTROY. Whether
it is a new capability type, a sealed variant or protected state is open.

**Mapping** = (id, gen, class, max protection, root table, PTEs, state). The
state is protected and takes the values ACTIVE (from CREATE), DETACHED (from
DETACH, the teardown token exists) and DESTROYED (from DESTROY). POPULATE
requires ACTIVE. It does not require that the mapping capability still exists:
SPLIT replaces a capability by its parts (Capstone §4.4), and those parts need
backing as much as the whole did. The class is fixed at CREATE and never
inferred from the frames later supplied. The id indexes the registry, protected
mapping state of bounded size, one entry per id holding gen, class, max
protection, root table and state; CREATE reserves the entry and only DESTROY
frees it. gen is a finite counter per id: an exhausted id is retired, never
wrapped, as the revocation-node generation field on the RTL already is (R-35).
Stage 1 defines one class:

| Class | Accepts | PTE after exit | UNMAP yields | Stage |
|---|---|---|---|---|
| PRIVATE | Linear frames only | locked | UNINIT(F) | 1 |
| SHARED | Non-linear frames only | refillable | nothing | 2 |

PRIVATE is Linux's `MAP_PRIVATE` as the domain sees it, without copy-on-write
and without `fork` (§9.5).

**PTE states.** none (never used), present, locked (unmapped, or revoked from
above). Table-entry slots have the same states. A slot is locked either because
UNMAP marked it so or because it holds a capability whose node is dead; the
walker treats both alike, so a revocation from above (§5.2) does not have to
find the parent slot to lock it. SHARED adds dead, a refillable state, in Stage
2 (§9.1).

## 5. Stage 1: PRIVATE, resident mappings

### 5.1 Transitions

CREATE binds, POPULATE supplies backing, DETACH ends accesses, DESTROY ends the
identity; physical reclamation is separate and goes through the frame and
table-page handles (§5.2).

| Transition | Before | After | Ends | Completion enforced by |
|---|---|---|---|---|
| CREATE(id, range, root page, max protection, PRIVATE) | One exclusive physical page, linear or UNINIT, for the root table, held by the monitor; the registry entry for id free | The mapping (id, gen), ACTIVE, in the registry entry for id; the page converted into the root page-table capability with every slot none (§4) and held by the entry; the detach handle at the monitor; the mapping capability, linear, logical, empty, rights ≤ max protection, delivered through the resume slot | The page is consumed | Instruction; gen is fresh for id; rejects an id whose entry is in use |
| POPULATE(detach handle, v, F, table pages) | The detach handle names the mapping, ACTIVE; F linear, physical, held by the monitor; PTE(v) none; one exclusive physical page for every level of the path whose entry is none | PTE(v) = F, present; table entries on the path present, the supplied pages converted with every slot none (§4) and published; the detach handle unchanged | F and the supplied pages are consumed; the monitor keeps nothing | Instruction, all or nothing: it rejects, consuming nothing, a non-linear or non-physical F, a PTE that is not none, rights above max protection, a mapping not ACTIVE, and a path that needs more pages than were supplied; no half-published path exists |
| DETACH(detach handle) | The detach handle; mapping ACTIVE | The teardown token at the monitor; mapping DETACHED; no descendant of the mapping capability's node exists | The mapping capability and every logical capability derived from it | The protocol of §8 in full, on every hart, for every access through a capability of (id, gen); R-45's fix covers only the issuing hart's younger instructions (§8); the returned authority is converted to the token inside the instruction and never exposed |
| UNMAP(v, token) | Token bound to this mapping; mapping DETACHED; PTE(v) present | UNINIT(F) at the monitor; PTE(v) locked | TLB entries and walk results for the PTE's node | Break-before-make with drain, §8; rejects a token bound elsewhere and PTEs that are none or locked |
| DESTROY(token) | The whole-mapping token, not a Stage 3 range token (§9.2); mapping DETACHED | Mapping DESTROYED; token consumed; the registry entry for id freed for a new generation; PTEs, table entries and table pages stay where they are and return through their frame and table-page handles (§5.2), before or after this step | Every binding to (id, gen); the generation is retired | Node and generation discipline: gen is never reused for id; a repetition is rejected by the mapping state; the instruction has no output, waits for nothing and has nothing to interrupt or resume |

DESTROY ends the identity only. It does not require that PTEs or table entries
are gone, and it could not check that: after a revocation from above, valid
PTEs can sit under a table page the tree no longer reaches. Frames and table
pages left behind, reachable or not, return through their frame and table-page
handles, one REVOKE each, yielding UNINIT, before or after DESTROY. UNMAP
remains the optional orderly return for frames still reachable before DESTROY,
without invoking REVOKE; UNMAP still obeys the completion protocol of §8.
A traversal from the root could not be the
return path: a page revoked from above cuts its subtree off the tree while the
subtree's pages stay valid, so a traversal would miss them, and a page returned
once through its handle and again by traversal would be two overlapping UNINIT
capabilities, which breaks exclusivity. One handle per page, kept by the
monitor as it keeps one frame handle per frame, reaches every page exactly once
whatever the tree's state; those handles are the management metadata the
mapping needs beyond the tree itself. The sequence CREATE, revoke the root
table's page, DETACH, DESTROY therefore ends the generation, and the remaining
pages come back through their handles as they would have anyway. What makes
this separation sound is DETACH's completed drain, the absence of any data
return from DESTROY, and the fresh generation every later CREATE for the id
gets.

The monitor scrubs UNINIT(F) by writing it, the existing scrub obligation,
before the frame returns to Linux. UNINIT is what forces the write.

### 5.2 Environment transitions

**Revocation of a mapped frame from above.** REVOKE on the frame handle while F
sits in a PTE: the PTE becomes locked (§4), TLB entries of the PTE's node are
invalidated and the accesses of §8 are drained before the REVOKE retires, and
the holder receives UNINIT(F) because a linear descendant was affected. The
mapping capability survives; accesses to v fault; other pages are unaffected.
This is the crash-cleanup and kill path, and it is the only way a frame leaves
a live mapping.

**Revocation of a table page from above.** The pages table pages are made from
are carved from monitor memory, or from frames the module supplied, and the
monitor keeps a table-page handle over each (§4). REVOKE on that handle while
the page is part of a table: the page's parent slot holds a capability with a
dead node and is therefore locked (§4); if it was the root table, the root
capability in the registry entry for id has a dead node, the entry stays
reserved, and new walks of the mapping fail at that root; every translation
that depends on the page is invalidated and the accesses of §8 are drained; new
walks fail at the revoked node, since the walker checks every page-table
capability on the path; the holder receives UNINIT over the page, and the
write-before-read discipline of UNINIT yields none of the entries the page
held. The revocation does not free the entry, only DESTROY does (§5.1), so
REVOKE needs no reverse index from a node to a registry entry, and a CREATE for
the id in between is rejected. A PTE-node tag in the TLB does not express the
dependency on a table page's node, so REVOKE needs one of two strategies to
find what to invalidate: a protected record, written by CREATE and POPULATE and
keyed by (id, gen) rather than by id, of which nodes are table pages of which
mapping, consulted for the revoked node and for every ancestor the REVOKE
invalidates; or, conservatively, a global translation invalidation with the
full drain of §8 on every REVOKE, since without such a record REVOKE cannot
tell a table page's node from any other. The first prototype and the model's
default take the conservative strategy. A separate model experiment exercises
the protected table-record alternative with foreign issue during barriers;
the record remains an optimisation, and §10.3 owns its cost. Under either
strategy a REVOKE of an old generation's table-page handle,
after DESTROY and a new CREATE for the same id, must not touch the new
generation's entry; §10.2 tests this. The frames in PTEs below, and the table
pages below, are recoverable only through their own handles, each returning
UNINIT. For the domain this is loss of access to the subtree, within the goal
of §3. DESTROY still requires the teardown token; revoking the root table does
not replace DETACH.

**Domain-internal operations.** SPLIT, DELIN, DROP, MREV and REVOKE on logical
capabilities act on the pointer tree only. They never touch a PTE. Their
effect on the invariants is in §7.

## 6. Invariants

- **I1 Single path.** PTEs and table entries are linear; the root table is
  linear and held in one place, the registry entry (§4). A frame is reachable
  through at most one translation path.
- **I2 In-place use.** Page-table capabilities and PTEs cannot be loaded or
  dereferenced. An entry leaves its slot only through UNMAP or revocation
  from above.
- **I3 Monitor authority.** Before DETACH the monitor holds the detach handle,
  the frame handles and the table-page handles, none of which gives data
  access. After DETACH it holds the teardown token. After UNMAP it holds
  UNINIT(F); after revoking a table-page handle, UNINIT over that page.
- **I4 Class discipline.** PRIVATE accepts linear frames only; POPULATE
  consumes them, and POPULATE is the only way an entry enters a table, because
  CREATE and POPULATE set every slot of a page to none before the page becomes
  part of a table (§4).
- **I5 No repopulation within a generation.** Locked PTEs and locked
  table-entry slots stay locked; replacing the root table is a new
  generation, and every logical capability is bound to (id, gen).
- **I6 Completion.** After UNMAP, DETACH or a revocation from above, of a frame
  or of a table page, retires, no TLB entry, no walk result and no data access
  that passed its check before the change, on any hart, can produce an effect
  for the affected entry or subtree.
- **I7 Inheritance.** Every capability derived from the mapping capability
  carries its kind and its (id, gen).

**Argument for the goal, Stage 1.** *Reading* delegated contents needs data
authority over a frame; the monitor holds handles (I3), then the teardown
token, then UNINIT(F), each write-before-read or no access at all.
*Redirecting* a live pointer needs a PTE under a live mapping capability to
accept a new entry; used PTEs and paths are locked, no entry enters a table
except through POPULATE (I4), and a new root table ends every binding (I5, I7).
*Duplicating* exclusive authority needs a second capability to F; F is linear,
sits in one PTE, and exits once as one UNINIT (I1, I2, I4).

This is an invariant argument over the transitions of §5 and the closure of
§7. It is not a proof over hardware interleavings; §8 names the protocol that
such a proof must cover.

## 7. Closure under the rest of the ISA

The five transitions are not the only operations. Between them, software may
run any Capstone instruction. The table records what each is allowed to do to
the objects of §4 and why the invariants survive.

| Operation | On | Effect |
|---|---|---|
| SPLIT, MREV, REVOKE, DROP | Logical capabilities inside the domain | Pointer tree only; PTEs untouched. Sublet's per-object revocation works unchanged: the object node is checked on every access, independent of translation |
| DELIN | The mapping capability or a descendant | Allowed. The domain loses Stage 3 partial unmap for that range (§9.2). DETACH is unaffected because it does not depend on the mapping capability's linearity |
| Store and reload | Logical capabilities | Binding (id, gen) and kind preserved (I7). Use in another context reaches the same object while the capability is valid (§9.4) |
| Store and reload | The teardown token | Allowed; no data authority travels. Linear, so one holder |
| Load, store, dereference | Page-table capabilities, PTEs | Refused (I2). The root table never leaves its registry entry |
| Install, move, uninstall or deregister the root table | Root table | No such operation exists. The entry holds the root from CREATE until DESTROY frees the entry; revocation of the root page (§5.2) invalidates it in place, and then every capability of (id, gen) faults wherever it is used while the entry stays reserved |
| REVOKE | The detach handle | Refused; POPULATE and DETACH are its only operations (§4) |
| REVOKE | The frame handle of a mapped frame | Environment transition §5.2 |
| REVOKE | Table-page handle | Environment transition §5.2, and the page's return path |
| Any C-mode data access | Through a logical capability | Requires a live object node, bounds, rights, a present PTE, and the binding check: (id, gen) selects the registry entry, whose gen must match and whose root must be valid |

## 8. Completion protocol: break-before-make with drain

A node-liveness re-check at TLB fill is not enough: it detects revocation but
not a permission change, and a fill that read an entry before the change can
publish stale rights afterwards. Every change to an occupied entry therefore
follows one protocol:

1. **Break.** The entry is marked invalid in its slot.
2. **Invalidate with drain.** TLB entries carrying the entry's node are
   invalidated on every hart. Walks in flight either abort or publish before
   the invalidation completes. Loads, stores, atomics and capability transfers
   that passed their check through the old entry, on any hart, have either
   reached memory or been cancelled. The instruction does not retire before all
   three have happened.
3. **Make.** For UNMAP, nothing; for later PROTECT and REPOPULATE (§9), the
   new entry is written only now.

An old fill may publish between steps 1 and 2 as long as step 2 catches it.
What must never happen is an access after the instruction retires that uses the
old entry, from a TLB, from a walk result or from an access checked before the
break. The data-access clause is the one a TLB-only reading of the protocol
misses, and it is where the confidentiality argument of §6 would fail: a store
that passed translation before the break, retired on its hart and waits in a
store buffer can land after UNMAP or REVOKE retires and after the monitor has
satisfied UNINIT by writing the page, and the monitor then reads the domain's
bytes. The same protocol serves DETACH and both revocations of §5.2.

What the RTL has shown so far is narrower. R-45's fix flushes the younger
instructions of the hart that issued REVOKE or DROP, leaves the D-cache alone
and says nothing about other harts or about accesses older than the REVOKE that
are still in flight. The cross-hart drain of accesses past their check is an
obligation this design adds, not one R-45 discharges.

The node tag in the TLB is the index this protocol uses; it is not the proof
of completion. The proof is the drain, and it belongs to the RTL and emulator
implementations.

## 9. Later stages and their conditions

### 9.1 Stage 2: SHARED class, repopulation, protection

- SHARED accepts non-linear frames only. Its contract is a view on a Linux
  object; the bytes may change under a live pointer.
- A SHARED PTE whose entry was revoked from above becomes dead, not locked.
  REPOPULATE(v, F′) fills a dead PTE while the mapping is ACTIVE. The
  authority is the mapping, not the frame kind. Old pointers stay valid by
  the class contract; there are no exports to consider until Stage 4.
- DETACH of a SHARED mapping produces a teardown token as well; UNMAP yields
  nothing, the frame's other holders are untouched.
- Exclusive reclamation of a shared frame goes through the existing
  share/revoke path on the frame's node. Under existing REVOKE semantics it
  returns LINEAR, because the owner never lost access; a stronger return is an
  extension and must be named as one.
- Concurrent mapping of one frame in two tables is Stage 2, not Stage 1.
- PROTECT(range, rights, detach handle) lowers or restores PTE rights within
  max protection, with the protocol of §8. Rights restored later apply to old
  pointers whose own rights allow it; a pointer attenuated to read-only stays
  read-only.

### 9.2 Stage 3: partial unmap

SURRENDER(C), executed by the domain, where C is a linear logical capability
obtained from the mapping capability by SPLITs: revokes every descendant of C
and converts C atomically into a teardown token bound to (id, gen) and to C's
range. The monitor never sees a readable C. UNMAP(v, token) then works for
every page fully inside the range; Stage 3 relaxes UNMAP's DETACHED
precondition to: the token covers v.

C must be linear. A linear logical capability obtained by a chain of SPLITs
is the only logical data authority over its range: SPLIT consumes its parent,
and a non-linear capability cannot be split. A domain that delinearized its
mapping capability, as level0 does with its arena, cannot return pages
individually, only the whole mapping. Sublet's linear chain can.

### 9.3 Stage 4: physical exports

A physical capability over bytes of a translated object needs two lifetimes:
the object's and the backing's. A revocation tree gives a node one parent. A
child of the object node misses revocation of the frame; a child of the frame
misses `free`. The necessary property is that object revocation and backing
revocation both take effect on the export. A node checking two dependencies
is one possible implementation, not the only one. Until a mechanism exists,
sharing across tables is by mapping the same frames (Stage 2), page-granular
and without object lifetime across the boundary.

### 9.4 D3: context semantics

Decided: capability-selected root. A logical capability's binding (id, gen)
selects the registry entry for id, and through it the root table; the same
capability reaches the same object in every context while it is valid, and
faults everywhere once the entry's gen has moved on or its root is invalid.
This is the addressing design's per-access binding rule, now with its lifetime
fixed: the registry is protected state of bounded size, one entry per id,
reserved by CREATE, invalidated in place by revocation of the root page (§5.2)
and freed only by DESTROY, with no install, move, registration or
deregistration operation. The cost is a binding lookup on the access path,
which the node check already pays for in part, since the binding can live in
the node (§4).

Rejected: ambient root with a bound check, where the root table is moved into
the executing context's register and an access checks its (id, gen) against the
installed root. It needs fewer mechanisms, but each mapping has its own root
table and its own logical space, so a context could use one translated mapping
at a time and a compiler could not switch roots per pointer; Stage 1 would have
been one translated mapping per context, the heap, and the several
independently managed mappings of §9.5 would have needed a different root
structure. It is kept here as the alternative and its reason, not as an open
variant; the model of §10.1 does not run it.

| Aspect | Capability-selected (taken) | Ambient, bound check (rejected) |
|---|---|---|
| Meaning of a logical capability | Its binding selects its table | Interpreted through the installed root; the access checks (id, gen) against it |
| Same capability in a context with another root | Reaches the same object while valid | Must fault |
| Root attachment | The registry entry for id, from CREATE to DESTROY | Moved into the executing context's register; one context at a time |
| Translated mappings usable at once in a context | Any number | One |

### 9.5 Reserved extensions

The following extension paths are requirements for later design work, not
implemented features or a proof that every existing mapping can support them.
Keep several independently managed mappings per domain and PROTECT on PRIVATE
as well as SHARED mappings in scope.

| Extension | Foundation in this candidate | Additional contract required |
|---|---|---|
| Allocation on first access | Logical bounds may cover PTEs that are none; POPULATE supplies backing | Fault classification, protected continuation, response binding, cancellation and restart |
| Private swap or page migration | Logical identity is distinct from physical backing | An authorized page lifecycle that preserves object identity and contents; protected transfer of capabilities (§9.6) |
| Executable private mappings, including a `dlopen`-style loader | Separate mappings and max protections | A loading-to-execution transition, instruction-fetch checks and cache completion; loader and ABI rules remain separate work |
| Authorized snapshots and `fork()` | A child can receive separate mappings and frames at the same logical addresses | Consistent state capture, authorized capability rebinding, private object/revocation lifetimes and rules for external resources |
| Copy-on-write | Translation could select distinct backing after separation | A distinct backing-ownership and write-fault contract, including implicit writes when loading linear capabilities |

The immutable class chosen at CREATE remains a contract. Future pageable or
COW classes may define additional transitions, but must not silently change
an existing PRIVATE mapping's promises. SHARED refill is not a substitute for
preserving private contents. New operations must retain the security goal of
§3: the detach handle, the teardown token and table ownership alone must not
authorize readable snapshots, arbitrary redirection or duplication of
exclusive authority.

Logical capabilities are bound to (id, gen) (§9.4). A fork cannot merely copy
their bits and select a different root table. For cloned private objects it
needs authorized child bindings and corresponding object/revocation state,
preserving bounds, rights and alias relationships. Parent-side free must not
revoke the child's independent copy. External linear resources cannot simply be
duplicated. Full copying and COW are separate designs; neither is supplied by
CREATE and POPULATE alone.

The compatibility review identifies possible extension paths and conflicts with
naive implementations. It does not establish a complete extension, unchanged
ABI compatibility or a software-only implementation. Before fixing an encoding
or claiming an extension, check the relevant cases in §10.4 under the registry
rule of §9.4. Hardware cost and interleavings remain separate obligations.

### 9.6 Private paging of capability-bearing memory

Ordinary byte I/O does not preserve capability authority. Restoring saved tag
bits without authorization would also be unsafe: replaying one saved page
could create two live copies of a linear capability. A future private pager
needs a protected transfer contract, not an ordinary memcpy or Linux swap path:

- Protect contents, capability fields and tags against disclosure, modification
  and replay. Bind the saved image to the mapping generation, page and current
  version. Linux may transport encrypted, authenticated images; authorization
  and freshness checks must not depend on Linux or the monitor being honest.
- Authorize capture and restoration without giving a holder of the detach
  handle access to private data. Drain affected accesses and transfers before
  reusing backing; a linear capability must never have two usable instances,
  including after retry, cancellation or duplicate delivery of a saved image.
- Preserve capability identities and revocation while the page is absent.
  A capability revoked during that time stays unusable after restoration.
  Node lifetime and generation handling must prevent reuse of an old identity
  from reviving it; authentic old bytes alone are insufficient evidence.
- Restore the same logical object through an explicitly authorized transition.
  A physical capability to the page does not follow logical remapping; such
  aliases require a separate relocation contract or resident backing. Moving
  capability-containing storage does not itself relocate the objects named by
  the capabilities stored there.

Table pages and the pager's essential code, stack and revocation metadata stay
resident in an initial paging extension. Paging those structures would need
its own progress and recovery argument. The existing
[fault-delegation conditions](../plans/delegation-memory-options.md#6-demand-paging-can-keep-the-monitor-file-blind)
also apply. These are requirements for a later design, not transitions added
to Stage 1 by this section.

## 10. Experiments

### 10.1 Executable model

The [host executable model](../../tests/mapping-model/README.md) implements this
experiment for PRIVATE under the capability-selected root. Its
[result record](../../tests/mapping-model/results.json) pins source hashes,
search bounds, operation coverage and eight faulty comparison traces. It checks
named contracts, exhausts forty fixed two-hart interleaving workloads in both
global and table-record modes, and explores fifteen further table-record
workloads with foreign issue and two successive removals. Three reduced
lifecycle alphabets run to depth six from ACTIVE, DETACHED and DESTROYED seeds;
coverage gates require memory effects and completed removal operations. Random
sequences cap revocations and require data effects per seed. The record states
the seed setup, bounds, depth frontier and per-seed coverage; these are restricted
workloads, not full lifecycle enumeration. In particular, the earlier depth-three
search covered prefixes only, and aggregate nonzero random coverage hid empty
seeds. The README retains those limits alongside the replacement checks.
The README defines the abstraction: two-level word-addressed tables, atomic
memory effects and no revocation-node reuse, compressed encoding, cross-page
instructions or hardware cache/coherence model. These bounded results do not
discharge §10.2 or §10.3 and are not an unbounded proof. The requirements below
remain the contract for extending the model and refining an implementation.

Before an emulator change, a small executable model: nodes, capabilities with
kind and binding, a table tree of at least two levels, a TLB with node and (id,
gen) tags, the five transitions, the environment transitions, the operations of
§7, and an adversarial monitor. The transitions are not atomic in the model:
walk, translation completion, an outstanding data access that passed its check,
invalidation, drain and return are separate steps that the scheduler
interleaves, so that the §8 window and the store-buffer case exist in the model
at all. It generates random and bounded-exhaustive sequences of allowed
operations and checks I1–I7 and the three goal clauses after every step. It
models at least two mappings and two contexts, otherwise foreign tokens and
cross-context use are only partly checkable. Every check has a positive
control, and each of the following deliberately faulty variants must be caught
while the corrected model rejects the same sequence: the detach handle as a
plain revocation handle, which must reproduce the DELIN counterexample of §4;
CREATE or POPULATE that converts a page without clearing its slots; a drain
that covers walks but not data accesses past their check; a binding check that
ignores gen or id; a table page returned both through its handle and by a
traversal from the root; a root revocation that frees the registry entry
instead of leaving it reserved, or a table-page record keyed by id alone, so
that an old generation's handle reaches the new generation's entry. One run,
under the capability-selected root of §9.4; the rejected ambient variant is not
modelled.

### 10.2 Adversarial-monitor tests for the emulator

Each with a positive control, each expected to fail in hardware, not in
monitor code:

| Test | Expected |
|---|---|
| POPULATE with a non-linear frame into PRIVATE | Rejected |
| POPULATE with a logical capability, or any non-physical kind, as F | Rejected |
| POPULATE whose path needs a table page that was not supplied | Rejected; nothing consumed, no partial path |
| CREATE or POPULATE with a page that holds a stored capability | Every slot of the converted page is none; the stored capability does not translate, and a copy the supplier kept reaches nothing through the mapping |
| CREATE with an id in use; DESTROY twice | Rejected |
| UNMAP without the token, with another mapping's token, or twice on one PTE | Rejected |
| Read an unmapped frame before writing it | Fault: UNINIT |
| Replace a table entry to redirect v under a live mapping capability | Rejected: locked |
| Install one linear frame in two PTEs | Impossible: the first POPULATE consumed it |
| Access during the §8 window | Served before completion and then invalidated, or faults; never after completion |
| Store past its check, then UNMAP or frame revocation from above, then the monitor scrubs and reads | The store lands before the instruction retires or not at all; the monitor reads only its own bytes |
| DELIN the mapping capability, then DETACH | The teardown token only; no readable capability appears |
| Revoke the frame handle of a mapped frame | That page faults, other pages live, the holder gets UNINIT |
| Revoke a table page with a warm TLB and a walk in flight | After completion every access through the subtree faults; the holder's UNINIT yields no entry; the PTEs' frames return through their frame handles and the pages below through their table-page handles |
| Revoke a table page, write and reuse it elsewhere, then DETACH and DESTROY the mapping and reclaim the remaining pages | Each remaining page returns once through its handle; no second capability over the reused page appears |
| Revoke a table-page handle, then revoke it again | The first yields UNINIT over the page; the second is rejected, since the handle became that UNINIT |
| DESTROY an old generation, CREATE a new one under the same id, then revoke a table-page handle of the old generation | The old page returns as UNINIT; the new generation's entry is untouched and its mapping keeps translating |
| CREATE twice for one id, then DESTROY the first | The second CREATE is rejected while the entry is in use; after DESTROY every holder of (id, gen) faults; a later CREATE for the id gets a fresh gen |
| Revoke the root page, then CREATE for the same id, then DETACH and DESTROY, then CREATE again | Every walk of the mapping fails after the revocation; the first CREATE is rejected, the entry is still reserved; the second succeeds with a fresh gen |
| DESTROY with PTEs still present, then revoke their frame handles | DESTROY succeeds; each frame returns once as UNINIT; no access through the old (id, gen) succeeds |
| malloc, free, malloc at the same logical address, then use the old pointer | Old pointer faults on its object node; the new one works |
| Three scattered pages as one logical interval | Bounded pointer arithmetic crosses page boundaries |
| Same logical capability used in another context | Reaches the same object while valid; after DESTROY or root revocation it faults there too |

Deferred with their stages: frame revocation with a live export (4), PROTECT
against a walk in flight (2), shared backing under a linear logical pointer
(rejected by class in Stage 1, tested again in 2).

### 10.3 RTL question

Whether a capability-checking walker, a node tag in the TLB and the drain of
§8 fit CVA6 is a question for the RTL lane before this candidate becomes a
target. No cost is claimed here.

The optional model experiment records table-node bindings at conversion and
allows issue for other bindings during a scoped barrier. Physical frame REVOKE
still uses the global fallback. The experiment checks interleavings and
generation isolation; it does not supply a bounded hardware record, its
reclamation protocol, node-reuse handling, lookup latency or a refinement proof.

### 10.4 Extension checks before an implementation or ABI claim

These checks are not yet implemented or passed. Extend the model before
claiming the corresponding feature:

- First-use paging: a valid access through a split child of the mapping
  capability can fault on a PTE that is none, populate and restart. A revoked
  pointer, a locked PTE, a foreign or stale pager response must fail, with
  successful controls for the valid cases.
- Private paging: save and restore a page containing a linear capability;
  preserve its single usable instance and object identity. Reject replay and
  cross-mapping substitution, and do not revive a capability revoked while
  absent. Repeat with outstanding accesses and cancellation before frame reuse.
- Fork: clone private objects with capabilities in memory and registers under
  the registry rule of §9.4; preserve child alias relationships and independent
  free. Reject duplication of external linear resources and snapshot requests
  that present only monitor handles. Establish full-copy semantics before COW.
- COW: separate private copies before an ordinary store or the implicit write
  of a linear-capability load. Check both backing authority and capabilities
  stored in the page for duplication; a read-only PTE alone is not the proof.
- Executable mappings: model independent module lifetimes and permission
  changes through instruction fetch as well as data accesses. For every new
  class or operation, retain the Stage-1 rejection of PRIVATE PTE refill and
  monitor access to delegated contents.

## 11. What this candidate changes in the sibling documents

- [Options §5 and §9](../plans/delegation-memory-options.md): capability-bearing
  PTEs move from an implementation footnote to the candidate itself, with the
  three decisions of §2 separated.
- [Addressing §3](delegated-memory-addressing.md): the per-access node-to-grant
  binding is D3's taken variant, now stated with the registry's lifetime in
  §9.4.
- [Addressing §6](delegated-memory-addressing.md): "physical authority
  encapsulated by the mapping object" becomes the linear PTE plus I2; the
  ownership rules are the transitions of §5, not a sentence.
- [Addressing §8](delegated-memory-addressing.md): partial unmap is answered by
  I5 and, for reuse, by a class fixed at CREATE.
- [Plan M3](../plans/delegation-memory.md): under translation the heap is one
  growing logical interval with lazy backing; the pool index may become
  unnecessary, the growth mechanism does not, and under this candidate it is
  POPULATE's table-page operand. Re-plan M3 only once §9.2 has a
  contract. M1 and M2 are unchanged: the resume-slot delivery is the frame path
  of POPULATE.
- `ports/musl-capstone/runtime/sublet_heap.c`, header comment: the statement
  that a stale capability's load retires on the RTL predates the R-35 and R-45
  fixes recorded in ISSUES.md.
- The Caplification reference cites the paper for its critique; Figure 1(d) is
  also the source of D1.

## 12. Boundary to the syscall path

Translation does not shorten a delegated round. What can be removed are the
copies, and that is independent of this candidate: a Linux-visible I/O pool
mapped in the launcher, plus a validation contract for pointer arguments so
that Linux does not act as a confused deputy between compartments of one
application. The contract must cover kind, rights per direction, the full
length with overflow, node validity, the pool membership of the frames, and
the values the launcher uses being the ones the monitor reported at STEP. A
per-context in-flight table in the libc keeps an object from being freed and
reissued during the kernel's access; it holds under the assumption of §3,
cooperating components and completion reported by the launcher. The details
belong to the [options document](../plans/delegation-memory-options.md), not
here.

## 13. References

- Caplification: Bridging Capability-Aware and Capability-Oblivious Software,
  SACMAT 2025, §1 and Figure 1(d);
  Table 4 for measured trap and switch costs on CapliFive-RTL.
- Capstone: A Capability-based Foundation for Trustless Secure Memory Access,
  §2.3 (physical addressing), §4.3 (REVOKE returns LINEAR or UNINIT
  depending on the affected descendants).
- AMD SEV-SNP reverse map table and PVALIDATE; Intel TDX TLB tracking
  (BLOCK, TRACK, REMOVE); Arm CCA granule delegation: precedents for the
  requirements of §6 and §8, not for capability-bearing entries.
- ARM break-before-make: the protocol of §8.
- Linux: `mmap(2)`, `MAP_POPULATE`, `VM_MAYREAD`/`VM_MAYWRITE`, `pte_none`,
  `pte_present`, `zap_page_range`, `free_pgtables`: the vocabulary borrowed
  above, where the meaning matches.
- Local: [ISSUES.md](../ref/ISSUES.md) R-31 (REVOKE must return UNINIT for a
  linear borrow), R-35 (fixed; its generation field retires rather than wraps),
  R-45 (fixed by a pipeline flush of the issuing hart's younger instructions;
  §8 needs more), R-44 (open);
  [sublet.h](../../runtime/include/sublet/sublet.h) lines 33 and 58 for the
  alias/linear distinction and the UNINIT discipline.
