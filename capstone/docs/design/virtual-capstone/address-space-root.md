# The kernel as the outermost allocator

[Guide](README.md) · Related: [ISA](isa.md) · [Runtime and Linux](runtime.md)

Status: design decided on 2026-11-12, on the branch `address-space-root` of
this repository and of capstone-qemu (both from `one-operand-crevoke-crenew`).
The decisions of 2026-11-13 in
[plan v5](../../plans/2026-11-13-kernel-trap-entry.md) replace three parts of
it: the reservation record M per mapping, the libc's
copies of system-call buffers and the code reference. This note records them,
and plan v5 lists the steps that implement them. QEMU implements and
qualifies the rules of supervisor references, but not the rules that plan v5
adds. The Linux adapter, the libc and the RTL still use earlier designs,
described under "What runs today".

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

Decided on 2026-11-13 (plan v5, decision 8), after review. This replaces the
kernel-private reservation record M of 2026-11-12. The adapter still
implements M until the stage of the adapter plan that removes it, the next
adapter stage. The paragraphs of this section that name a stage of the adapter
plan describe the adapter's code with that next stage already applied, and the
paragraph "The adapter until M goes", at the end of this section, lists what
differs in the code today.

Each region of a protected program is a block of depth 1, a child of R: the
image, the startup block, the main stack, each `mmap` and the shared region
while it exists. The kernel derives each block with one `CDERIVE` from R and
hands its reference P to the program. P names that reference also after the
program renewed it, and an object below P is any deeper record in P's subtree.
The kernel keeps no reference to the block and no private object below R. It
keeps only bookkeeping: a block object with the block's extent (padding
included), its visible length, the pages that a partial `munmap` removed and
the kernel-owned flag, and the launcher's record with the state of each page,
its protection or removed. Linux splits and merges VMAs for its own reasons, so
the block object, not a VMA, identifies the mapping, and VMAs of different
blocks never merge. The kernel holds no copy of P either: the program's first
`CRENEW` would leave the copy stale, and a `CREVOKE` through it would end
nothing while the kernel still released the range.

Every request of the program to end a block or to remove pages of it comes
with its reference: `munmap`, `mremap`, a park in the mapping cache, the unmap
of a thread's stack at thread exit, and the hand-over of a thread's frame at
thread creation. The kernel accepts the reference if and only if all of these
hold:

1. The operand is a reference: it is tagged.
2. Its lifetime is live.
3. Its record has depth exactly 1.
4. Its bounds cover the pages that the call names, and those pages lie inside
   one block. The cursor is free, as after pointer arithmetic on P.

Conditions 2 and 3 identify the block. Only the kernel holds R, so only the
kernel derives objects of depth 1, and every live record of depth 1 is a
block. A renewal keeps a record's place and depth, so a renewed P passes.
Every object that an allocator derives inside a block lies deeper, so an
object over the whole block never passes. Blocks are disjoint, so the
reference of one block covers no page of another. The depth is a field of
every record of the Sublet format, and the kernel reads it in the same step as
the generation check, so condition 3 needs no parent field.

Condition 4 is a policy choice, not a safety condition. The abstract machine
lets any live reference release its object, whatever its bounds, and
`CREVOKE` checks only the lifetime. The kernel asks for the bounds so that a
reference narrowed to part of a block removes no page outside its bounds and
cannot end the block.

A `munmap` that names the whole block ends it: the kernel executes `CREVOKE`
on the passed reference, and only then does Linux remove the pages and free
the address range. A `munmap` that names part of the block is a partial
`munmap` (below): it removes pages and ends no lifetime, also when it removes
the last mapped pages. Only a call that names the whole block ends a block. A
refused operand never reaches an instruction that can trap, and the kernel
changes nothing on a refused request. The `CREVOKE` has an exception-table
entry, and a `CREVOKE` that faults refuses the request before anything
changes. Only the teardown of the process and the rollback of a failed `mmap`
end a block without a request of the program: the rollback executes `CREVOKE`
on the fresh reference that the kernel still holds, and teardown frees the
table.

Kernel-owned blocks. The image, the startup block, the main stack and the
shared region carry the kernel-owned flag, and the kernel refuses `munmap`,
`mremap` and `mprotect` on them. The image's data reference covers the whole
image, so only the page's W bit keeps stores out of the text, and only the
kernel may change it.

Several harts. The kernel runs the check and the `CREVOKE` in one step while
no thread of the program runs (in the native kernel of plan v5, with
preemption off after `mmap_write_lock`). On one hart the passed reference
therefore stays live from the check to the `CREVOKE`. With several harts,
another hart can renew or end the reference in between, and the kernel must
meet three conditions. A `CREVOKE` that faults refuses the request with
nothing committed, and the kernel never reads a stale operand as a block
without a live successor. The kernel reads the depth in one coherent
observation with the generation: a reference that another hart ended can leave
its slot to a new block elsewhere, whose record shows depth 1. The refill test
(below) holds the program's other harts off until the commit, since another
hart can derive below P right after the test. The prototype keeps one hart
(plan v5, decision 12).

Every other service of the launcher takes a checked reference as well
(decided on 2026-11-13, stage 4c of the adapter plan): the kernel side reads
or writes the program's memory, or holds a mapping of the program, only
through a reference that the program passed and the kernel checked, or
through the system-call region it shares with the program. `mprotect` and
`madvise` come with a reference that passes conditions 1 to 4. A worker's
attachment of its transport names its meta block and its exchange region with
references that pass conditions 1 to 4 for the whole block, and ends nothing.
The exit word that the kernel writes at a thread's exit and the word of a
futex wait come with a reference that is live and covers the word;
the word may be any object's, so condition 3 does not apply, but it may not
lie in the allocator's metadata. No request may reach a page that holds the
frame of a thread, whatever reference it carries. A futex wake takes an
address, because it neither reads nor writes the word.

Check and access in one step (decided on 2026-11-13, stage 4d of the adapter
plan). Every access of the kernel side to program memory through a program
reference happens inside one step of the kernel that checks the reference and
performs the access, and no thread of the program runs during that step. In
the prototype the step is one module ioctl under the module's lock, which
every thread of the program needs to execute. So no other thread can end the
reference's lifetime, unmap the page or map another page there between the
check and the access. A check whose access comes later, outside the step, is
not enough: in stage 4c the launcher stored the exit word after the module had
checked it, and Linux read a futex word after the service lock dropped, so a
concurrent `CREVOKE` or a whole-mapping `munmap` could put another object or
another mapping under the access. The design reaches the same property by
construction (section System-call buffers): the kernel accesses through the
reference, and the processor checks it at the access and again at the retry
after a page fault.

- The exit word. The kernel checks the reference while the exiting thread's
  registers still hold it and stores 0 at the end of the same step, after the
  thread's frame and stack releases are committed. It checks again what those
  releases can change (the lifetime, the mapping, the frames), admits the page
  as a fault of the program would, and stores through the program's page
  table, which decides the write right. A word inside the stack that the same
  exit releases does not keep the stack: the stack is released and the store
  is dropped without an error, as Linux drops a `clear_child_tid` store into a
  stack that is gone.
- A futex wait. The kernel checks the reference, reads the word and compares
  it with the expected value in one step; the read admits no page, so a
  read-only page compares as any other. If they are equal, the waiter sleeps
  on the address the kernel checked, as Linux keys a private futex, and Linux
  never sees the program's address. When a range stops being mapped (an
  unmap, a park, a thread's stack or frame at its exit, a partial `munmap`),
  the kernel wakes every waiter in it. The libc then reads the word again
  through its own reference, which faults if the reference is stale and waits
  again otherwise; a spurious wake is legal for a futex. So no waiter of an
  ended mapping meets a wake of a later mapping at the same address. (Stage 4d
  keyed the waiter by the mapping's arena id and the offset, and a revive from
  the mapping cache, which gives the mapping a new id, stranded its waiters.)
  The same can happen inside a live mapping: the program frees the object
  that holds the word and a new object takes its address. So a waiter keeps
  the lifetime of the reference the kernel checked, and a wake counts only
  waiters whose lifetime is still live (stage 4f): a stale waiter is woken as
  well, since nothing else would end its sleep, but it does not take the wake
  from a live waiter at the same address. Without root grants the kernel
  knows no lifetime of a word, and every waiter counts.
  A waiter enters the queue in the same critical section as the compare, and
  a wake needs that section, so a wake between the compare and the sleep marks
  the waiter, whose sleep then returns at once. The kernel's wait for its own
  lock comes before any effect, so a signal that interrupts it makes the
  launcher repeat the step.

A page operation is one step as well (decided on 2026-11-13, stage 4f of the
adapter plan). A partial `munmap`, a `MAP_FIXED` refill and `mremap`'s shrink
and growth in place change pages of a block in three moves: the kernel
checks the request, Linux changes the range, and the kernel commits. In the
prototype these are two module calls with Linux's change between them, and
the launcher holds its service lock across all three. The check opens a
transition in the module. Until the commit or an abort closes it, no thread of
the program runs, and no page of the range is admitted. So no thread can
derive an object below P, store a capability into a page of the range, or
have a page of the range admitted between the check and the commit. Stage 4e
let the program run in between: a thread could derive an object below P after
the refill rule was checked, or have a page of the new range admitted that the
commit then unpinned while it stayed mapped. The transition protocol:

1. Check. The kernel checks the reference, the range and, for a refill, the
   refill rule. It pins every resident page of the range that it does not hold
   yet, so that the commit clears that page's tags too. Then it opens the
   transition. A call that names the whole block opens nothing: it ends the
   block at once with `CREVOKE` on the passed reference. A removal of the last
   mapped pages opens a transition like any other removal and ends no
   lifetime.
2. Linux. The launcher changes the protection of the range with `mprotect`:
   `PROT_NONE` for a removal, the asked protection for a refill. A refill then
   discards the old contents with `MADV_DONTNEED`, while the kernel still holds
   its pages pinned. No `MAP_FIXED` replaces the range: the VMA, its
   `VM_NOHUGEPAGE` and the address range stay, and no other mapping can take
   the range even for a moment. `mprotect` can fail before or part way through
   its change, with `ENOMEM` when a VMA split meets `vm.max_map_count`. Then
   the launcher restores the protection the range had, the kernel aborts the
   transition, and the program gets `ENOMEM` with nothing changed. A refill's
   discard cannot be undone that way: it may fail after it discarded part of
   the range, and Linux would then supply fresh frames there that neither the
   program nor the kernel's pins describe. So a discard that fails ends the
   whole block, fail closed (block A): the kernel executes `CREVOKE` on the
   request's reference, the program gets `EINVAL`, and every reference into
   the block is stale.
3. Commit. A refill checks the refill rule again. If an object below P exists
   now, the commit fails closed: it executes `CREVOKE` on the request's
   reference, which ends the block and the object, so the object is stale
   before any access can see its pages replaced, and the program gets
   `EINVAL`. Otherwise the kernel clears the tags of every page of the range
   that it held, unpins it and lets Linux reuse the frame; it updates the
   removed pages and closes the transition. A commit that fails closed retires
   the block as an unmap does: it first pins every resident page it does not
   hold, so that the retire clears its tags. If that preparation fails (a page
   that migration moves past the pin), the kernel does not go on with frames
   that may keep their tags: it overwrites every page of the block that it
   does not hold in place through the page table, data and tags, while no
   thread runs, and then retires (block A). A page it cannot overwrite (a
   swapped, shared or huge page, outside the profile) does not stop it: it
   clears every other page, retires, and tells the launcher, which keeps the
   block's whole extent mapped `PROT_NONE` and never gives it back to Linux,
   so no frame that may keep a tag reaches Linux's allocator while the
   process runs (block A, F4).
4. A removal's pages are discarded after the commit. If Linux refuses that
   discard, the pages stay removed: `PROT_NONE`, never admitted, and without
   tags.

A fault on a page of an open transition does not stop the program: the thread
waits for the transition to end and then meets the page as the transition
left it. In the prototype that is a thread whose fault reached the
launcher before the check, since no thread runs during the transition. If the
launcher's release of a block that the kernel has already ended fails in
Linux, the range stays mapped in Linux. Linux then gives no address inside
it to another mapping, and the launcher drops its record, since every
reference into the range is stale.

One access stays outside the rule, as a diagnostic: after a fault has stopped
the program for good, the launcher walks the frame chain of the main stack for
the fault record. Every read is bounded to the main stack's mapping, a grant
of the launcher that is never unmapped while the process lives, including
reads reached through a corrupted frame pointer, and nothing it reads is
returned to the program, which never runs again.

Sublet checks bounds and lifetime, and the page table checks mapping and
rights. An access needs both: the page must be mapped with the right the
access needs, and the reference must be live and in bounds. Read, write and
execute rights are the page table's alone, and so is the presence of a page.
A reference passed to `mprotect` or `munmap` only proves ownership of the
mapping block.

No capability right remains in the adapter (block A, 2026-11-13). R is minted
with every right, the one descriptor the Sublet format accepts, and every
grant inherits it: the kernel executes no TIGHTEN, the launcher names no
rights, and the libc's map service takes none. What a right protected before
is now a page right. The startup block is mapped read-only after the launcher
packs it (it was a read-only grant over read-write pages). The image's text is
mapped read-execute and its data read-write, as before. Every other mapping,
the heap, its metadata, stacks, frames and the transport included, is mapped
without execute, and `mmap`, `mprotect` and `MAP_FIXED` refuse `PROT_EXEC`, so
only the image's text ever executes. The legacy grants without root grants
(the 30-bit and 55-bit formats) are minted with read, write and execute, and
the page table decides there as well.

Allocation can fail (block A). Where the lifetime table has no slot and cannot
grow, the processor's `CDERIVE` gives its failure result, an untagged 0, and
no record changes. The kernel turns a 0 from its `CDERIVE` of a block into
`ENOMEM` for `mmap`, and it refuses an `mmap` with `ENOMEM` before any change
when the table cannot grow and has no room for the block's record (the adapter
needs room for two records, M and P, until M goes). The libc's allocator
turns a 0 from its object's `CDERIVE` into `NULL` from `malloc` with
`ENOMEM`, and gives the slot back to its free list. `realloc` ends the old
object only after the new one exists: in place it derives the new object
first, and a resize that would move a large object's mapping takes `malloc`,
a copy and `free`, so a `realloc` that fails answers `NULL` with the original
intact (block A, F4; part 1 moved the mapping
first and ended the old object before the new one was issued).

The kernel grows the table itself (block A, the Sublet format). It never lets
its own allocations trap: before its `CDERIVE`s it makes room for them, and a
table it cannot grow has NOGROW set, so its `CDERIVE` then fails with the 0
above instead of raising cause 30 in S. A program's context escapes to the
kernel while at most 256 slots are left (the pressure escape, cause 30). Every
growth attempt of the kernel, for its own `CDERIVE`s, for the pressure escape
or for the allocator's preflight, names how many slots it needs, adds record
pages when fewer are available (at least a batch of 16 pages, 4,096 slots,
published before the larger capacity), and then sets NOGROW exactly when the
need is still unmet or no more than 256 slots remain. NOGROW thus means that
after the last attempt the requirement is unmet: an attempt whose need is met
clears it, also without adding a page (releases gave slots back), and an
attempt that falls short (a page allocation fails, or the table's page quota is
reached) sets it, so the resumed instruction takes the last slots and fails
with 0 afterwards. Either answer to the escape makes progress: more than 256
slots available, or NOGROW. The table starts with at least one batch.

No mapping block holds the topmost page below 2^38 (block A, F4). The Sublet
format keeps a reference's cursor as the zero extension of 38 bits, so a
cursor at 2^38 is not representable, and an object ending exactly there could
not hold a tagged pointer one past its end, which C allows every object. Only
R ends at 2^38. The launcher holds that page with a `PROT_NONE` mapping of its
own from its start, so Linux hands no part of it to a block, and both the
launcher and the kernel refuse a block that reaches it.

| Call | Rule |
|---|---|
| `mmap` | a new block, a child of R of depth 1, as above. `PROT_EXEC` is refused (`EACCES`): only the image's text is executable. |
| `munmap` | through an accepted reference (conditions 1 to 4). A call that names the whole block ends it: `CREVOKE` on the passed reference, then Linux removes the pages and the address range becomes free. A call that names any other page range inside it removes those pages and ends no lifetime (a partial `munmap`, below). A range beyond the block, across two blocks or by a plain address is refused. |
| `MAP_FIXED` | inside a block: only with an accepted reference and under the refill rule, only while the program derived no object below P (the record after P's in the preorder list is not deeper, one step). The pages of the range, removed ones included, read zero afterwards with the protection asked for; no lifetime ends. By a plain address and into free address space it is refused. |
| `mremap` | first validates the whole old range: it lies inside one block, every page of it is mapped, and the reference is accepted (conditions 1 to 4, any cursor). Then it shrinks in place (a partial `munmap` of the tail; no lifetime ends, and the block keeps its address range) or grows in place inside the block's visible length, into its own removed pages, under the refill rule of `MAP_FIXED`. Elsewhere `MREMAP_MAYMOVE` moves the range: a new block, a copy of the contents with their tags, and a `munmap` of the old range through its reference. That `munmap` ends the old block only when the old range is the whole block; otherwise it removes those pages, and the block keeps its address range. `mremap` thus gives an address to another mapping only after the block that covered it has ended. Without `MREMAP_MAYMOVE` the answer is `ENOMEM`. `MREMAP_FIXED` and `MREMAP_DONTUNMAP` are refused. |
| `mprotect` | with an accepted reference (conditions 1 to 4); the named page range may be part of the block. It ends nothing. `PROT_EXEC` is refused (`EACCES`), and so is a removed page (`ENOMEM`, as for an unmapped page). |
| `madvise` | `MADV_DONTNEED` and `MADV_FREE` as `mprotect`: an accepted reference (conditions 1 to 4). The range reads zero afterwards. It ends nothing. Other advice answers `ENOSYS`. |
| kernel-owned blocks | the image, the startup block, the main stack and the shared region: the kernel refuses `munmap`, `mremap` and `mprotect` on them, and `madvise` of the image as well. The image's data reference covers the whole image, so only the page's write bit protects the text, and only the kernel may change it. |
| kinds | anonymous mappings only. File-backed mappings, mappings shared between processes, `brk` and `userfaultfd` are refused. |
| shared region | the system-call region belongs to the kernel, is never unmapped, and its address range is never reused. It carries no system-call buffers (section System-call buffers) and goes once no path uses it. |

A partial `munmap` (decided on 2026-11-13 after review, stage 4e of the
adapter plan) removes pages and ends no lifetime. The kernel makes the pages
`PROT_NONE` and discards their contents; they stay part of the block's VMAs
(stage 4f; stage 4e mapped a `PROT_NONE` placeholder `MAP_FIXED` over them).
It never admits a removed page again until a refill maps it, and gives no
other mapping an address inside the block while the block lives. A reference
into a removed page stays live; an access through it is a page fault, as on a
page of no mapping, never a lifetime fault. The block ends with `CREVOKE` only
when one call names the whole block, or at exit. A partial `munmap` that
removes the last mapped pages leaves the block and its address range in
place, and unmapping a removed page again changes nothing. The kernel clears
the tags of every removed page it held and unpins it before Linux can reuse
its frame, and the frame leaves the duplicate check of the block, so a refill
may receive it again. Without root grants the kernel cannot read the
operand, and a partial `munmap` is authorised by the address alone: a legacy
profile without the reference guarantee of this section, kept until the
bitstream has supervisor references (adapter plan, stage 5). No refill is
possible there, since the refill rule needs the reference.

The abstract machine keeps its lifetimes exact under this rule. Page presence
is the page table's business, as page rights are: a partial `munmap` changes
which pages exist, not which objects live, just as `mprotect(PROT_NONE)` makes
pages of live objects fault and `MADV_DONTNEED` replaces their contents. What
the machine needs is that no address of a live block reaches another object,
and the block keeps its whole range until its lifetime ends. The claim is
therefore stated for the checks Sublet makes: for faithfully reported
lifetimes, Sublet's bounds and lifetime checks accept exactly the accesses
the abstract machine defines; page presence, page rights and resource limits
are separate conditions of the page table, and a page fault on a removed page
of a live block is the program's own request, not a verdict on memory safety.
An address range returns to other mappings only after the lifetime of its
whole block has ended, so no stale reference reaches a new mapping. The cost
is address space: a hole stays unusable by other mappings until its block
ends.

The refill rule (stage 4f): a refill gives pages of the block fresh zero
pages, `MAP_FIXED` and the growth of `mremap` into the block's own removed
tail alike, so it is allowed only while the program derived no object
below P. An object that a nested allocator derived below P would otherwise
see its memory replaced or, after a partial `munmap` of its pages, reappear.
One rule for both operations, checked in one place, at the check and again at
the commit. Stage 4e exempted the growth, on the grounds that it maps only
removed pages; but an object derived over a page before that page's removal
lives on, and the growth would give its memory back.

The prototype's `mremap` (stages 4b, 4e and 4f of the adapter plan) validates
the whole old range first on every path, the same way: the range lies inside
one block, every page of it is mapped (`EFAULT` otherwise, as Linux answers
for an unmapped part of the old range), and the reference passes conditions 1
to 4 for that range. Stage 4e checked only the tail for a shrink and a growth:
a shrink over a hole could remove the last mapped page, end the mapping and
still answer success, and a growth succeeded with holes in the old range or
with a cursor outside the mapping. A shrink and a growth in place are one
kernel step each, the validation included. A move is map, copy and unmap
in the libc, not one kernel step. Before it copies anything it asks the
kernel, without changing anything, whether that range may move: the
validation, and every page of it read-write. So a stale reference, an object
below P or a range beyond the block fails with nothing changed and never
faults in the copy. The answer holds when it is given: another thread of the
program may end the old mapping or change its protection between the check
and the copy, and the copy then faults instead of answering `EINVAL`. That
affects only the program itself, as a concurrent `free` does for `realloc`.
It moves only a range that is read-write over its whole length, and refuses
every other one (`ENOTSUP`). The copy is not atomic for the program's other
threads: a write to the old range during the copy may be lost, as for
`realloc`. Shrinking and growing in place need no copy.

`madvise(MADV_DONTNEED)` and `MADV_FREE` change contents, not lifetimes, and
need no revocation. They destroy contents, so they come with a reference, as
`mprotect` does. The pinned kernel overwrites every page of the range with
zeros, data and tags, whatever protection the page has, `PROT_NONE` included:
the pages it holds through its pin, and any other page with contents in place
through the page table (there is none in a heap mapping, since every page the
program touches is admitted at its fault, but the kernel does not rely on
that). It checks every page first, that each is still private to the mapping,
and zeroes only when all pass, so a refusal of that check changes nothing; a
page that migration or compaction is moving makes it wait and look again. A
migration that outlasts the wait (about 10 s) makes the zeroing fail after it
zeroed part of the range: the zeroed pages hold no tag and stay mapped, the
others keep their contents, and no lifetime changes (stage 4f corrects the
claim of stage 4e that every failure changes nothing). It marks each zeroed
page dirty, and keeps its pages pinned and mapped, so no tagged page goes back
to Linux. The unpinned kernel holds no pages, and Linux gives the range back,
as at the unpinned release of a whole mapping. The unpinned kernel relies on
Linux to clear the tags of every frame it frees (the tag-aware page free of
the unpinned kernel patch); the module refuses to load in the unpinned mode
unless the kernel reports that hook active.

A program that ends a block's lifetime itself and hands back the stale
reference is refused, and the range stays unusable until `exit`. That is a
leak in a faulty program, not a safety problem: every reference into the
range is already stale.

A program that narrows P and then renews the narrowed reference can no longer
end that block. The renewal ends the lifetime of P, of every copy of it and of
every object below it, and gives R a new child of depth 1 with the narrowed
bounds. That child passes conditions 1 to 3, so it can remove, protect or
advise the pages it covers. It does not cover the whole block, so no call can
name the whole block with it, and the block keeps its address range until
exit. That is a leak, not a safety problem: every reference to the old object
is stale, and the kernel frees no address range.

The adapter until M goes. With root grants (module parameter `root_grants=1`),
the adapter's code of today differs from this section in these points, which
the next adapter stage changes. `root_derive` derives a reservation record M
from R for each mapping, keeps the only reference to M, and derives P from M
with the same bounds, so P has depth 2, and an `mmap` needs room for two
records. Condition 3 asks that the operand be the child of M: its record
directly follows M's in the preorder list, has M's depth plus one and is live
(`ref_child`; `ref_reservation` finds M by its depth of 1). For `munmap`,
`mremap` and the other requests that end a mapping or remove pages of it,
condition 4 asks that the bounds cover the whole visible length, so a narrowed
and renewed P, a new child of M, can no longer remove pages either. The
adapter ends a mapping with `CREVOKE` of M, also when a partial `munmap`
removes the last logically mapped page, which the block object counts, and
every commit that fails closed ends M. No kernel-owned flag exists yet: the
adapter refuses `mprotect`, `madvise` and `munmap` of the image, and the
launcher never unmaps the main stack. Without root grants the adapter runs the
earlier design (section What runs today).

## Supervisor references

The kernel computes with plain addresses, and its loads and stores to its own
memory pass no capability check. To hold R and to check an operand it needs
two rules, enabled together by the QEMU property `x-capstone-supervisor-refs`:

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

Plan v5 (decision 7) adds a third rule and a hardening for system calls; QEMU
implements neither yet. In S, a load, store, AMO, LR/SC, `LDC` or `STC` whose
base register holds a reference passes the checks of U against the table that
`srevroot` names, and a stale or out-of-bounds access raises its cause in S.
While `srevroot` is nonzero, a plain load or store of S to a user page faults
whatever SUM says. Today QEMU treats a tagged base in S as a plain address
(`translate.c:150`, `trans_capstone.c.inc:271`).

## Entering and leaving the kernel

Decided on 2026-11-12. The kernel saves the registers of a program itself.

A trap leaves the program's registers in place and enters the kernel as any
RISC-V trap does. The kernel stores the registers with their tags into its
own memory with `STC` and loads them with `LDC` before it returns. On a
switch to another process it also writes the table root, which clears the
processor's lifetime caches. Each thread has its own save area in kernel
memory.

Saving and restoring need no rule beyond supervisor references: a kernel that
holds R can save a register the same way. Plan v5 adds two processor rules
around them (decisions 1 and 2): U is protected exactly when `srevroot` is
nonzero, and sscratch and mscratch keep their tags. The kernel also saves one
designated register, t6, with its tag at every trap from S, in a 16-byte slot
of the trap frame, because a routine for user memory holds a program reference
in t6 across a page fault (section System-call buffers). The design replaces
the supervised entry of today, in which the processor saves all registers
into a frame at every event (`CSRUNV`, its frame format, event kinds, actions
and context slots). Three
conditions hold in both forms: a paused program cannot reach its saved
registers, the registers keep their tags, and a change of process changes
the table root.

Not implemented. QEMU has the trap path as a test mode (protected U, which
the address-space root gate runs in). The adapter and the RTL use `CSRUNV`.
Steps 1 to 3 of plan v5 implement the design.

## The program counter

Decided on 2026-11-13 (plan v5, decision 3). It replaces the code reference
of 2026-11-12. The program counter is a plain address, and no reference
covers the code. A fetch passes the page table and nothing else: the page's X
bit decides whether it executes, as the page table decides every read and
write right. The kernel maps only the text of the program image executable
and refuses `PROT_EXEC` on every other mapping (section Mappings). The image
is a kernel-owned block, so the program can neither unmap its text nor change
its rights. Jumps and returns are plain `jal` and `jalr`, and the trap's saved
pc stays an address. The design does not constrain where a program jumps
inside its own code. Today QEMU still checks a program's fetches against its
code capability in virtual C (`translate.c:1525`), and step 11 of plan v5
removes that check.

## System-call buffers

Decided on 2026-11-13 (plan v5, decisions 4 and 7), after review. It
replaces the libc's copies through the shared region and the direct path of
2026-11-12. The kernel accesses the program's buffer in place, through the
reference that the program passed for it, and the processor checks each
access. The data moves once, as in stock Linux.

Bind. Inside the kernel a user pointer is a handle: the 38-bit address in bits
37 to 0, zero in bits 39 and 38, and above them the number of its saved
reference. Only two places create handles. The entry code writes one for each
pointer argument into the integer view of the saved registers: numbers 1 to 6
name the slots of a0 to a5, which already hold the tagged arguments, and an
untagged value in a pointer position gets number 0, which names no reference.
A compat-style loader serves the structures that hold pointers (iovec, msghdr
and mmsghdr, argv and envp). It reads the user layout, with 16-byte pointers,
through the outer reference, loads each embedded reference with its tag by
`LDC` into a per-call list, and writes the kernel layout with handles numbered
from 7. The kernel clears the list at the end of the call, so a handle is
valid only during its call. The pointer positions come from the
`SYSCALL_DEFINE` types, and an allow-list of about 100 calls adds those the
types miss (the address arguments of the mmap family, and arguments whose
meaning depends on a command). The kernel refuses every other call with
`ENOSYS`. A handle only names a reference that the program passed in this
call, so it creates no authority. An integer read from user memory never
becomes a handle, since the loader takes an embedded reference by `LDC` with
its tag and gives an untagged one number 0. A handle used as a plain address
is not canonical in Sv39 and faults.

Check. The kernel's routines for user memory (`copy_to_user`,
`copy_from_user`, `get_user`, `put_user`, `clear_user`, the string routines
and the futex atomics) load the reference that the handle names into t6, set
its address, access through it and clear t6. The processor rule of section
Supervisor references checks each of these accesses as it checks the
program's own: tag, bounds and lifetime against the table that `srevroot`
names. A stale or out-of-bounds access raises its cause in S (25 or 28), and
the kernel's exception table turns it into `EFAULT`, or into the residual
count of a partial transfer. A page fault during such an access is an
ordinary page fault: Linux resolves it and may sleep, and the processor checks
the reference again at the retry. So a buffer that another thread releases
while the call sleeps makes the call fail, and so does a `munmap` and an
`mmap` at the same address in that time. The firmware refuses to emulate a
misaligned S access through a reference, and the routines never issue one.

Hardening. For a protected address space, SUM does not open user pages to
plain addresses: while `srevroot` is nonzero, a plain load or store of S to a
user page faults. A kernel path that nobody ported then fails with `EFAULT`
instead of bypassing the check.

Fallback. If the processor rule slips, in QEMU or on the RTL, the binding
stays and the routines check in a kernel-checked window instead. They read
bounds and lifetime id with `LCC` and the generation from T, access by plain
address with interrupts, preemption and page faults off, and check again from
scratch after a page fault. The window accesses by plain user address, so the
hardening stays off under the fallback, and an unported path then bypasses
the check silently.

Refused. The kernel refuses every path that pins user pages and accesses them
through its own map (O_DIRECT, `vmsplice`, `io_uring`, `process_vm_*`,
ptrace, `/proc/PID/mem`). It keeps a user pointer that outlives its call as a
full 16-byte reference (the exit word, the alternate signal stack), or it
refuses the call (robust lists, rseq, aio). The kernel returns `epoll_data`
and `sigval` as integers, so the libc keeps a reference in them in a side
table and passes the table's index.

Rejected after review: a search of the passed references for one that covers
the address. The search is unsound. In `recvfrom` with a 64-byte buffer and
the live `addr` object right behind it, the second chunk of a long datagram
passes on `addr`'s reference, and bytes land past the buffer. It is also
incomplete: a stale leftover reference in an unused argument register that
covers the address of a live buffer rejects a valid call. Morello's PCuABI
makes `__user` pointers capabilities in the kernel. It needs a compiler with
hybrid capability types, which llvm-capstone does not have, so handles carry
the binding through the kernel's plain C instead.

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

With root grants (module parameter `root_grants=1`, stages 1 to 4f and block
A of the adapter plan), the adapter creates R with the table and derives every
grant from R, but it still keeps a reservation record M per mapping (section
Mappings, "The adapter until M goes"). Without root grants, the default, the
adapter creates one root per mapping with `CSMINT`, which makes a private
parent record and a child, and it ends a mapping with `CSRETIRE` on the
parent's number. The kernel then holds no reference, and `munmap` names the
mapping by its registered range and not by a reference. In both modes the
program runs through `CSRUNV`.

## Work that follows from the decision

1. Adapter: create R when it creates the table, make every registered range
   (image, stack, startup data, exchange, `mmap`) a child of R, check and end
   the operand at `munmap`, carry `mremap` over, and drop the private parent
   records and their collection paths. Stages 1 to 4f did this with M, and
   the next adapter stage removes M.
2. libc: `munmap` and `mremap` hand the reference to the kernel.
3. Model and contract tests for the adapter path, then the landing gates, the
   port gates and the bug corpora.
4. RTL: neither rule of supervisor references exists there, nor the rules of
   plan v5 (protected U from `srevroot`, tagged scratch CSRs, the check of S
   accesses through a reference and the hardening). Until it follows, a kit
   runs on QEMU or on the board and not on both, as with the one-operand
   `CREVOKE`.
5. Block B: the kernel runs a protected program through ordinary traps,
   without launcher, `CSRUNV` or monitor. Plan v5 lists its steps.

## Further decisions of 2026-11-12 and 2026-11-13

Plan v5 lists the steps for the decisions of 2026-11-13.

- Signals. The kernel writes the signal frame with tagged registers through
  the saved stack reference, with ordinary accesses that the processor checks
  (section System-call buffers). `sigreturn` restores capabilities only from
  that frame and validates the restored execution state.
- Long releases. Above a threshold of records, `CREVOKE` traps to the
  kernel, which finishes the walk preemptibly. No thread of the process runs
  until the walk is done, and the release takes effect when every
  generation is advanced and the lifetime caches are invalidated, before any
  slot or memory of the released objects is reused.
- Resource limits. Allocation can fail, and the abstract machine gets that
  outcome: `CDERIVE` reports failure and `malloc` returns NULL, no fault. A
  slot at its last generation retires. `CRENEW` on such a slot takes a free
  slot for the new lifetime before any lifetime ends. When no slot is free and
  the table cannot grow, `CRENEW` fails with every record unchanged: it
  writes its failure result, an untagged 0, to rd, and the old object stays
  live. With rd equal to rs1 the failure result replaces the register's copy
  of the reference (capstone-qemu `helper_csrenew`), so an allocator keeps the
  old reference in a separate register until `CRENEW` succeeds. The adapter
  side is in the section Mappings (block A: `ENOMEM` for `mmap`, `NULL` for
  `malloc`, the kernel's growth and NOGROW).
- `fork`, as a design only (plan v5, decision 10). posix_spawn covers the
  programs, and the prototype does not fork a protected process. In the
  design the child gets a copy of the lifetime table with the same slots and
  generations, since the processor writes the table through physical
  addresses and Linux's copy-on-write does not cover it. The copy is one
  snapshot: the kernel forces the record cache to write back and holds the
  program's other threads off until it has copied the table and the memory,
  so no lifetime ends in one copy and not in the other. The copy costs 16 bytes
  per slot of the table's high-water mark.
- Swap of pages with few references, a planned late step (plan v5, step 10).
  While a page is out, the kernel keeps the page's references in its own
  memory, 16 bytes per reference, and moves them out and back with the
  existing raw moves (`LDC` and `STC` with a plain kernel base), so no
  instruction makes a reference from numbers. It swaps out only a page in
  which at most about one eighth of the granules hold a reference (32 of 256);
  every other tagged page stays resident. T stays resident, so a reference
  whose object the program released while its page was out is stale after
  swap-in.
- Shared pages. A page that two address spaces share carries no references.
  The kernel refuses such mappings today, since it accepts only anonymous
  private mappings. Later a page attribute can make a tagged store to such a
  page lose its tag.
- More than one hart and lifetimes for stack objects: out of scope. One hart
  stays (plan v5, decision 12). The claim covers heap objects and nested
  allocations of statically linked programs.
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
  free without losing a tag, and CSCRATCH holds the monitor's stack. Step 2
  of plan v5 patches it: it swaps through mscratch, which keeps its tag under
  plan v5, and saves every register with `STC`.

## Open design points

- Exact bounds.
