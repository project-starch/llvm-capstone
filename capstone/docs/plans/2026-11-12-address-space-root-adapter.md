# Address-space root: adapter and libc plan

Status: plan of 2026-11-12 on the branch `address-space-root`. Stages 0 to 4
are done (see each stage); stage 5 is next, in a reduced form until the
project lead decides on the old path.
Design: [the kernel as the outermost allocator](../design/virtual-capstone/address-space-root.md).

## What the code does today

Read at `41f94d8eb7`, in `runtime/virtual/module/capstone_vm.c`,
`runtime/virtual/vm.h`, `mapping.c` and `start.S`.

- `add()` registers a range and executes `CSMINT`. The instruction makes a
  private parent record and a child, and writes the child's capability into a
  register slot of the thread's saved frame. The adapter keeps the parent's
  number in `arena.ancestor`.
- Every mapping that a running program requests arrives as a **linear**
  capability in a0 (`r->linear == 1`, `r->reg == 10`). The libc moves it into
  a slot (`cap_vm_acquire`) and makes it copyable with `DELIN`
  (`cap_vm_copyable`). Startup grants (image, stack, startup data) are minted
  non-linear.
- `munmap` in the libc passes the address and the length, not the reference
  (`__capstone_vm_unmap(address, bytes)`). The launcher resolves the range to
  an arena number and calls `CV_RETIRE`. `retire()` executes `CSRETIRE` on the
  parent's number.
- The mapping cache (`park()`, `revive()`), `mremap` (`remap_begin`,
  `remap_end`), thread frames (`ancestor_retired`, S-21) and teardown use the
  parent's number as well. There are 12 `CSMINT` or `CSRETIRE` sites.

## Target

- The adapter creates R, one reference over the user address range, when it
  creates the table, and keeps it in a 16-byte slot of pinned kernel memory.
- Every grant is a child of R made with `CDERIVE`, with its address moved by
  `CINCOFFSET` and its rights reduced by `TIGHTEN`, and stored into the frame
  slot with `STC`. No grant is linear.
- The libc hands the reference to `munmap` and `mremap`. The adapter reads it
  from the saved a0 of the calling thread, checks the four conditions of the
  design and executes `CREVOKE`.
- The adapter keeps no identity per mapping. Teardown frees the table.

## Stages

Each stage ends with one short unit test on QEMU: a contract program of a
few dozen lines, one boot, with a negative control that shows the test
measures the change. A stage also runs one smoke test of the unchanged path.
No stage runs a full gate. The gates, the corpora and the studies run once,
in the last stage, when every feature is in.

0. **Instruction rules in QEMU. Done** (capstone-qemu `6bad7a2ee7`,
   `tests/address-space-root`, 22 of 22).
1. **R and grants behind a module parameter. Done** (2026-11-12, see the
   commit "Adapter stage 1"). `root_grants=1` (default 0):
   create R with the table, make every grant a child of R, store it with
   `STC`. The libc accepts a copyable grant where it expects a linear one.
   `munmap` stays on the range in this stage: the adapter keeps a copy of
   each mapping's reference only to end it, and this copy goes away in
   stage 2. Unit test: a mapped page works through the grant, `malloc` and
   a second `mmap` work on copyable grants, an object derived in a page
   faults with 25 after the page's `munmap`, the other mapping lives on.
   Control: without the `CREVOKE` the access succeeds.
2. **Unmap by reference, with reservations. Done** (2026-11-12, commit
   "Adapter stage 2"). The kernel keeps its per-mapping copy. With a paged
   table without generations, CRENEW moves a mapping to a new record, so the
   operand is accepted if its id equals the copy's, or if the copy has ended
   and the operand is a live direct child of R that covers the call's range.
   The covering check uses the length the call names, since the public mmap
   narrows its reference to the requested length. The libc passes the reference
   in a0, the adapter reads it from the frame and applies the conditions of
   the design note (section Mappings): live, directly under R, the
   reservation's lifetime, bounds covering the whole reservation. A partial
   unmap is refused. The reservation object keeps the extent and the mapped
   page count, and VMAs of different reservations never merge. Unit test: the
   cases of the QEMU gate in one program (nested object, stale reference,
   address, narrowed reference, renewed mapping). The per-mapping copy of
   stage 1 goes.
3. **Reservation records and requests with references** (decided after the
   review of stage 2, 2026-11-12). Each mapping gets a kernel-private record M
   between R and the program's reference P. Every program request that ends a
   mapping comes with P and ends M: munmap, park, the stack at thread exit
   (today an unauthenticated unmap by address through `pthread_exit`, which
   must check the stack reference before the frame is cleared), the frame at
   thread creation (kill last, after every fallible step). The module checks
   the visible length itself. Done (2026-11-12, commit "Adapter stage 3"):
   `root_derive` makes M and P in one run; `request_operand` checks the four
   conditions for munmap and park (one `CV_UNMAP` decides park or retire),
   the stack at `CV_THREAD_EXIT` (checked before the frame is cleared, M ended
   after the thread stopped) and the frame at `CV_THREAD_CREATE` (M ended
   after every fallible step); `CV_RETIRE` releases only a reservation that
   has ended; after a failed create the launcher releases the frame and
   answers EAGAIN. Unit test: root-contract.c, eleven modes, root_grants=1
   pinned and unpinned 102 of 102, root_grants=0 88 of 88, hosted contract
   94 of 94, mapcache park and revive, the partial munmap refused by the
   module behind a launcher without its check; controls without condition 3
   and with the stack ended through the cleared operand fail where expected.
3b. **Fixes after the review of stage 3** (2026-11-12). `CV_THREAD_EXIT`
   releases the stack and the frame itself, after the thread has stopped and
   the frame is cleared, ending M first; the launcher only unmaps the ranges
   the module reports. A refused stack request does not stop the exit: the
   stack stays mapped and registered, counted in `exit_kept`. `CV_UNMAP` and
   `CV_THREAD_EXIT` decide first, write the outcome to the launcher, and then
   commit, which cannot fail with root grants. `mprotect` passes the
   reference, and `CV_PROTECT` checks it: conditions 1 to 3, and the bounds
   cover the named page range, which may be part of the mapping (musl opens a
   thread stack and a mallocng meta area past a guard page this way). Thread
   creation passes the launcher's mapping check first, and the module
   compares the frame with the cursor of the creator's a0. One helper checks
   every request. The record margin of a grant run counts its two records,
   the reserve and slack. Unit tests: three new modes of root-contract.c
   (`exit-length`, `protect`, `create-gate`); root_grants=1 pinned and
   unpinned 160 of 160, root_grants=0 157 of 157, hosted contract 94 of 94,
   mapcache park and revive; controls without the reference check in
   `mprotect` and without the a0 comparison at creation (behind a launcher
   that names the mapping around a0's cursor) fail where expected.
   Then the remaining items of the old stage 3:
4. **Fixes after the review of stage 3b, and `mremap`. Done** (2026-11-13).
   The mapping cache, the frame at thread creation and the stack at thread
   exit already take the program's reference since stages 3 and 3b. Partial
   `munmap` stays refused, as the design note decides. The changes:
   - Thread exit. The launcher checks the exit word against the frame and
     stack mappings that it found, over their whole extent, and no longer
     against the base and length that the program names. A wrong stack
     length, however large, never stops the exit. A word inside the stack
     keeps the stack, like every refused stack. The launcher tells its own
     joiners that the thread is done only after the frame and stack releases
     and the store and wake of the exit word. An exit with an exit word frees
     its table entry at once, unless a joiner already waits on it. The module
     refuses a stack that is the thread's own frame before every other check
     (`exit_kept`, reason "is the thread's frame").
   - `mprotect`. The launcher refuses a mapping that a worker's transport
     holds (EBUSY), as `munmap` does: it writes that mapping through Linux.
   - Thread creation. The module hands the cursor of the creator's a0 to
     `request_accept`, whose whole-mapping check compares it with the frame
     that the launcher names. The separate comparison is gone.
   - `mremap`. No libc code calls the module's remap transaction
     (`__capstone_vm_remap` in `start.S` and `vm.h`, `CV_SERVICE_REMAP`, the
     launcher's `remap_mapping`, `CV_REMAP_BEGIN` and `CV_REMAP_END`). The
     public `mremap` maps a new reservation, copies the contents with their
     tags and unmaps the old one through the program's reference
     (`mapping.c`). mallocng's realloc of a large group does the same through
     `__heap_remapping` (`malloc-mapping.c`, musl patch 0009), and musl's
     `pthread_getattr_np` calls `mremap` without `MREMAP_MAYMOVE` and gets
     ENOSYS. The transaction stays refused with root grants, and stage 5
     deletes it.
   No ABI change: the ioctl structures and the services keep their layout.
   Unit test: root-contract.c, three new modes (`exit-huge`, `exit-frame`,
   `protect-held`) and `mremap` rewritten (refused with a wrong old length and
   through an object over the mapping, then a growth and a shrink that keep
   the data and a stored reference, and the old reference faults with 25).
   Two batches: root_grants=1 193 of 193 pinned and unpinned, root_grants=0
   190 of 190, hosted contract 94 of 94, mapcache park and revive, a launcher
   without its length check 80 of 80 (the module refuses the wrong stack
   lengths and the wrong old length of `mremap`), a launcher that names the
   frame around a0's cursor 28 of 28. Controls: the old exit-word check stops
   the launcher at the huge length (exit 125); without the hold check the
   held mapping's `mprotect` succeeds; without the a0 cursor in
   `request_accept` a moved frame reference creates a thread; the stage 3b
   controls (condition 3, the stack's CREVOKE, the `mprotect` check) still
   fail where expected.
5. **Remove the old path.** Drop the parameter, `arena.ancestor`, the private
   parents and the collection paths that exist only for them. `CSMINT` stays
   at one site. Pending a decision of the project lead (2026-11-13): the
   root_grants=0 path stays for now, since the FPGA bitstream lacks the new
   instruction rules (supervisor references), and without that path the
   adapter does not run on the board. Until the decision, this stage deletes
   only the remap transaction, which neither path calls (`remap_begin`,
   `remap_end`, `CV_REMAP_BEGIN`, `CV_REMAP_END`, `CV_SERVICE_REMAP`, the
   launcher's `remap_mapping` and the `__capstone_vm_remap` stub).
6. **End to end, once.** After the open design points that we implement are
   in as well: landing gates, port gates, bug corpora, Juliet, the reuse
   study. Regenerate the recorded results.

## Not in this plan

The native task route, the RTL, and the open design points of the design
note (context switch, system-call buffers, signals, `fork`, harts).
