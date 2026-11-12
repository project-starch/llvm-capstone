# Address-space root: adapter and libc plan

Status: plan of 2026-11-12 on the branch `address-space-root`. Stage 0 is
done. No adapter or libc code has changed yet.
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
   the visible length itself. Implemented on 2026-11-12, not yet committed:
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
   Then the remaining items of the old stage 3:
4. **`mremap`, mapping cache, thread frames.** Found in stage 2: after the
   program renews a mapping with CRENEW, the kernel's copy no longer names it.
   Two paths still end mappings through that copy and must take the
   program's reference instead: thread creation (a frame mapping renewed
   before creation would keep a live reference to the protected frame) and
   the unmap of a thread's stack at thread exit (the range would return to
   use while the renewed lifetime lives). Carry each over or remove it
   with its reason, one unit test each. Decide partial unmap here (design
   note, open points).
5. **Remove the old path.** Drop the parameter, `arena.ancestor`, the private
   parents and the collection paths that exist only for them. `CSMINT` stays
   at one site.
6. **End to end, once.** After the open design points that we implement are
   in as well: landing gates, port gates, bug corpora, Juliet, the reuse
   study. Regenerate the recorded results.

## Not in this plan

The native task route, the RTL, and the open design points of the design
note (context switch, system-call buffers, signals, `fork`, harts).
