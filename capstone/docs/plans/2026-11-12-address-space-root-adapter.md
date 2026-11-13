# Address-space root: adapter and libc plan

Status: plan of 2026-11-12 on the branch `address-space-root`. Stages 0 to 4d
are done (see each stage); stage 4e (partial `munmap` as a page operation) is
next, then block A (the format alignment) and stage 5, in a reduced form until
the project lead decides on the old path.
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
   commit, which cannot fail with root grants. A stack that is the thread's own
   frame is the exception (stage 4): the module refuses it as a stack, and the
   frame's release still ends that mapping. `mprotect` passes the
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
     keeps the stack mapped. The launcher tells its own
     joiners that the thread is done only after the frame and stack releases
     and the store and wake of the exit word. An exit with an exit word frees
     its table entry at once, unless a joiner already waits on it. The module
     refuses a stack that is the thread's own frame before every other check
     (`exit_kept`, reason "is the thread's frame"); the frame's release then
     ends that mapping, once.
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
     ENOSYS. The transaction stays refused with root grants (stage 4b deletes
     it).
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
4b. **Fixes after the review of stage 4. Done** (2026-11-13).
   - Thread table: the last one to leave frees an entry. An exit frees its
     entry unless joiners wait; then it marks the entry done and wakes them,
     and the last joiner to leave frees it. Earlier joiners leave it alone, so
     two joiners of one thread both return and the entry is freed once. A join
     answers 0 for every thread that has exited, before or after the call: the
     module never reuses a thread id, so an id at or below the highest one
     issued without an entry has exited. An id never issued answers ESRCH, the
     caller's own EDEADLK. Exits without an exit word that are never joined
     free their entries, so they cannot fill the table. `thread_remove` and the
     `remove` parameter are gone; a rolled-back create frees its entry as an
     exit does, after the frame. The table (32 entries) now fills only with
     live threads, as the module's limit (32 threads) does, so the module
     refuses first (ENOSPC, the frame stays the program's); the launcher's
     rollback after the module took the frame is tested behind a launcher with
     a 4-entry table.
   - Exit word: the launcher writes it only if it is an aligned int inside one
     registered mapping, in pages the program's grant may write and Linux maps
     read-write (each mapping keeps a conservative writable range that
     `protect_range` updates), outside the thread's frame, and admitted by the
     module. Overlap tests are overflow-safe. A word that fails is not written,
     the launcher logs `CAPSTONE_VM_EXIT_WORD_REFUSED` with the reason, and the
     exit completes; joiners through the launcher return at its completion. A
     word inside the stack keeps the stack mapped, as in stage 4.
   - Attach: a worker's attachment requires both transport blocks inside the
     writable range (`CAPSTONE_VM_TRANSPORT_REFUSED` otherwise, and the launcher
     ends the program, as for every malformed transport). Held mappings still
     answer EBUSY to `mprotect`.
   - `mremap`: before it copies anything the libc asks the new mapping check
     (`CV_SERVICE_MAPPING_CHECK`, 15): the launcher's gate as for munmap, the
     module's whole-reference check through `CV_PROTECT` (which reads the
     service from the caller's a7), and ENOTSUP unless the mapping is
     read-write over its whole length. Nothing changes. Then map, copy at most
     the old visible length and unmap the old mapping. A same-size resize
     returns p after the check. A failed cleanup unmap of the new mapping traps,
     here and in `__heap_remapping`, which asks no check (its group and length
     are the allocator's own). Limits: only read-write anonymous mappings move,
     and a write of another thread during the copy may be lost, as for
     `realloc`.
   - The remap transaction is deleted for both `root_grants` values: the
     module's `remap_begin`, `remap_end` and its wait, the ioctls 25 and 26 and
     `struct cv_remap`, the launcher's `remap_mapping`, service 13 (it answers
     ENOSYS) and the `__capstone_vm_remap` stub.
   ABI: service 13 removed, service 15 added, ioctls 25 and 26 removed; the
   other ioctl structures keep their layout. The launcher's stats print a
   thread census, `CAPSTONE_VM_THREADS count=... issued=...`.
   Unit test: root-contract.c, five new modes (`join-two`, `join-after`,
   `exit-many`, `exit-word`, `attach-readonly`), `mremap` extended (an
   oversized old length, a same-size resize, a stale same-size reference, a
   read-only mapping) and `create-fail` rewritten for live threads. Two
   batches on the final images, equal counts: root_grants=1 274 of 274 pinned
   and unpinned, root_grants=0 270 of 270, hosted contract 94 of 94, mapcache
   park and revive, a launcher without its length check 90 of 90, a launcher
   that names the frame around a0's cursor 30 of 30, the 4-entry table 50 of 50
   with and without root grants. Controls: the stage 4 thread table makes
   join-two's second joiner hang, the join after a worded exit answer ESRCH
   and the 32nd unjoined create fail; without the writable range the launcher
   faults at the read-only exit word and at the read into the read-only
   exchange; without the alignment check the launcher's store to the unaligned
   word traps to the firmware, which halts the machine at that address; a libc
   whose mremap copies first takes a bounds fault (28) at the oversized old
   length. The stage 4 controls still fail where expected.
4c. **Every service through a checked reference. Done** (2026-11-13). Rule,
   decided after the review of stage 4b: the kernel side (module and launcher)
   reads or writes program memory, or holds a program mapping, only through a
   reference that the program passed and the module checked, or through the
   system-call region it shares with the program (the process transport). The
   module checks bounds and lifetime only; read, write and execute rights are
   the page table's (design note, section Mappings).

   Audit of every launcher service and module path that touched program memory
   at an address the program named, read at `29672da5e` (line numbers of the
   stage 4c tree):

   | Path | Before | Decision |
   |---|---|---|
   | exit word, `CV_SERVICE_PTHREAD_EXIT` a0 (exec.c `exit_word_refusal` 982, store and wake 1572) | integer; written if registered, aligned, writable, outside the exiting thread's own frame | (Superseded in stage 4d: check and store in one module step.) a0 is the word's reference. Launcher: inside one mapping, writable pages, outside every transport it writes itself. Module (`CV_WORD`, capstone_vm.c `word_check` 3814): with root grants a live non-linear reference covering the word; in both modes aligned, inside one arena past the image text, in no thread's frame. Then `CV_TOUCH`. A refused word is not written, logged, the exit completes; futex waiters on it stay blocked (program bug) |
   | futex wait, `CV_SERVICE_FUTEX` a0 (exec.c 1581, `syscall(SYS_futex)` 1603) | integer; Linux read the word if registered and aligned | (Superseded in stage 4d: the module compares, the waiter sleeps on a launcher key.) Linux reads the word, so `CV_WORD` checks it as an exit word (no write conditions); EFAULT and a log line otherwise. The wait runs with `service_lock` released after the check: a concurrent unmap by another thread of the program affects only that program's wait |
   | futex wake, same service | integer | (Superseded in stage 4d: wakes a launcher key.) stays an integer: a wake neither reads nor writes the word (Linux hashes the address of a private futex). It must be an aligned int inside one registered mapping (EFAULT otherwise), which keeps it off the launcher's own futexes |
   | worker attach, `CV_SERVICE_THREAD_DELEGATE` a0, a1, a2 (exec.c `transport_attach` 1435) | integers; held and written if registered, writable, not held | a0 and a1 are references. Each block a whole mapping the program may end (`mapping_gate`, which refuses the launcher's own transport: no heap mapping), the two distinct, every page writable, and with root grants both references pass the whole-mapping check of munmap (`CV_ATTACH`, capstone_vm.c `attach` 3866), neither in a thread's frame; then `CV_TOUCH` and the hold. A refusal ends the program |
   | process transport, same service (exec.c 1622) | integers checked by registration, compared with the launcher's own | the shared system-call region: compared with the launcher's own meta block and exchange region, never resolved; registration check dropped |
   | an attached worker's later calls (exec.c, after 1622) | compared with the attached blocks | unchanged: equality with blocks checked at the attachment |
   | `CV_SERVICE_DELEGATE`, `capstone_delegate_serve` (exec.c 1659, linux/delegate-service.c) | the launcher's own meta block; exchange offsets checked against its length | unchanged: shared region only |
   | `exchange_prepare_write` (exec.c 911) | `CV_TOUCH` of the process exchange | unchanged: shared region |
   | park requests (linux/delegate-service.c `park_request` 775) | integer key | unchanged: the key is never dereferenced (park.h) |
   | `madvise` | the domain answered ENOSYS (ports/musl-capstone/runtime/delegate.c 1566, group memory) and the launcher's validator refuses the memory group (common/delegate.c `capstone_delegate_validate` 337) | new service 16 (`CV_SERVICE_ADVISE`, exec.c `advise_range` 861, capstone_vm.c `advise` 3894) for `MADV_DONTNEED` and `MADV_FREE`, with a reference that passes conditions 1 to 3 and covers the page range, as mprotect; libc `madvise` in mapping.c. Other advice still answers ENOSYS in the libc |
   | `set_tid_address`, `set_robust_list` | answered in the domain (ports/musl-capstone/runtime/delegate.c 1385, 1528) | unchanged: never reach the launcher |
   | thread create a0, stack at exit a1, munmap, park, mprotect, mapping check (exec.c 1481, 1558, `unmap_arena_for` 778, `protect_range` 839, `check_mapping` 880) | checked references since stages 2 to 4b | unchanged |
   | fault diagnostics, `frame_chain` (exec.c 1356) | reads the main stack at the faulting thread's fp | unchanged and documented: a diagnostic after the program stopped for good, bounded to the main stack (a launcher grant, never unmapped while the process lives), written to the fault record and never returned to the program |
   | map, join, self, wait, nodes, detach | no program memory | unchanged |
   | module `CV_TOUCH` (capstone_vm.c `touch` 3647) | pins pages the launcher names | unchanged: the launcher names only spans checked above or the process exchange |
   | module `CV_THREAD_CREATE` `pin_thread_frame` (capstone_vm.c 4190) | pins the frame a0 names | unchanged: `request_accept` with root grants since stage 3 |
   | module `resolve` (capstone_vm.c 3325) | the faulting address | unchanged: the processor checked the reference before the fault |

   Other changes:
   - Writable state: the conservative interval (`write_from`, `write_to`) is gone.
     Each mapping records the protection Linux maps on every page of its visible
     length, exactly: one value while all pages share it (no allocation for the
     mappings that no mprotect changed in part), one byte per page after the
     first partial change, back to one value after a whole-mapping change. A byte
     and not a bit: three protection bits and room for stage 4e's removed state;
     bounded by 512 KiB per mapping and 1 MiB per context (`CV_MAX_REGION_BYTES`,
     `CV_MAX_BYTES`). Interface for stage 4e: `page_split`, `page_set`, `page_and`,
     `page_all`, `page_get`, `page_free` (exec.c, section Page state).
     `protect_range` intersects old and new rights before mprotect and sets the
     new ones after it. A page counts as writable when Linux maps it with
     `PROT_WRITE` (RWX included), and mremap's check asks for exactly read-write.
   - Join: the module issues a thread id only when the create succeeds, and the
     launcher's rollback of the newest create gives it back (capstone_vm.c
     `thread_exit`); the launcher publishes an id only after `pthread_create`
     succeeded (`thread_publish`, `thread_withdraw`). A failed create consumes
     no id, so a join of it answers ESRCH until the next create takes it. The
     main thread's join of itself answers EDEADLK (the EDEADLK test now comes
     first). `registered_word`, which the reviewers called dead, was the futex
     service's check; it is folded into `word_mapping`. Dead code removed:
     `registered_span`, `writable_cut`, `writable_join`, the frame parameter of
     `exit_word_refusal`.
   - `die()` runs its cleanup once: a second failing thread waits for the exit.
     Before, an attachment refused by a worker while the main thread stepped
     could free the host state twice (one SIGSEGV instead of exit 125 in the
     first unpinned batch).
   - mremap's check-to-copy window is documented (mapping.c, design note).
   ABI: `CV_SERVICE_PTHREAD_EXIT` and `CV_SERVICE_FUTEX` take the word's reference
   in a0 (`__capstone_vm_thread_exit_clear(int *, ...)`), a worker's
   `CV_SERVICE_THREAD_DELEGATE` takes references in a0 and a1, service 16
   `CV_SERVICE_ADVISE` is new; ioctls 34 `CV_WORD`, 35 `CV_ATTACH` and 36
   `CV_ADVISE` are new. The other ioctl structures keep their layout.
   Unit test: root-contract.c, six new modes (`exit-word-refs`, `futex`,
   `advise`, `join-burned`, `attach-exchange`, `attach-address`), `exit-word`
   through references, `create-fail` with a join of the failed id.
   `pthread-contract.c` passes its exit word as a reference too (compiled, its
   gate not run in this stage). Two batches on the
   final images, equal counts: root_grants=1 374 of 374 pinned and unpinned,
   root_grants=0 370 of 370, hosted contract 94 of 94, mapcache park and
   revive, the launcher without its length check 96 of 96, the launcher that
   names the frame around a0's cursor 32 of 32, the 4-entry table 68 of 68 with
   and without root grants. The first batch, before the `die()` fix, read 373 of
   374 unpinned (`attach-readonly` ended with SIGSEGV after the refusal); after
   it, three unpinned runs of the attach modes and `advise` read 51 of 51 each.
   Controls: with stage 4b's exit-word check (by address alone) the word in
   another thread's frame is written, with and without root grants, and that
   thread resumes at a return address whose low half is zero and faults (12);
   with stage 4b's single writable interval the word in the third page of the
   middle-protected mapping is refused ("not writable") and the program fails
   there; the module controls of stage 3b and 4 (condition 3, the stack's
   CREVOKE, the `mprotect` check, the a0 cursor at create) and the hold control
   still fail where expected, and condition 3's control now also breaks
   `advise` at the nested object.
4d. **Check and access in one step. Done** (2026-11-13). Rule, decided after the
   review of stage 4c: every access of the kernel side to program memory through a
   program reference happens inside one module ioctl under `vm_lock`, which also
   excludes the program's execution (`CV_STEP` holds it while the program runs):
   the module checks the reference and performs the access in the same call, so no
   other thread can end the lifetime, unmap or remap in between. The end version
   has this by construction (the kernel accesses through the reference, and the
   processor checks it at the access). Stage 4c checked in the module and accessed
   later: the launcher stored the exit word after `transport_detach` and
   `CV_THREAD_EXIT`, while another thread could `CREVOKE` the word's object and the
   allocator reuse it; and Linux read a futex word inside `SYS_futex` after
   `service_lock` dropped, while another thread's whole-mapping munmap could put a
   new program or launcher mapping at the address (an equality oracle: EAGAIN or
   sleep).

   Audit of the paths that changed (the rest of the stage 4c table holds):

   | Path | Stage 4c | Stage 4d |
   |---|---|---|
   | exit word, `CV_SERVICE_PTHREAD_EXIT` a0 | `CV_WORD` checked, `CV_TOUCH` admitted, the launcher stored later through Linux | the launcher checks its own conditions (one mapping, not metadata, writable pages, no transport) and asks `CV_THREAD_EXIT` with `word` = 1; the module checks the reference while the frame holds a0 (`word_check`), and at the end of the same ioctl, after the frame and stack releases are committed, checks again what those releases change (lifetime, arena id, frames), admits the page (`touch`) and stores 0 with `put_user` through the program's page table (`word_access`); the outcome comes back in `word_reason` |
   | futex wait, `CV_SERVICE_FUTEX` a0 | `CV_WORD` checked, then Linux read the word at the program's address with `service_lock` released | `CV_WORD` checks, admits, reads with `get_user` and compares with a2 in one ioctl (`futex_compare`); if equal the waiter sleeps on the launcher's key (arena id, offset) with a private futex on its own word (`futex_wait`); Linux never sees the program's address |
   | futex wake, same service | an aligned int in a registered mapping; Linux woke the program's address | the same address check; it wakes the launcher's waiters of the key of the mapping that holds the address now (`futex_wake_key`), reads and writes nothing, needs no reference |
   | exit word's wake | Linux `FUTEX_WAKE` at the program's address | the launcher's key, after the releases, then `thread_finish` |
   | words in the allocator's metadata | accepted (`word_check` did not exclude them) | refused by the launcher, for the exit word and the futex wait, as `range_mapping` refuses them |
   | pinned `madvise` | `pin_resident` then zero the held pages; a resident page it did not hold in a `PROT_NONE` VMA failed with EFAULT | held pages zeroed after a check that each is still exclusive anonymous memory; every other page with contents zeroed in place under its page-table lock (`clear_unheld`, `walk_resident` with clear), whatever the VMA's protection; no GUP |
   | fault diagnostics, `frame_chain` | documented in the plan | stated in the design note as the one diagnostic exception: every read inside the main stack's mapping, the program stopped for good |

   No lost wakeup: the waiter compares and enters the launcher's queue in one hold
   of `service_lock`, and a wake needs `service_lock`, so a wake after the
   program's store either precedes the compare (which reads the new value) or
   marks the queued waiter, whose private futex wait then returns at once.

   Other changes:
   - Join: `thread_withdraw` marks an entry with waiting joiners withdrawn and
     broadcasts; its joiners return ESRCH and the last one frees it; lookups skip
     withdrawn entries (`thread_entry`). The false comment at `thread_add` (joins
     start under `service_lock`) is corrected: `thread_join` runs without it.
   - A refused `CV_THREAD_EXIT` in `thread_rollback` ends the program (`die`), so
     the module and the launcher never disagree about issued ids.
   - `die()` prints its line and calls `_exit(125)`; it frees and closes nothing
     (another worker may be inside `capstone_delegate_serve`); losers wait.
   - `grant_mapping` aborts unless the visible length and the image prefix are whole
     pages inside the grant. `ref_bounds` is the one register read of
     `request_operand` and `word_check`. The libc reads a futex word before a wait
     only; a wake reads nothing.
   - Test hooks (`exec.c`, `-DCAPSTONE_VM_TEST_HOOKS`, absent from the default
     build, checked by `strings`): delays in the futex wait after the compare, in
     the exit after the launcher's checks and in the join, a forced create failure
     and a refused rollback.
   ABI: `struct cv_thread_exit` gains `word` and `word_reason` (ioctl 23 changes
   size), `struct cv_word` gains `equal`, `arena` and `offset` and serves only the
   futex wait (ioctl 34 changes size); `CV_WORD_NOT_ADMITTED` (7) and
   `CV_WORD_NO_ACCESS` (8) are new. The services keep their numbers and arguments
   (the controls run the stage 4d program on stage 4c's launcher and module).
   Unit test: root-contract.c, new modes `futex-remap`, `die-blocked`,
   `word-metadata`, `advise` extended by a `PROT_NONE` page that held data, and the
   hook modes `futex-race`, `exit-race`, `join-withdrawn`, `rollback-fail`
   (run-root-grants.py `--launcher-hooks`). Batch d1 on the final images:
   root_grants=1 408 of 408 pinned and unpinned, root_grants=0 403 of 403, hook
   modes 51 of 51 pinned and unpinned with root grants and 50 of 50 without (the
   exit race needs root grants), the pthread gate 29 of 29 with and without root
   grants (first run of that gate on this branch), hosted contract 94 of 94,
   mapcache park and revive, the launcher without its length check 96 of 96, the
   launcher that names the frame around a0's cursor 32 of 32, the 4-entry table
   68 of 68 with and without root grants. Controls: stage 4c's module and launcher
   with the two delays at their open windows: the futex waiter compares the new
   mapping's word (EAGAIN instead of a timeout), a wake of the new mapping wakes
   the old mapping's waiter, and the exit word's store reaches the object reused
   after the revoke (it reads 0 instead of 0x7777); the hooks launcher with stage
   4c's `thread_withdraw` leaves the joiner of the withdrawn id waiting; the
   module controls of stages 3b and 4 (condition 3, the stack's CREVOKE, the
   `mprotect` check, the a0 cursor at create) and the launcher controls no-hold and
   writable-interval still fail where expected. Stage 4c's launcher control
   exit-word-4b patches `exit_word_refusal`, which is gone, and was not rebuilt.
4e. **Partial munmap as a page operation** (next). A partial `munmap` inside a
   mapping block removes pages and leaves a `PROT_NONE` placeholder; no lifetime
   ends, and no other mapping gets addresses inside the block while M lives. M
   ends with `CREVOKE` when the last logically mapped page goes, when one call
   names the whole block, or at exit. Exact per-page accounting on the page
   state of stage 4c: `PROT_NONE` and untouched pages count as mapped, and a
   repeated unmap of a hole is idempotent. `MAP_FIXED` into a hole only under
   today's `MAP_FIXED` rule (an accepted reference, no object derived below P).
   `mremap` shrinks in place and grows in place within M. The module's frame
   set needs deletion (it is insert-only today, `frame_insert`), and the module
   clears the tags of each removed page range before it unpins the pages.
A. **Format alignment: no capability rights** (block A, after stage 4e; decided
   2026-11-13). Sublet checks bounds and lifetime only, and the page table
   checks mapping and rights. Remove capability rights from the module and the
   launcher: the grant's permissions (0 to 7, `cv_map.permissions`),
   `CV_TIGHTEN` and the rights switch of `root_derive` (capstone_vm.c 1569,
   1949, 2004), the rights checks of `add()` against the VMA (2493 to 2535) and
   of `revive()` (2342), the exec.c kind-dependent rights checks of
   `acquire_mapping` (996 to 1003), `CV_GRANT_WRITE` and `grant_writes`
   (exec.c 57 to 61, 577, 887, 937), the rights arguments of the grants
   (exec.c 1918, 1937 to 1944; `__capstone_vm_map` in start.S 64;
   `cap_vm_acquire` in vm.h, mapping.c 31, malloc-mapping.c 58, thread.c 62)
   and the "maximum rights" comment in mapping.c 29. Places that rely on a
   capability right for protection today, so the kernel must map them with the
   matching page rights instead:
   - the startup block (exec.c 1940): a read-only grant (4) over pages Linux
     maps read-write; the launcher must map it `PROT_READ`, and its own
     `grant_writes` gate goes with it;
   - the image (exec.c 1918): one grant with all rights over text (Linux maps
     read-execute) and data (read-write), and the module requires the execute
     right for register 3 (capstone_vm.c 2500); the page table already matches;
   - heap, metadata, stacks, frames, the transport and the heap region (rights 6)
     are mapped read-write without execute; the page table already matches, but
     a program's `mprotect(PROT_EXEC)` of such a mapping then makes it
     executable, where today the missing execute right refuses the fetch;
   - application mappings (rights 7) already rely on the page table;
   - without root grants `CSMINT`'s descriptor carries the rights (capstone_vm.c
     2384, 2648), and the processor checks them. The design note's section on the
     program counter states the decided rule since stage 4d: Sublet checks only the
     bounds and the lifetime of the code reference, and the page's X bit decides
     execution; the module's execute-right requirement for register 3 goes here.

   Decided with block A (2026-11-13): with generational ids and no parent field
   in the record, `ref_child` (capstone_vm.c) decides "P is a child of M" without
   reading a parent. P is the child of M if and only if P's record directly
   follows M's in the preorder list, depth(P) = depth(M) + 1, and P's generation
   matches. This is sound because only the kernel holds M, it derives P right
   after M in the same run (`root_derive`), `CRENEW` keeps a record's position,
   and every derivation of the program lies deeper, below P.
5. **Remove the old path.** Drop the parameter, `arena.ancestor`, the private
   parents and the collection paths that exist only for them. `CSMINT` stays
   at one site. Pending a decision of the project lead (2026-11-13): the
   root_grants=0 path stays for now, since the FPGA bitstream lacks the new
   instruction rules (supervisor references), and without that path the
   adapter does not run on the board. The remap transaction, which neither
   path called, is already gone (stage 4b).
6. **End to end, once.** After the open design points that we implement are
   in as well: landing gates, port gates, bug corpora, Juliet, the reuse
   study. Regenerate the recorded results.

## Not in this plan

The native task route, the RTL, and the open design points of the design
note (context switch, system-call buffers, signals, `fork`, harts).
