#!/usr/bin/env python3
"""Unit test of root grants: root-contract.c in one boot, one run per mode, with the module loaded
root_grants=1 (or 0 with --root-grants 0).

The processor needs supervisor references: set CAPSTONE_QEMU_CPU_EXTRA to the CPU properties of
the memory mode plus x-capstone-supervisor-refs=true.

Modes (root-contract.c lists the checks of each):
  unmap        munmap through the mapping's reference, a second munmap, the old reference at the
               reused address; ends with a stale read that must fault with 25 (exit 139)
  refuse       an integer address, an object over the whole page, a range beyond the mapping (a
               page range inside it is a partial munmap since stage 4e), a reference the program
               ended itself, the same after a collection; nothing faults (exit 0)
  narrow       a reference narrowed to part of the mapping (exit 0)
  renew        the reference from before CRENEW, then the renewed one; ends with a fault through
               an object derived in the renewed mapping (139)
  renew-ref    the same, and the fault is through the renewed reference itself (139)
  mremap       the libc's mremap: a range beyond the mapping, an oversized old length and an
               object are refused without a fault; a same-size resize returns the reference, through
               a stale one it is refused; a read-only mapping is refused (ENOTSUP); then grows (moves)
               and shrinks in place with data and a stored reference; the munmap of the last page
               ends the mapping; ends with a fault through its reference (139)
  exit-victim  a thread exits naming a victim mapping it holds no reference to (exit 0)
  exit-renew   a thread renews its stack's mapping and exits; the main thread's copy of the renewed
               reference must fault with 25 (139)
  create       the frame's hand-over with the reference from before CRENEW (refused) and with the
               renewed one; the renewed reference must fault with 25 afterwards (139)
  create-fail  contexts that wait until released fill the thread table until a create fails: the
               module refuses (ENOSPC) and the frame stays the program's, or behind a launcher with
               a smaller table (--launcher-small-table) the launcher fails the create after the
               module took the frame and releases the frame's mapping (exit 0)
  detached     a detached pthread's stack is unmapped at its exit (exit 0)
  exit-length  a thread exits naming its stack with a wrong length: the exit completes, the stack
               stays mapped and unmaps later through the kept reference (exit 0)
  exit-huge    the same with the length 2^64 - 1 and an exit word outside the stack (exit 0)
  exit-frame   a thread exits naming its own frame as its stack: the module keeps the stack and
               releases the frame once (exit 0)
  protect      mprotect through the mapping's reference, part of it, and refused through an integer
               address, a stale or narrowed reference and a nested object (exit 0)
  protect-held a pthread's mprotect of its own exchange region, held by the launcher: EBUSY; after
               the thread detaches its transport, the same mprotect succeeds (exit 0)
  create-gate  thread create with a metadata page, the launcher's stack and a frame reference
               moved inside the page: refused; then through the frame's own reference (exit 0)
  join-two     two contexts join one; both return after its exit, and a new context follows
               (exit 0)
  join-after   joins after an exit with and without an exit word return 0; an id never issued
               answers ESRCH (exit 0)
  exit-many    40 contexts exit without an exit word and are never joined; one more is created
               (exit 0)
  exit-word    exit words that are unaligned, straddle the released stack, lie in a read-only page
               or in the thread's frame are not written; the exits complete (exit 0)
  attach-readonly a pthread detaches its transport and makes its exchange region read-only; its
               next delegated call attaches again, and the launcher refuses the attachment and
               ends the program (exit 125)
  exit-word-refs exit words through references (stage 4c): a word in another thread's frame, a
               stale and a narrowed reference are refused, and that thread runs on; a word in a
               mapping whose middle page was protected and restored is written and wakes its futex
               waiter (exit 0)
  futex        a futex wait needs a live reference that covers the word, outside every thread's
               frame (EFAULT otherwise); a wake takes an integer address (exit 0)
  advise       madvise(MADV_DONTNEED) and MADV_FREE need a reference that covers the range, as
               mprotect does; an accepted call leaves the range reading zero (exit 0)
  join-burned  a create on a read-only frame fails and issues no id (a join of it answers ESRCH,
               the next create takes it); the main thread's join of itself answers EDEADLK (exit 0)
  attach-exchange a pthread names the launcher's own exchange region as its transport: the
               launcher refuses the attachment and ends the program (exit 125)
  attach-address the same with a mapping named by its plain address (exit 125; skipped without
               root grants, exit 0)
  futex-remap  a waiter of a mapping that is unmapped and mapped again at the same address is woken
               by the unmap, and a wake of the new mapping finds nobody (exit 0)
  die-blocked  the launcher ends the program (a refused attachment) while the main thread blocks in
               a delegated read: exit 125 without a crash
  word-metadata a futex wait and an exit word in the allocator's metadata are refused (exit 0)
  advise       (stage 4d) also zeroes a page made PROT_NONE after it held data
Stage 4e (root-contract.c lists the checks of each):
  futex-readonly  a futex wait on a read-only page compares; with diag_admit the read admits no page
                  (no touch line for it) and an exit word's store does (one touch line)
  futex-park   (the mapping pool on) a waiter on a parked mapping is woken; the revived mapping's
               wake finds nobody
  futex-end    a spurious wake makes the libc's loop wait again; the end of the mapping wakes it and
               its read through the stale reference must fault with 25 (exit 139)
  futex-flags  FUTEX_CLOCK_REALTIME without WAIT_BITSET and an unknown flag: ENOSYS; a huge relative
               timeout waits
  die-stderr   stderr is a pipe nobody reads and full: the launcher still ends with 125 at once
  protect-exec the image refuses mprotect, madvise and munmap; PROT_EXEC is refused everywhere
  partial-hole the middle page of three goes; ends with a page fault (13) in the hole (exit 139)
  partial-all  pages go one by one, repeats change nothing, the last one ends the block; ends with
               the stale read (25, exit 139)
  partial-narrow  narrowed, nested and integer references cannot remove a page (root grants)
  partial-fixed   MAP_FIXED refills a hole with zeros, refused while an object exists below P; with
                  diag_admit the removal reports the tags of the removed page before and after it
                  cleared them, and the refill's admission reports its frame
  partial-mremap  mremap shrinks in place; growth into the removed tail is refused while an object
                  exists below the block (stage 4f, root grants), grows back in place once it ended;
                  ENOMEM beyond the extent, moves
Stage 4f (root-contract.c lists the checks of each):
  mremap-validate  shrink over a hole and growth with a hole in the old range: EFAULT, nothing
                  changed; a cursor below the block: EINVAL
  pages-enomem    vm.max_map_count lowered to the map count: a partial munmap and (root grants)
                  MAP_FIXED answer ENOMEM with nothing changed; the launcher logs each abort
                  (CAPSTONE_VM_PAGES_ABORTED step=protect errno=12)
  pages-tagged    the partial munmap of a page that holds a reference: with root grants and pinned the
                  module's census shows the tags cleared; with --module-hooks (pinned) the page was
                  unheld first (test_unhold_pages) and the check pinned it again (scan line); the
                  kernel's count of tagged frames cleared at their free (captag cleared_pages) grows
                  in the unpinned mode and stays in the pinned one
  futex-stale     (root grants) a wake of one reaches the live waiter, not the stale one at the same
                  address
Hook modes of stage 4f (--launcher-hooks): pages-derive, pages-fault, pages-admit (root grants;
CAPSTONE_VM_TEST_PAGES_DELAY_MS or _LATE_DELAY_MS open a window inside a MAP_FIXED; the module stops
the program in it, "step waits" in the census) and release-fail (CAPSTONE_VM_TEST_RELEASE_FAIL: the
block's munmap answers 0, the range stays out of reuse, the stale reference faults 25). With
--pages-run (needs --module-hooks) the module lets the program run in the window
(test_pages_run): pages-derive's commit then fails closed (the block ends, EINVAL, fault 25 on the
object, refusal "objects derived below the mapping" at "map fixed commit"), and pages-fault's and
pages-admit's faults are deferred ("resolve refused at moving"). pages-admit also checks that no page
admitted before the commit was unpinned by it.
Hook modes of block A (the follow-ups of stage 4f; root grants): pages-closed (--launcher-hooks and
--module-hooks, pinned: the module lets the program run in a MAP_FIXED's window, test_pages_run, and a
context derives an object below the block there, so the commit fails closed; test_close_unheld then
unholds the block's first page, which holds a reference, and fails the retire preparation: the module
must clear that page in place, "clear in place" with the unheld frame in the census, and the kernel's
count of tagged frames cleared at their free must not grow) and pages-discard (--launcher-hooks:
CAPSTONE_VM_TEST_DISCARD_FAIL makes a refill's discard fail after its first page: MAP_FIXED answers
EINVAL, the block has ended, CAPSTONE_VM_PAGES_FAILED step=discard, and the block's reference faults 25).
Modes that need a launcher built with -DCAPSTONE_VM_TEST_HOOKS (--launcher-hooks; the runner sets
their delays, HOOK_ENV, and runs them by default only then):
  futex-race   the futex window: a waiter compared and sleeps while the word's mapping is replaced
               (the unmap wakes it, 0, and it never compares the new mapping), and a wake inside the
               window reaches the waiter (exit 0)
  exit-race    the exit-word window: the word's object is revoked and its memory reused between the
               launcher's checks and the store: the word is refused (not live) and the new object
               keeps its value (exit 0; skipped without root grants)
  join-withdrawn a joiner waits on the entry of a create that fails: ESRCH (exit 0)
  rollback-fail a refused rollback of a failed create ends the program (exit 125)
  eintr-word, eintr-exit (also --module-hooks) a signal interrupts the wait for vm_lock of the
               futex wait's and the thread exit's ioctl; the launcher repeats it (exit 0)
Modes that need a module built with CV_TEST_HOOKS=1 (--module-hooks with --module; the runner sets
its parameters for the mode, MODULE_HOOK_PARAMS):
  advise-refuse a madvise whose third page the module refuses changes nothing (exit 0)
The modes that join every thread also read the launcher's thread census at exit
(CAPSTONE_VM_THREADS count=1: every entry freed once). Every refused exit word and futex wait is
logged by the launcher with its reason (CAPSTONE_VM_EXIT_WORD_REFUSED, CAPSTONE_VM_FUTEX_REFUSED),
and the runner compares them per mode.
--launcher-range-bypass says that the --launcher lacks the check that a page range lies inside one
block (range_mapping): partial-narrow's munmap beyond the block then reaches the module, which must
refuse it ("range is not the mapping").
Every refusal of a request is counted (parameter unmap_refused) and reported in the kernel log
with its reason and request, and so is every CV_RETIRE that the module refuses because the
mapping's reservation is live (parameter retire_refused; no launcher path asks for one any more)
and every arena that a thread exit names but keeps (parameter exit_kept). The runner reads them
back per mode (the log is read and cleared after each mode) and compares them with what each mode
does. Every thread's frame must be released at its exit (no CAPSTONE_VM_FRAME_RELEASE_FAILED).

--root-grants 0 loads the module without the parameter. The checks that hold only with root
grants print ROOT_SKIP with a reason, and the runner requires those skips instead.
--control NAME runs against a module that is one change away from the adapter's (--module):
  cover          the unmap check skips condition 4 (the bounds cover the mapping): the narrowed
                 munmap is accepted, and the narrow mode fails at that check (exit 1)
  parent         the check skips condition 3 (the operand's record is a child of M): the object
                 over the whole page ends the whole mapping, and the refuse mode fails there; the
                 protect mode fails at the object's mprotect, the mremap mode at the object's
                 mremap and the advise mode at the object's madvise
  exit-operand   the stack's arena at thread exit is released without the CREVOKE of M: the
                 renewed stack lifetime survives, and exit-renew reads through it
  protect-check  CV_PROTECT accepts without checking the reference: the protect mode fails at the
                 integer address (exit 1)
  create-address thread create without the comparison of the launcher's frame with a0's cursor:
                 with --launcher-frame-floor the create-gate mode fails at the moved reference
Every other mode must pass as without the control.
--launcher PATH runs another capstone-vexec. --launcher-bypass says that it lacks the launcher's
whole-mapping check: the partial munmap, the mremap with a wrong old length and the stacks that
exit-length and exit-huge name with a wrong length then reach the module, which must refuse each
(reason "range is not the mapping") and complete the exits. --launcher-frame-floor says that it
names the mapping around a0's cursor as the frame at thread create: the moved frame reference of
create-gate then reaches the module, which must refuse it (reason "range is not the mapping").
--launcher-small-table says that the --launcher's thread table has 4 entries, fewer than the
module's limit: create-fail then reaches the launcher's rollback.
--launcher-control NAME says that the --launcher is one change away from the adapter's:
  old-exit-word  the exit word is checked against the stack base and length that the program
                 names, as before stage 4: exit-huge stops the launcher at the exit service
  no-hold        protect_range() lacks the check of the transport hold: protect-held fails at the
                 held mapping's mprotect (exit 1)
  old-join       the thread table of stage 4 (an exit with an exit word frees its entry unless a
                 joiner waits, every other exit keeps it, the first waking joiner frees it):
                 join-two's second joiner never returns, join-after's join after a worded exit
                 answers ESRCH, exit-many's 32nd create fails
  no-writable-check
                 the exit word and the transport are checked against the visible length only, not
                 the writable range: exit-word's read-only word makes the launcher fault (139), and
                 attach-readonly's read makes the launcher fault instead of the refusal
  no-align       the exit word's alignment is not checked: the launcher's store to exit-word's
                 unaligned word traps to the firmware, whose misaligned-access emulation takes a
                 capability fault at the word's address and halts the machine (the gate never
                 ends). Stage 4b; since stage 4c the module checks the alignment.
  exit-word-4b   the exit word is checked as in stage 4b, by its address alone (one mapping,
                 aligned, writable, outside the exiting thread's own frame), without the kernel's
                 check of the reference and of every thread's frame: exit-word-refs' word in another
                 thread's frame is written, and that thread faults when it resumes (139)
  writable-interval
                 the writable state is stage 4b's single interval per mapping, which keeps only the
                 larger part after a split: exit-word-refs' word in the third page of a mapping
                 whose middle page was protected and restored is refused (exit 1)
  futex-4c       the --launcher and --module are stage 4c's (exec.c with the two delays of the hooks
                 build): futex-race's waiter compares the new mapping (EAGAIN) and futex-remap's
                 wake of the new mapping wakes the old mapping's waiter
  exit-word-4c   the same pair: exit-race's store reaches the object reused after the revoke
  withdraw-4c    the hooks launcher with stage 4c's thread_withdraw (the entry of a failed create is
                 freed without a broadcast): join-withdrawn's joiner never returns
--launcher-hooks says that the --launcher (or the adapter's) is a hooks build; the hook modes then run.
--libc-control no-remap-check says that the --application was built with a libc whose mremap
copies before any check (stage 4): the mremap mode faults at the oversized old length (139).
--block-frames loads the module with block_frames=1 (every arena gets a frame set, the program's
mappings included), and partial-fixed must then show that Linux handed the removed page's frame back
at the refill and the module admitted it.
Stage 4e module controls (--control, with --module): frames-insert-only (the frame set is never
deleted from: with --block-frames the refill's admission is refused and partial-fixed faults),
count-repeats (the count decrements for removed pages too: partial-all's live read after the repeated
munmaps faults with 25), no-tag-clear (the removal does not clear tags: partial-fixed's census line
reports tags after the clearing; the program's own read after the refill cannot see it, since Linux
zeroes every page it maps fresh).
Stage 4f controls: --control no-mark (--module a hooks build of the module without the transition
mark, with the hooks launcher): pages-derive's object, derived while the MAP_FIXED waits, has its
page replaced under it (exit 1), pages-admit's frame admitted in the window is unpinned at the commit
while it stays mapped, and pages-fault's store into the page being refilled ends the program with a
page fault; --control no-check-pin (--module a hooks build whose check pins no unheld page): with
test_unhold_pages, pages-tagged's unheld tagged page gets no census line and reaches Linux tagged
(captag cleared_pages grows); --launcher-control futex-count-all (a wake counts stale waiters too):
futex-stale's wake of one wakes the stale waiter and not the live one (exit 1); --libc-control
shrink-unchecked (--application built against a libc whose mremap shrinks without validating the
old range): mremap-validate's shrink over a hole ends the block and answers success (exit 1).
Block A controls: --control no-close-clear (--module a hooks build whose fail-closed commit retires
without clearing the unheld pages in place, with the hooks launcher): pages-closed's unheld tagged page
reaches Linux with its tag (no "clear in place" line, captag cleared_pages grows); --control
child-depth (--module whose ref_child accepts any live record deeper than M): the object over the whole
page or mapping is accepted, so refuse, protect, mremap and advise fail at it, and renew still passes;
--launcher-control discard-abort (a hooks launcher that aborts after a failed discard, as stage 4f did):
pages-discard's MAP_FIXED answers EAGAIN and the block stays live (exit 1).
Block A, F4 part 2 (root-contract.c lists the checks of each): protect-exec also checks that the
startup block stays read-only through every service; top-page (--launcher-hooks:
CAPSTONE_VM_TEST_TOP_PAGE, the runner reads CAPSTONE_VM_TEST_TOP_PAGE guard=1 launcher=12 module=22: the
launcher holds the topmost page below 2^38, grant_mapping refuses it with ENOMEM and the module's add()
with EINVAL); pages-unclear (--launcher-hooks and --module-hooks, test_close_unheld=2: as pages-closed,
and the in-place clearing leaves the unheld page, so the launcher keeps the ended block's extent mapped
PROT_NONE, CAPSTONE_VM_RELEASE_HELD); grow-maps and grow-objects (the table grows beyond its first batch
through the module's grants and through the pressure escape); nogrow-map, nogrow-malloc and
realloc-exhausted, which need --module-args node_max_pages=16 (one batch, so the table cannot grow) and
are not in the default list: mmap answers ENOMEM and malloc NULL without a loop of escapes, both recover
after frees, and realloc at a full table answers NULL with the original intact. The growth modes skip
unless the module is a CAPSTONE_VM_IDS=63 build (--ids 63, the processor's x-capstone-ids=63); there the
refuse mode expects no collection (the 63-bit format reuses slots through generations).
Every run reads CAPSTONE_VM_RELEASES at exit: no failed release, no held extent except in pages-unclear,
and the launcher's guard of the top page held (top_guard=1).
--module-args ARGS appends ARGS to the module's insmod line. --no-unload leaves the module loaded at
the end (the firmware without the capability monitor, which the 63-bit format needs, hangs the unload
after a program ran; module_unloaded is then not checked).
--libc-control old-realloc says that the --application was built against the libc of F4 part 1, whose
realloc moves a mapping through __heap_remapping and ends the old object before it issues the new one,
with the CDERIVE that follows the move given its failure result (errno ENOMEM, an untagged 0): the
first realloc that gets past the move answers NULL with the object lost, and realloc-exhausted faults
25 at its read (exit 139). Without that failure the move's munmap frees the old mapping's slots before
the CDERIVE, so a single-threaded run never reaches the window.

--no-refs runs on a processor without supervisor references. The module must not load. Where the
firmware hands cause 24 to S the load is refused with a message. The gate platform's monitor keeps
the trap and halts the machine at the module's probe, after the module printed why.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import memory_mode

HERE = Path(__file__).resolve().parent
MODES = ('unmap', 'refuse', 'narrow', 'renew', 'renew-ref', 'mremap',
         'exit-victim', 'exit-renew', 'create', 'create-fail', 'detached',
         'exit-length', 'exit-huge', 'exit-frame', 'protect', 'protect-held', 'create-gate',
         'join-two', 'join-after', 'exit-many', 'exit-word', 'attach-readonly',
         'exit-word-refs', 'futex', 'advise', 'join-burned', 'attach-exchange', 'attach-address',
         'futex-remap', 'die-blocked', 'word-metadata',
         'futex-readonly', 'futex-park', 'futex-end', 'futex-flags', 'die-stderr', 'protect-exec',
         'partial-hole', 'partial-all', 'partial-narrow', 'partial-fixed', 'partial-mremap',
         'mremap-validate', 'pages-enomem', 'pages-tagged', 'futex-stale', 'grow-maps', 'grow-objects')
# The modes that need a table that cannot grow (--module-args node_max_pages=16); not in the default list.
NOGROW_MODES = ('nogrow-map', 'nogrow-malloc', 'realloc-exhausted')
# The modes that run only on the 63-bit format (they skip in the older ones).
SUBLET_MODES = {'grow-maps', 'grow-objects'} | set(NOGROW_MODES)
# The modes that need the launcher's test hooks, and the environment that sets their delays.
HOOK_ENV = {
    'futex-race': 'CAPSTONE_VM_TEST_FUTEX_DELAY_MS=2000',
    'exit-race': 'CAPSTONE_VM_TEST_EXIT_DELAY_MS=3000',
    'join-withdrawn': ('CAPSTONE_VM_TEST_JOIN_DELAY_MS=1000 CAPSTONE_VM_TEST_CREATE_FAIL=2 '
                       'CAPSTONE_VM_TEST_CREATE_DELAY_MS=2000'),
    'rollback-fail': 'CAPSTONE_VM_TEST_CREATE_FAIL=1 CAPSTONE_VM_TEST_ROLLBACK_FAIL=1',
    'eintr-word': 'CAPSTONE_VM_TEST_EINTR=word',
    'eintr-exit': 'CAPSTONE_VM_TEST_EINTR=exit',
    'pages-derive': 'CAPSTONE_VM_TEST_PAGES_DELAY_MS=2000',
    'pages-fault': 'CAPSTONE_VM_TEST_PAGES_DELAY_MS=2000',
    'pages-admit': 'CAPSTONE_VM_TEST_PAGES_LATE_DELAY_MS=2000',
    'release-fail': 'CAPSTONE_VM_TEST_RELEASE_FAIL=1',
    'pages-closed': 'CAPSTONE_VM_TEST_PAGES_DELAY_MS=2000',
    'pages-discard': 'CAPSTONE_VM_TEST_DISCARD_FAIL=1',
    'pages-unclear': 'CAPSTONE_VM_TEST_PAGES_DELAY_MS=2000',
    'top-page': 'CAPSTONE_VM_TEST_TOP_PAGE=1',
}
# The hook modes whose window --pages-run opens to the program (the hooks module's test_pages_run).
PAGES_RUN_MODES = ('pages-derive', 'pages-fault', 'pages-admit')
HOOK_MODES = tuple(HOOK_ENV)
# The modes that need a module built with CV_TEST_HOOKS, and the parameter the gate sets for each.
MODULE_HOOK_PARAMS = {
    'eintr-word': ('test_hold_ms', '1000'),
    'eintr-exit': ('test_hold_ms', '1000'),
    'advise-refuse': ('test_advise_refuse', '3'),
    'pages-closed': ('test_close_unheld', '1'),
    'pages-unclear': ('test_close_unheld', '2'),
}
MODULE_HOOK_MODES = tuple(MODULE_HOOK_PARAMS)
# The environment of a mode that needs one without a hooks build.
MODE_ENV = {'futex-park': 'CAPSTONE_VM_MAPCACHE=1048576'}
# The modes that run with diag_admit on; the runner reads their admission and removal lines.
DIAG_MODES = {'futex-readonly', 'partial-fixed', 'pages-tagged', 'pages-derive', 'pages-fault',
              'pages-admit', 'pages-closed', 'pages-unclear'}
# Per mode: the checks, those that need root grants (skipped otherwise), and the expected exit.
CHECKS = {
    'unmap': ('grant_reads_and_writes', 'libc_accepts_copyable_grant',
              'unmap_through_mapping_reference', 'other_mapping_survives_unmap',
              'second_unmap_refused', 'stale_reference_at_reused_address_refused'),
    'refuse': ('address_refused', 'nested_object_refused', 'unmap_beyond_block_refused',
               'stale_reference_refused', 'stale_reference_after_collection_refused'),
    'narrow': ('narrowed_reference_refused', 'whole_reference_after_narrowed'),
    'renew': ('renewed_old_reference_refused', 'renewed_mapping_unmaps'),
    'renew-ref': ('renewed_old_reference_refused', 'renewed_mapping_unmaps'),
    'mremap': ('mremap_beyond_block_refused', 'mremap_oversized_old_refused',
               'mremap_nested_object_refused', 'mremap_same_size_returns_reference',
               'mremap_same_size_stale_refused', 'mremap_read_only_refused', 'public_mremap_grows',
               'mremap_old_reference_refused', 'public_mremap_shrinks'),
    'exit-victim': ('exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
                    'victim_unmaps_afterwards'),
    'exit-renew': ('stack_renewed_before_exit',),
    'create': ('create_with_stale_frame_refused', 'create_with_renewed_frame'),
    'create-fail': ('module_limit_keeps_frame', 'failed_create_id_refused'),
    'detached': ('detached_exit_unmaps_stack',),
    'exit-length': ('exit_with_wrong_length_completes', 'wrong_length_stack_stays_mapped',
                    'wrong_length_stack_unmaps_afterwards'),
    'exit-huge': ('exit_with_huge_length_completes', 'huge_length_stack_stays_mapped',
                  'huge_length_stack_unmaps_afterwards'),
    'exit-frame': ('exit_with_frame_as_stack_completes', 'frame_as_stack_released_once'),
    'protect': ('protect_through_mapping_reference', 'protect_integer_address_refused',
                'protect_stale_reference_refused', 'protect_narrowed_reference_refused',
                'protect_nested_object_refused', 'protect_narrowed_covering_range'),
    'protect-held': ('protect_held_mapping_refused', 'protect_after_detach'),
    'create-gate': ('create_metadata_frame_refused', 'create_launcher_page_refused',
                    'create_mismatched_frame_refused', 'create_after_refusals'),
    'join-two': ('two_joiners_return', 'thread_after_two_joiners'),
    'join-after': ('join_after_word_exit', 'join_after_raw_exit', 'join_unknown_id_refused'),
    'exit-many': ('exit_unjoined_many_then_create',),
    'exit-word': ('unaligned_exit_word_not_written', 'straddling_exit_word_not_written',
                  'read_only_exit_word_not_written', 'writable_exit_word_written',
                  'stack_exit_word_dropped', 'frame_exit_word_not_written'),
    'attach-readonly': ('attach_protect_before_attach',),
    'exit-word-refs': ('other_frame_word_not_written', 'middle_protected_word_written',
                       'stale_reference_word_not_written', 'narrow_reference_word_not_written'),
    'futex': ('futex_wait_through_reference', 'futex_wait_integer_refused', 'futex_wait_stale_refused',
              'futex_wait_narrow_refused', 'futex_wait_frame_refused', 'futex_wake_integer_answers'),
    'advise': ('advise_integer_refused', 'advise_stale_refused', 'advise_narrowed_refused',
               'advise_nested_object_refused', 'advise_zeroes_covered_page', 'advise_free_zeroes',
               'advise_prot_none_zeroes'),
    'join-burned': ('join_failed_create_refused', 'failed_create_consumes_no_id',
                    'main_self_join_deadlock'),
    'attach-exchange': ('launcher_exchange_reference',),
    'attach-address': ('attach_address_refused',),
    'futex-remap': ('futex_wake_reaches_only_its_mapping',),
    'die-blocked': ('die_reader_blocks',),
    'word-metadata': ('metadata_futex_wait_refused', 'metadata_exit_word_not_written'),
    'futex-race': ('futex_wait_never_reads_new_mapping', 'futex_wake_in_window_caught'),
    'exit-race': ('exit_word_after_revoke_not_written',),
    'join-withdrawn': ('joiner_on_withdrawn_id_returns',),
    'rollback-fail': ('rollback_create_started',),
    'futex-readonly': ('futex_wait_read_only_compares', 'futex_wait_protected_compares'),
    'futex-park': ('futex_waiter_woken_at_park',),
    'futex-end': ('futex_spurious_wake_rewaits',),
    'futex-flags': ('futex_clock_realtime_refused', 'futex_unknown_flag_refused',
                    'futex_huge_timeout_waits'),
    'eintr-word': ('eintr_futex_wait_retries',),
    'eintr-exit': ('eintr_thread_exit_retries',),
    'advise-refuse': ('advise_refused_changes_nothing',),
    'die-stderr': ('die_stderr_blocks',),
    'protect-exec': ('image_protect_refused', 'image_unmap_refused', 'map_exec_refused',
                     'protect_exec_refused', 'fixed_exec_refused', 'startup_stays_read_only'),
    'partial-hole': ('partial_unmap_middle', 'partial_unmap_wakes_waiter', 'outer_pages_work',
                     'no_mapping_in_hole', 'hole_unmap_repeats', 'hole_requests_refused'),
    'partial-all': ('hole_unmap_idempotent', 'prot_none_page_counts', 'count_exact_with_mprotect',
                    'last_page_ends_block'),
    'partial-narrow': ('partial_narrowed_refused', 'partial_nested_object_refused',
                       'partial_address_refused', 'partial_beyond_block_refused',
                       'partial_through_moved_reference', 'whole_unmap_with_hole'),
    'partial-fixed': ('fixed_refill_reads_zero', 'removed_page_tags_cleared',
                      'fixed_replaces_mapped_page', 'fixed_refused_with_object',
                      'fixed_after_object_released'),
    'partial-mremap': ('mremap_shrink_in_place', 'mremap_grow_refused_with_object', 'mremap_grow_into_tail',
                       'mremap_grow_beyond_enomem', 'mremap_grow_beyond_moves'),
    'mremap-validate': ('mremap_shrink_over_hole_refused', 'mremap_grow_over_hole_refused',
                        'mremap_cursor_outside_refused'),
    'pages-enomem': ('partial_unmap_enomem_changes_nothing', 'fixed_enomem_changes_nothing',
                     'partial_unmap_after_limit'),
    'pages-tagged': ('tagged_page_removed',),
    'futex-stale': ('futex_stale_waiter_not_counted',),
    'pages-derive': ('pages_derive_after_refill',),
    'pages-fault': ('pages_fault_waits_then_proceeds',),
    'pages-admit': ('pages_admit_store_after_refill',),
    'release-fail': ('release_failure_keeps_range_unused',),
    'pages-closed': ('pages_closed_fails_closed',),
    'pages-discard': ('pages_discard_fails_closed',),
    'pages-unclear': ('pages_unclear_extent_held',),
    'top-page': ('top_page_mapping_below',),
    'grow-maps': ('grow_maps_through_grants',),
    'grow-objects': ('grow_objects_through_escape',),
    'nogrow-map': ('nogrow_map_enomem', 'nogrow_map_recovers'),
    'nogrow-malloc': ('nogrow_malloc_null', 'nogrow_malloc_recovers'),
    'realloc-exhausted': ('realloc_exhausted_keeps_object',),
}
# pages-derive with --pages-run: the commit fails closed instead.
CHECKS_PAGES_RUN = {'pages-derive': ('pages_derive_fails_closed',)}
# create-fail behind a launcher whose table is smaller than the module's limit.
CHECKS_SMALL_TABLE = {'create-fail': ('failed_create_releases_frame', 'failed_create_id_refused')}
ROOT_ONLY = {'stale_reference_at_reused_address_refused', 'address_refused',
             'nested_object_refused', 'stale_reference_refused',
             'stale_reference_after_collection_refused', 'narrowed_reference_refused',
             'whole_reference_after_narrowed', 'renewed_old_reference_refused',
             'exit_with_own_stack_at_victim_refused', 'exit_with_victim_address_refused',
             'victim_unmaps_afterwards', 'create_with_stale_frame_refused',
             'protect_integer_address_refused', 'protect_stale_reference_refused',
             'protect_narrowed_reference_refused', 'protect_nested_object_refused',
             'mremap_nested_object_refused', 'mremap_same_size_stale_refused',
             'stale_reference_word_not_written', 'narrow_reference_word_not_written',
             'futex_wait_integer_refused', 'futex_wait_stale_refused', 'futex_wait_narrow_refused',
             'advise_integer_refused', 'advise_stale_refused', 'advise_narrowed_refused',
             'advise_nested_object_refused', 'attach_address_refused',
             'exit_word_after_revoke_not_written',
             'partial_narrowed_refused', 'partial_nested_object_refused', 'partial_address_refused',
             'fixed_refill_reads_zero', 'removed_page_tags_cleared', 'fixed_replaces_mapped_page',
             'fixed_refused_with_object', 'fixed_after_object_released',
             'mremap_grow_refused_with_object', 'mremap_grow_into_tail', 'fixed_enomem_changes_nothing',
             'futex_stale_waiter_not_counted', 'pages_derive_after_refill', 'pages_derive_fails_closed',
             'pages_fault_waits_then_proceeds', 'pages_admit_store_after_refill',
             'pages_closed_fails_closed', 'pages_discard_fails_closed', 'pages_unclear_extent_held'}
FAULTING = {'unmap', 'renew', 'renew-ref', 'mremap', 'exit-renew', 'create', 'partial-all', 'futex-end',
            'release-fail', 'pages-discard'}
# Modes whose stale read does not map the address again first (the frame is not reused, or the
# fault comes from a waiter's own read).
NO_REMAP = {'create', 'futex-end', 'release-fail', 'pages-discard'}
# Faulting modes that end with the stale read only with root grants (without, they skip and exit 0).
ROOT_FAULTING = {'pages-discard'}
# Modes that end with a page fault of a live reference (stage 4e: a removed page), by cause.
PAGE_FAULTING = {'partial-hole': 13}
# The kernel's refusals per mode with root grants, as (reason, request) in order
# (capstone_vm.c cv_unmap_reasons, request_refused). The other refusals (a second munmap, the
# second page of a partial munmap, the old reference after mremap, the wrong old length of mremap)
# are the launcher's, which finds no whole mapping at that address or of that length. exit-victim's
# first exit names the victim with the own stack's reference moved there, the stack's end: a live reference to another
# reservation (its reason is filled in from the operand's type). A reason may be a tuple of
# alternatives: the collection in the refuse mode may clear the tag of the program's own stale
# reference before the munmap hands it over.
REFUSALS = {
    'unmap': [('not live', 'munmap')],
    'refuse': [('not a reference', 'munmap'), ('not the mapping', 'munmap'),
               ('not live', 'munmap'), ('not live', 'munmap'),
               (('not live', 'not a reference'), 'munmap')],
    'narrow': [('does not cover the mapping', 'munmap')],
    'renew': [('not live', 'munmap')],
    'renew-ref': [('not live', 'munmap')],
    'mremap': [('not the mapping', 'mremap'), ('not live', 'mremap')],
    'exit-victim': [(None, 'thread exit'), ('not a reference', 'thread exit')],
    'exit-renew': [],
    'create': [('not live', 'thread create')],
    'create-fail': [],
    'detached': [],
    'exit-length': [],
    'exit-huge': [],
    'exit-frame': [],
    'protect': [('not a reference', 'mprotect'), ('not live', 'mprotect'),
                ('does not cover the mapping', 'mprotect'), ('not the mapping', 'mprotect')],
    'protect-held': [],
    'create-gate': [],
    'join-two': [],
    'join-after': [],
    'exit-many': [],
    'exit-word': [],
    'attach-readonly': [],
    'exit-word-refs': [],
    'futex': [],
    'advise': [('not a reference', 'madvise'), ('not live', 'madvise'),
               ('does not cover the mapping', 'madvise'), ('not the mapping', 'madvise')],
    'join-burned': [],
    'attach-exchange': [],
    'attach-address': [('not a reference', 'attach')],
    'futex-remap': [],
    'die-blocked': [],
    'word-metadata': [],
    'futex-race': [],
    'exit-race': [],
    'join-withdrawn': [],
    'rollback-fail': [],
    'futex-readonly': [],
    'futex-park': [],
    'futex-end': [],
    'futex-flags': [],
    'eintr-word': [],
    'eintr-exit': [],
    'advise-refuse': [],
    'die-stderr': [],
    'protect-exec': [],
    'partial-hole': [],
    'partial-all': [],
    'partial-narrow': [('does not cover the mapping', 'munmap'), ('not the mapping', 'munmap'),
                       ('not a reference', 'munmap')],
    'partial-fixed': [('objects derived below the mapping', 'map fixed')],
    # Block A: the check of an mremap growth answers ENOMEM while an object exists below P and logs no
    # refusal (the libc then moves the range, or answers ENOMEM without MREMAP_MAYMOVE).
    'partial-mremap': [],
    'mremap-validate': [],
    'pages-enomem': [],
    'pages-tagged': [],
    'futex-stale': [],
    'pages-derive': [],
    'pages-fault': [],
    'pages-admit': [],
    'release-fail': [],
    'pages-closed': [('objects derived below the mapping', 'map fixed commit')],
    'pages-discard': [],
    'pages-unclear': [('objects derived below the mapping', 'map fixed commit')],
    'top-page': [],
    'grow-maps': [],
    'grow-objects': [],
    'nogrow-map': [],
    'nogrow-malloc': [],
    'realloc-exhausted': [],
}
REFUSALS_PAGES_RUN = {'pages-derive': [('objects derived below the mapping', 'map fixed commit')]}
# What each control changes: the mode it must break, and how.
CONTROL_CUTS = {
    'cover': {'narrow': ('ROOT_FAIL:munmap through a narrowed reference', 'narrowed_reference_refused')},
    'parent': {'refuse': ('ROOT_FAIL:munmap through an object over the whole page', 'nested_object_refused'),
               'protect': ('ROOT_FAIL:mprotect through an object over the whole page',
                           'protect_nested_object_refused'),
               'mremap': ('ROOT_FAIL:mremap through an object over the whole mapping',
                          'mremap_nested_object_refused'),
               'advise': ('ROOT_FAIL:madvise through an object over the whole mapping',
                          'advise_nested_object_refused')},
    'exit-operand': {'exit-renew': ('ROOT_STALE_READ', None)},
    'protect-check': {'protect': ('ROOT_FAIL:mprotect with an integer address',
                                  'protect_integer_address_refused')},
    'create-address': {'create-gate': ('ROOT_FAIL:thread create with a frame reference whose cursor '
                                       'is not the frame\'s address', 'create_mismatched_frame_refused')},
    'frames-insert-only': {'partial-fixed': ('ROOT_PARTIAL_FIXED base=', 'fixed_refill_reads_zero',
                                             'domain-fault')},
    'count-repeats': {'partial-all': ('ROOT_PARTIAL repeats_done', 'hole_unmap_idempotent',
                                      'domain-fault'),
                      'partial-hole': ('ROOT_CHECK:no_mapping_in_hole:PASS', 'hole_unmap_repeats',
                                       'domain-fault')},
    'no-tag-clear': {},
    # Stage 4f: the module without the transition mark (no wait in CV_STEP, no deferred admission, no
    # second check of the refill rule at the commit): pages-derive's object, derived while the
    # MAP_FIXED waits, has its page replaced under it (the program fails at that check, exit 1).
    # pages-admit's program passes, and the runner reads the frame admitted in the window and
    # unpinned at the commit (control_frame_unpinned_while_mapped). pages-fault's store into the page
    # being refilled is refused as a removed page, and the program ends with a page fault.
    'no-mark': {'pages-derive': ('ROOT_PAGES_DERIVE value_now=', 'pages_derive_after_refill'),
                'pages-fault': ('ROOT_PAGES_FAULT base=', 'pages_fault_waits_then_proceeds', 'domain-fault')},
    # The check of a page operation does not pin the resident pages it does not hold (stage 4e): with
    # the hooks module's test_unhold_pages, pages-tagged's unheld tagged page gets no census line and
    # reaches Linux with its tag (the kernel's count of tagged frames cleared at their free grows).
    'no-check-pin': {},
    # Block A: the fail-closed commit retires without clearing the unheld pages in place when its retire
    # preparation fails (stage 4f's pages_finish): pages-closed's unheld tagged page reaches Linux with
    # its tag (no "clear in place" line, the kernel's count grows).
    'no-close-clear': {},
    # Block A: ref_child by depth alone (a live record deeper than M passes condition 3, without the step
    # from M to P in the preorder list): the object over the whole page or mapping, a grandchild of M,
    # is accepted, so the refuse mode fails at it, and protect, mremap and advise as for 'parent'.
    'child-depth': {'refuse': ('ROOT_FAIL:munmap through an object over the whole page', 'nested_object_refused'),
                    'protect': ('ROOT_FAIL:mprotect through an object over the whole page',
                                'protect_nested_object_refused'),
                    'mremap': ('ROOT_FAIL:mremap through an object over the whole mapping',
                               'mremap_nested_object_refused'),
                    'advise': ('ROOT_FAIL:madvise through an object over the whole mapping',
                               'advise_nested_object_refused')},
}
# Launcher controls (--launcher-control): the mode each must break, and how. A third element says
# how the mode ends: 'launcher-fault' (exit 139 without a domain fault: the launcher itself faulted),
# 'domain-fault' (exit 139 with a domain fault) or 'machine-halt' (the firmware halts the machine
# at the address the mode printed, and the gate never ends); without it nothing faults.
LAUNCHER_CUTS = {
    'old-exit-word': {'exit-huge': ('capstone-vexec: service:', 'exit_with_huge_length_completes')},
    'no-hold': {'protect-held': ('ROOT_FAIL:mprotect of a held mapping', 'protect_held_mapping_refused')},
    'old-join': {'join-two': ('ROOT_FAIL:both joiners return', 'two_joiners_return'),
                 'join-after': ('ROOT_FAIL:a join after an exit with an exit word',
                                'join_after_word_exit'),
                 'exit-many': ('ROOT_FAIL:create a context while earlier ones exited unjoined',
                               'exit_unjoined_many_then_create')},
    'no-writable-check': {'exit-word': ('ROOT_EXIT_WORD case=read-only',
                                        'read_only_exit_word_not_written', 'launcher-fault')},
    'no-align': {'exit-word': ('[CAPSTONE] domain halted by capability fault',
                               'unaligned_exit_word_not_written', 'machine-halt')},
    'exit-word-4b': {'exit-word-refs': ('ROOT_EXIT_WORD case=other-frame', 'other_frame_word_not_written',
                                        'domain-fault')},
    'writable-interval': {'exit-word-refs': ('ROOT_FAIL:the launcher writes the word in the third page',
                                             'middle_protected_word_written')},
    'futex-4c': {'futex-race': ('ROOT_FUTEX_RACE remap rc=-11', 'futex_wait_never_reads_new_mapping'),
                 'futex-remap': ('ROOT_FUTEX_REMAP same_address=1 woken=1 rc=0',
                                 'futex_wake_reaches_only_its_mapping')},
    'exit-word-4c': {'exit-race': ('ROOT_EXIT_RACE value=0x0', 'exit_word_after_revoke_not_written')},
    'withdraw-4c': {'join-withdrawn': ('ROOT_FAIL:the joiner of the failed create\'s id returns',
                                       'joiner_on_withdrawn_id_returns')},
    # Stage 4f: a wake counts every waiter, stale or live (stage 4e): futex-stale's wake of one wakes
    # the stale waiter, and the live one is not woken.
    'futex-count-all': {'futex-stale': ('ROOT_FUTEX_STALE woken=1 live_done=0',
                                        'futex_stale_waiter_not_counted')},
    # Block A: a refill whose discard fails is aborted as if nothing had changed (stage 4f's pages_abort):
    # pages-discard's MAP_FIXED answers the discard's error and the block stays, its first refilled page
    # already zero (the program fails at that check, exit 1).
    'discard-abort': {'pages-discard': ('ROOT_PAGES_DISCARD fixed=0 errno=11 registered=1',
                                        'pages_discard_fails_closed')},
}
# Libc controls (--libc-control): the --application was built against a libc one change away.
LIBC_CUTS = {
    'no-remap-check': {'mremap': ('ROOT_MREMAP case=oversized', 'mremap_oversized_old_refused',
                                  'domain-fault')},
    # Stage 4f: a libc whose mremap shrinks through a partial munmap of the tail without validating
    # the old range first (stage 4e): the shrink over a hole removes the last mapped page, ends the
    # block and answers success.
    'shrink-unchecked': {'mremap-validate': ('ROOT_MREMAP_VALIDATE shrink_over_hole failed=0',
                                             'mremap_shrink_over_hole_refused')},
    # Block A, F4 part 2: the libc of F4 part 1, whose realloc moves a mapping before it issues the new
    # object (__heap_remapping, then heap_alloc_reissue), the CDERIVE after the move failing: the NULL
    # after frees lost the object, and the read faults 25.
    'old-realloc': {'realloc-exhausted': ('ROOT_REALLOC_NULL_AFTER_FREES', 'realloc_exhausted_keeps_object',
                                          'domain-fault')},
}
# The attach modes end when the launcher refuses the attachment: it exits 125 after it logged the
# refusal with the reason here, and the worker's read (attach-readonly) or its attachment call never
# completes (the marker). Behind the no-writable-check launcher attach-readonly's launcher faults at
# the read's copy into the read-only exchange region instead (139). attach-address refuses only with
# root grants; without, it skips and exits 0.
ATTACH_REFUSED = 'CAPSTONE_VM_TRANSPORT_REFUSED '
ATTACH_MODES = {'attach-readonly': ('not writable', 'ROOT_ATTACH_READ'),
                'attach-exchange': ('not a whole mapping of the program', 'ROOT_ATTACH_RETURNED'),
                'attach-address': ('reference refused', 'ROOT_ATTACH_RETURNED'),
                'die-blocked': ('not a whole mapping of the program', 'ROOT_DIE_READ')}
# Modes that the launcher ends with 125 at another failure: the line it prints, and the marker the
# program prints only if it returns from the service that fails.
ENDING_MODES = {'rollback-fail': ('capstone-vexec: thread rollback:', 'ROOT_ROLLBACK_RETURNED')}
# The exit words and futex waits the launcher refuses per mode, as the reasons it logs, in order,
# with and without root grants. A reason may be a tuple of alternatives: a stale frame reference
# moved by SCC may lose its tag, which the kernel then reports as no reference.
STALE = ('not live', 'not a reference')
EXIT_WORD_REFUSALS = {
    1: {'exit-word': ['unaligned', 'not inside one mapping', 'not writable', STALE],
        'exit-word-refs': [STALE, 'not live', 'does not cover the word'],
        'exit-race': [STALE], 'word-metadata': ["in the allocator's metadata"]},
    0: {'exit-word': ['unaligned', 'not inside one mapping', 'not writable', "in a thread's frame"],
        'exit-word-refs': ["in a thread's frame"], 'word-metadata': ["in the allocator's metadata"]},
}
FUTEX_REFUSALS = {
    1: {'futex': ['not a reference', 'not live', 'does not cover the word', STALE],
        'word-metadata': ["in the allocator's metadata"], 'partial-hole': ['not inside one mapping']},
    0: {'futex': ["in a thread's frame"], 'word-metadata': ["in the allocator's metadata"],
        'partial-hole': ['not inside one mapping']},
}
EXIT_WORD_REPORT = re.compile(r'^CAPSTONE_VM_EXIT_WORD_REFUSED tid=\d+ word=0x[0-9a-f]+ reason=(.*)$', re.M)
FUTEX_REPORT = re.compile(r'^CAPSTONE_VM_FUTEX_REFUSED tid=\d+ word=0x[0-9a-f]+ reason=(.*)$', re.M)


def reasons_match(seen, expected):
    """seen matches expected entry by entry; an expected entry may be a tuple of alternatives."""
    return len(seen) == len(expected) and all(
        r in (w if isinstance(w, tuple) else (w,)) for r, w in zip(seen, expected))


# Modes whose threads have all exited and been joined at the program's exit: the launcher's census
# must count only the main thread.
CENSUS_MODES = {'create-fail', 'join-two', 'join-after', 'exit-many', 'exit-word', 'exit-word-refs',
                'futex', 'join-burned', 'futex-race', 'futex-remap', 'exit-race', 'join-withdrawn',
                'word-metadata', 'futex-park', 'futex-flags', 'eintr-exit', 'futex-stale', 'pages-derive',
                'pages-fault', 'pages-admit'}
# The arenas that a thread exit names and the module keeps, per mode, as the reasons it reports:
# the two stacks that exit-victim's threads named (root grants only), and the frame that
# exit-frame's thread named as its stack. exit-length's and exit-huge's stacks reach the module
# only behind a launcher without its length check (--launcher-bypass); otherwise the launcher
# reports them (CAPSTONE_VM_STACK_UNMAP_REFUSED) and the module not.
EXIT_KEPT = {'exit-victim': ['request refused'] * 2, 'exit-frame': ["is the thread's frame"]}
EXIT_KEPT_RG0 = {'exit-frame': ["is the thread's frame"]}
EXIT_KEPT_BYPASS = {'exit-length': ['request refused'], 'exit-huge': ['request refused']}
# The launcher's reports of a stack it leaves mapped at a thread exit, per mode.
STACKS_KEPT = {'exit-victim': 2, 'exit-length': 1, 'exit-huge': 1, 'exit-frame': 1, 'exit-word': 0,
               'exit-word-refs': 0}
REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):unmap refused \(([^)]*)\) at ([a-z ]+):', re.M)
ABORT_REPORT = re.compile(r'^CAPSTONE_VM_PAGES_ABORTED op=\d+ base=0x[0-9a-f]+ pages=\d+ step=(\w+) errno=(\d+)$',
                          re.M)
RETIRE_REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):retire refused:', re.M)
KEPT_REPORT = re.compile(r'^ROOT_KERNEL:([\w-]+):thread exit keeps the (stack|frame) \(([^)]*)\)', re.M)


def refusals_match(seen, expected):
    """seen matches expected entry by entry; an expected reason may be a tuple of alternatives."""
    if len(seen) != len(expected):
        return False
    for (reason, request), (want, want_request) in zip(seen, expected):
        wants = want if isinstance(want, tuple) else (want,)
        if reason not in wants or request != want_request:
            return False
    return True


def expected_refusals(mode, control, bypass, frame_floor, victim_reason, range_bypass=False,
                       pages_run=False):
    """The (reason, request) list a mode's kernel log must show, or None where a control cuts the
    mode short (then only a prefix is checked)."""
    refusals = list(REFUSALS_PAGES_RUN[mode] if pages_run and mode in REFUSALS_PAGES_RUN else REFUSALS[mode])
    if frame_floor and mode == 'create-gate' and control != 'create-address':
        # The frame reference moved inside the page reaches the module, which compares it.
        refusals.append(('range is not the mapping', 'thread create'))
    if control == 'protect-check' and mode == 'protect':
        return []
    if mode == 'exit-victim':
        refusals[0] = (victim_reason, 'thread exit')
    if range_bypass and mode == 'partial-narrow':
        # The munmap beyond the block reaches the module (stage 4e).
        refusals.insert(3, ('range is not the mapping', 'munmap'))
    if bypass and mode in ('exit-length', 'exit-huge'):
        # The stack named with a wrong length.
        refusals.append(('range is not the mapping', 'thread exit'))
    if control in ('parent', 'child-depth') and mode == 'refuse':
        return refusals[:1]                 # the address; the nested object is accepted
    if control in ('parent', 'child-depth') and mode == 'protect':
        return refusals[:3]                 # the nested object is accepted
    if control in ('parent', 'child-depth') and mode == 'mremap':
        return []                           # the nested object is accepted
    if control in ('parent', 'child-depth') and mode == 'advise':
        return refusals[:3]                 # the nested object is accepted
    if control == 'parent' and mode == 'exit-victim' and victim_reason == 'not the mapping':
        refusals[0] = ('does not cover the mapping', 'thread exit')
    if control == 'cover' and mode == 'narrow':
        return []
    if control == 'frames-insert-only' and mode == 'partial-fixed':
        return []                           # the refill faults before the refused MAP_FIXED
    return refusals


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ('adapter', 'application', 'qemu', 'images', 'work'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--module', type=Path, help='module to load instead of the adapter\'s (a control)')
    p.add_argument('--launcher', type=Path, help='capstone-vexec to run instead of the adapter\'s')
    p.add_argument('--launcher-bypass', action='store_true',
                   help='the launcher lacks its whole-mapping check; the module must refuse a partial munmap')
    p.add_argument('--launcher-frame-floor', action='store_true',
                   help='the launcher names the mapping around a0\'s cursor as the frame; the module must compare')
    p.add_argument('--launcher-small-table', action='store_true',
                   help='the launcher\'s thread table has 4 entries; create-fail reaches its rollback')
    p.add_argument('--launcher-hooks', action='store_true',
                   help='the launcher is a hooks build (-DCAPSTONE_VM_TEST_HOOKS); the hook modes run')
    p.add_argument('--launcher-range-bypass', action='store_true',
                   help='the launcher lacks the check that a page range lies inside one block; the '
                        'module must refuse partial-narrow\'s munmap beyond the block')
    p.add_argument('--module-hooks', action='store_true',
                   help='the --module is a CV_TEST_HOOKS build; the module hook modes run')
    p.add_argument('--block-frames', action='store_true',
                   help='load the module with block_frames=1; partial-fixed must show the frame reused')
    p.add_argument('--pages-run', action='store_true',
                   help='the hooks module lets the program run inside a page operation (test_pages_run) '
                        'in the modes ' + ', '.join(PAGES_RUN_MODES))
    p.add_argument('--root-grants', type=int, choices=(0, 1), default=1)
    p.add_argument('--control', choices=sorted(CONTROL_CUTS), help='the module is this control (see above)')
    p.add_argument('--launcher-control', choices=sorted(LAUNCHER_CUTS),
                   help='the --launcher is this control (see above)')
    p.add_argument('--libc-control', choices=sorted(LIBC_CUTS),
                   help='the --application was built against this control libc (see above)')
    p.add_argument('--modes', help='comma-separated subset of the modes (default: every mode, the hook '
                                   'modes only with --launcher-hooks)')
    p.add_argument('--no-refs', action='store_true', help='the processor has no supervisor references: insmod must fail')
    p.add_argument('--unpinned', action='store_true', help='load the module with unpinned=1')
    p.add_argument('--module-args', default='', help='more insmod arguments (node_max_pages=16 for '
                                                       + ', '.join(NOGROW_MODES) + ')')
    p.add_argument('--no-unload', action='store_true', help='skip the final rmmod (see above)')
    p.add_argument('--ids', type=int, choices=(30, 63), default=30,
                   help='the module\'s CAPSTONE_VM_IDS build and the processor\'s x-capstone-ids')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    if a.modes is None:
        hooked = [m for m in HOOK_MODES + ('advise-refuse',)
                  if (m not in HOOK_MODES or a.launcher_hooks) and
                     (m not in MODULE_HOOK_MODES or a.module_hooks)]
        a.modes = ','.join(MODES + tuple(hooked))
    modes = [m for m in a.modes.split(',') if m]
    if any(m not in MODES + HOOK_MODES + MODULE_HOOK_MODES + NOGROW_MODES for m in modes):
        p.error(f'unknown mode in {a.modes}')
    if any(m in NOGROW_MODES for m in modes) and a.ids == 63 and 'node_max_pages=' not in a.module_args:
        p.error('the modes ' + ', '.join(NOGROW_MODES) + ' need --module-args node_max_pages=16')
    if any(m in HOOK_MODES for m in modes) and not a.launcher_hooks:
        p.error('the hook modes need --launcher-hooks')
    if any(m in MODULE_HOOK_MODES for m in modes) and not a.module_hooks:
        p.error('the module hook modes need --module-hooks')
    if a.module_hooks and not a.module:
        p.error('--module-hooks needs --module')
    if a.pages_run and not a.module_hooks:
        p.error('--pages-run needs --module-hooks')
    if a.control and not a.module:
        p.error('--control needs --module')
    if a.launcher_control and not a.launcher:
        p.error('--launcher-control needs --launcher')
    if a.control == 'create-address' and not a.launcher_frame_floor:
        p.error('--control create-address needs a --launcher built --launcher-frame-floor')

    a.work.mkdir(parents=True, exist_ok=False)
    stage = a.work / 'stage'
    stage.mkdir()
    shutil.copy2(a.launcher or a.adapter / 'capstone-vexec', stage / 'capstone-vexec')
    shutil.copy2(a.module or a.adapter / 'module/capstone_vm.ko', stage / 'capstone_vm.ko')
    shutil.copy2(a.application, stage / 'root.dom')
    module_args = memory_mode.args(a.unpinned).replace(' root_grants=1', '')
    if a.root_grants:
        module_args += ' root_grants=1'
    if a.block_frames:
        module_args += ' block_frames=1'
    if a.module_args:
        module_args += ' ' + a.module_args
    parameters = '/sys/module/capstone_vm/parameters'

    def run_mode(mode):
        env = ' '.join(e for e in (HOOK_ENV.get(mode, '') if a.launcher_hooks else '',
                                   MODE_ENV.get(mode, '')) if e)
        before, after = [], []
        if mode in MODULE_HOOK_PARAMS:
            name, value = MODULE_HOOK_PARAMS[mode]
            before.append(f'echo {value} > {parameters}/{name}')
            after.append(f'echo 0 > {parameters}/{name}')
        if mode in DIAG_MODES:
            before.append(f'echo 1 > {parameters}/diag_admit')
            after.append(f'echo 0 > {parameters}/diag_admit')
        if (a.pages_run and mode in PAGES_RUN_MODES) or mode in ('pages-closed', 'pages-unclear'):
            before.append(f'echo 1 > {parameters}/test_pages_run')
            after.append(f'echo 0 > {parameters}/test_pages_run')
        if a.module_hooks and not a.unpinned and mode == 'pages-tagged':
            before.append(f'echo 1 > {parameters}/test_unhold_pages')
            after.append(f'echo 0 > {parameters}/test_unhold_pages')
        if mode == 'die-stderr':
            # stderr goes into a FIFO whose reader holds it open and never reads, so the pipe fills;
            # the launcher must still end with 125 at once (its line is dropped).
            command = f'''rm -f /tmp/die.fifo; mkfifo /tmp/die.fifo
( exec 3</tmp/die.fifo; sleep 40 ) &
die_reader=$!
die_start=$(date +%s)
{env} CAPSTONE_VM_STATS=1 ./capstone-vexec root.dom {mode} 2>/tmp/die.fifo
echo ROOT_EXIT:{mode}:$?
echo ROOT_DIE_SECONDS:$(( $(date +%s) - die_start ))
kill $die_reader 2>/dev/null'''
        else:
            command = f'''{env} CAPSTONE_VM_STATS=1 ./capstone-vexec root.dom {mode}
echo ROOT_EXIT:{mode}:$?'''
        lines = ['echo ROOT_MODE_BEGIN:' + mode] + before + [command] + after
        lines.append(f"dmesg -c | grep 'capstone_vm: \\(unmap refused\\|retire refused\\|thread exit keeps"
                     f"\\|resolve refused at\\|touch arena\\|unpinned touch arena\\|remove arena"
                     f"\\|admit arena\\|step waits\\|scan arena\\|test unhold\\|clear in place"
                     f"\\|page operation closed\\)'"
                     f" | sed 's/^.*capstone_vm: /ROOT_KERNEL:{mode}:/'")
        return '\n'.join(lines) + '\n'
    runs = '\n'.join(run_mode(mode) for mode in modes)
    (stage / 'gate.sh').write_text(f'''#!/bin/sh
cd /mnt/vm || exit 1
dmesg -n 1
insmod capstone_vm.ko{module_args}
echo ROOT_INSMOD:$?
{memory_mode.echo('ROOT_')}
echo "ROOT_GRANTS_PARAMETER:$(cat /sys/module/capstone_vm/parameters/root_grants 2>/dev/null || echo A)"
dmesg -c > /dev/null
{runs}
echo "ROOT_UNMAP_REFUSED:$(cat /sys/module/capstone_vm/parameters/unmap_refused 2>/dev/null || echo A)"
echo "ROOT_RETIRE_REFUSED:$(cat /sys/module/capstone_vm/parameters/retire_refused 2>/dev/null || echo A)"
echo "ROOT_EXIT_KEPT:$(cat /sys/module/capstone_vm/parameters/exit_kept 2>/dev/null || echo A)"
{'echo ROOT_CLEANUP:skipped' if a.no_unload else 'rmmod capstone_vm' + chr(10) + 'echo ROOT_CLEANUP:$?'}
echo VIRTUAL_STAGED_DONE
''')
    subprocess.run([sys.executable, str(HERE / 'run-staged.py'), '--qemu', str(a.qemu.resolve()),
                    '--images', str(a.images.resolve()), '--stage', str(stage.resolve()),
                    '--work', str((a.work / 'guest').resolve()), '--timeout', str(a.timeout),
                    '--disk-mib', '64'])
    serial = a.work / 'guest/serial.log'
    log = serial.read_text(errors='replace').replace('\r', '') if serial.exists() else ''

    def line(pattern, text=log):
        m = re.search(pattern, text, re.M)
        return m.group(1) if m else None

    insmod = line(r'^ROOT_INSMOD:(\d+)$')
    causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', log)]
    tests = {}
    outcome = None
    exits = {}
    refusals_seen = {}
    frames_reused = {}
    if a.no_refs:
        refused = insmod not in (None, '0') and 'root_grants needs supervisor references' in log
        halted = insmod is None and 'Print = Scalar(0x18)' in log and 'VIRTUAL_STAGED_DONE' not in log
        outcome = 'refused' if refused else 'halted at the probe' if halted else 'unexpected'
        tests['probe_announced'] = 'Probing supervisor references' in log
        tests['module_not_loaded'] = 'capstone_vm: loaded' not in log
        tests['refused_or_halted_at_the_probe'] = refused or halted
        tests['no_kernel_oops'] = 'Oops' not in log and 'Kernel panic' not in log
    else:
        tests['module_loaded'] = insmod == '0'
        tests['parameter_reads_back'] = line(r'^ROOT_GRANTS_PARAMETER:(\w+)$') == ('Y' if a.root_grants else 'N')
        tests['memory_mode_verified'] = memory_mode.verified(log, a.unpinned, 1)
        cuts = dict(CONTROL_CUTS.get(a.control, {}))
        cuts.update(LAUNCHER_CUTS.get(a.launcher_control, {}))
        cuts.update(LIBC_CUTS.get(a.libc_control, {}))
        checks = dict(CHECKS)
        if a.launcher_small_table:
            checks.update(CHECKS_SMALL_TABLE)
        if a.pages_run:
            checks.update(CHECKS_PAGES_RUN)
        halting = [m for m, cut in cuts.items() if m in modes and cut[2:] == ('machine-halt',)]
        reports = REPORT.findall(log)
        retire_reports = RETIRE_REPORT.findall(log)
        kept_reports = KEPT_REPORT.findall(log)
        retires_seen = {}
        for mode in modes:
            begin = log.find(f'ROOT_MODE_BEGIN:{mode}\n')
            end = log.find(f'ROOT_EXIT:{mode}:', begin)
            part = log[begin:end] if begin >= 0 and end >= 0 else ''
            if mode in halting and begin >= 0 and end < 0:
                part = log[begin:]              # the machine halted inside this mode
            status = line(rf'^ROOT_EXIT:{mode}:(\d+)$')
            exits[mode] = status
            mode_causes = [int(n) for n in re.findall(r'domain fault cause=(\d+)', part)]
            seen = [(reason, request) for m, reason, request in reports if m == mode]
            refusals_seen[mode] = seen
            retires_seen[mode] = retire_reports.count(mode)
            tests[f'{mode}:kernel_retire_refusals'] = retires_seen[mode] == 0
            kept = [(what, why) for m, what, why in kept_reports if m == mode]
            if a.root_grants:
                want_kept = EXIT_KEPT.get(mode, []) + (EXIT_KEPT_BYPASS.get(mode, []) if a.launcher_bypass else [])
            else:
                want_kept = EXIT_KEPT_RG0.get(mode, [])
            tests[f'{mode}:kernel_exit_kept'] = kept == [('stack', why) for why in want_kept]
            tests[f'{mode}:frames_released'] = 'CAPSTONE_VM_FRAME_RELEASE_FAILED' not in part
            tests[f'{mode}:grant_type'] = line(r'^ROOT_GRANT_TYPE:(\d+)$', part) == str(a.root_grants)
            if mode in ATTACH_MODES and (a.root_grants or mode != 'attach-address'):
                # The program cannot report the refusal, which ends it: the runner reads the
                # launcher's log and the exit status.
                reason, marker = ATTACH_MODES[mode]
                for name in checks[mode]:
                    if name != 'attach_address_refused':
                        tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                tests[f'{mode}:attach_never_completed'] = marker not in part
                refused = re.search(rf'^{ATTACH_REFUSED}.*reason={reason}$', part, re.M) is not None
                if a.launcher_control == 'no-writable-check' and mode == 'attach-readonly':
                    tests[f'{mode}:control_not_refused'] = ATTACH_REFUSED not in part
                    tests[f'{mode}:control_launcher_fault'] = status == '139' and mode_causes == []
                else:
                    tests[f'{mode}:transport_refused'] = refused
                    tests[f'{mode}:launcher_ended_program'] = (status == '125' and mode_causes == [] and
                                                              'capstone-vexec: service:' in part)
                    tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
                if a.root_grants:
                    tests[f'{mode}:kernel_refusals'] = refusals_match(seen, REFUSALS[mode])
                continue
            if mode == 'die-stderr':
                # stderr went into a full pipe: the program's and the launcher's lines are not in
                # the log, only the exit status and the time the launcher took to end.
                seconds = line(r'^ROOT_DIE_SECONDS:(\d+)$')   # printed after ROOT_EXIT
                tests[f'{mode}:die_stderr_blocks'] = 'ROOT_CHECK:die_stderr_blocks:PASS' in part
                tests[f'{mode}:launcher_ended_program'] = status == '125' and mode_causes == []
                tests[f'{mode}:ended_at_once'] = seconds is not None and int(seconds) < 20
                tests[f'{mode}:write_never_returned'] = 'ROOT_DIE_WRITTEN' not in part
                tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
                continue
            if mode in ENDING_MODES:
                # The launcher ends the program inside the service: the runner reads its line and
                # the exit status.
                ending_line, marker = ENDING_MODES[mode]
                for name in checks[mode]:
                    tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                tests[f'{mode}:service_never_returned'] = marker not in part
                tests[f'{mode}:launcher_ended_program'] = (status == '125' and mode_causes == [] and
                                                          ending_line in part)
                tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
                if a.root_grants:
                    tests[f'{mode}:kernel_refusals'] = refusals_match(seen, REFUSALS[mode])
                continue
            if mode in cuts:
                # The control must break this mode, at this point, and nothing else.
                marker, check, *ending = cuts[mode]
                for name in checks[mode]:
                    if name == check:
                        tests[f'{mode}:{name}:control_fails'] = f'ROOT_CHECK:{name}:PASS' not in part
                        break
                    if a.root_grants or name not in ROOT_ONLY:
                        tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                tests[f'{mode}:control_breaks_here'] = marker in part
                if not ending:
                    tests[f'{mode}:control_no_fault'] = mode_causes == []
                elif ending[0] == 'launcher-fault':
                    tests[f'{mode}:control_launcher_fault'] = status == '139' and mode_causes == []
                elif ending[0] == 'machine-halt':
                    word = line(r'^ROOT_EXIT_WORD case=unaligned word=(0x[0-9a-f]+)$', part)
                    halt = line(r'domain halted by capability fault: .*badaddr = (0x[0-9a-f]+)', part)
                    tests[f'{mode}:control_machine_halt_at_word'] = (status is None and word is not None
                                                                   and halt == word)
                else:
                    tests[f'{mode}:control_domain_fault'] = status == '139' and mode_causes != []
                if a.root_grants:
                    expected = expected_refusals(mode, a.control, a.launcher_bypass,
                                                 a.launcher_frame_floor, None)
                    if a.libc_control == 'no-remap-check' and mode == 'mremap':
                        expected = []               # the fault comes before the first refusal
                    tests[f'{mode}:kernel_refusals'] = refusals_match(seen[:len(expected)], expected)
                continue
            for name in checks[mode]:
                if mode in SUBLET_MODES and a.ids != 63:
                    tests[f'{mode}:{name}:skipped'] = re.search(rf'^ROOT_SKIP:{name}:\S', part, re.M) is not None
                elif a.root_grants or name not in ROOT_ONLY:
                    tests[f'{mode}:{name}'] = f'ROOT_CHECK:{name}:PASS' in part
                else:
                    tests[f'{mode}:{name}:skipped'] = re.search(rf'^ROOT_SKIP:{name}:\S', part, re.M) is not None
            tests[f'{mode}:no_contract_failure'] = 'ROOT_FAIL' not in part
            tests[f'{mode}:exit_word_refusals'] = reasons_match(
                EXIT_WORD_REPORT.findall(part), EXIT_WORD_REFUSALS[a.root_grants].get(mode, []))
            tests[f'{mode}:futex_refusals'] = reasons_match(
                FUTEX_REPORT.findall(part), FUTEX_REFUSALS[a.root_grants].get(mode, []))
            if mode in CENSUS_MODES and not (a.pages_run and mode == 'pages-derive' and a.root_grants):
                tests[f'{mode}:thread_census_one'] = \
                    re.search(r'^CAPSTONE_VM_THREADS count=1 ', part, re.M) is not None
            if mode == 'exit-word':
                tests['exit-word:straddle_adjacent'] = 'ROOT_STRADDLE adjacent=1' in part
            if mode == 'join-withdrawn':
                # The hooks build's witness: the joiner waited on the entry when it was withdrawn.
                tests['join-withdrawn:joiner_waited_on_entry'] = re.search(
                    r'^CAPSTONE_VM_TEST_WITHDRAWN id=\d+ waiters=[1-9]', part, re.M) is not None
            kernel = [l for l in re.findall(rf'^ROOT_KERNEL:{re.escape(mode)}:(.*)$', log, re.M)]
            waits = sum(l.startswith('step waits for a page operation') for l in kernel)
            moving = [l for l in kernel if l.startswith('resolve refused at moving')]
            if mode in PAGES_RUN_MODES and a.root_grants and a.control != 'no-mark':
                if a.pages_run:
                    tests[f'{mode}:fault_deferred_or_run'] = bool(moving) or mode == 'pages-derive'
                else:
                    tests[f'{mode}:program_stopped_in_window'] = waits >= 1 and not moving
            if mode == 'pages-derive' and a.pages_run and a.root_grants:
                tests['pages-derive:commit_fails_closed'] = 'CAPSTONE_VM_PAGES_ENDED op=2 ' in part
                tests['pages-derive:stale_access_faults_25'] = mode_causes == [25] and status == '139'
                tests['pages-derive:stale_access_did_not_read'] = 'ROOT_STALE_READ' not in part
            elif mode in FAULTING and (a.root_grants or mode not in ROOT_FAULTING):
                if mode not in NO_REMAP:
                    tests[f'{mode}:remapped_at_the_same_address'] = 'ROOT_REMAP same_address=1' in part
                tests[f'{mode}:stale_access_faults_25'] = mode_causes == [25] and status == '139'
                tests[f'{mode}:stale_access_did_not_read'] = 'ROOT_STALE_READ' not in part
            elif mode in PAGE_FAULTING:
                # A live reference into a removed page: a page fault, which the module reports as a
                # refused resolution at a removed page, never a lifetime fault.
                tests[f'{mode}:hole_access_page_fault'] = (mode_causes == [PAGE_FAULTING[mode]] and
                                                          status == '139')
                tests[f'{mode}:hole_access_did_not_read'] = 'ROOT_HOLE_SURVIVED' not in part
                tests[f'{mode}:module_refused_removed_page'] = \
                    f'ROOT_KERNEL:{mode}:resolve refused at removed' in log
            else:
                tests[f'{mode}:exit_zero'] = mode_causes == [] and status == '0'
            if mode == 'exit-word':
                tests['exit-word:stack_word_dropped_once'] = \
                    len(re.findall(r'^CAPSTONE_VM_EXIT_WORD_DROPPED ', part, re.M)) == 1
            if mode in ('eintr-word', 'eintr-exit'):
                op = mode.split('-')[1]
                retries = line(rf'^CAPSTONE_VM_TEST_EINTR op={op} retries=(\d+)$', part)
                tests[f'{mode}:ioctl_interrupted_and_repeated'] = retries is not None and int(retries) >= 1
            if mode == 'futex-park' and not a.unpinned:
                pool = re.search(r'^CAPSTONE_VM_MAPCACHE .*\bparks=(\d+) hits=(\d+)', part, re.M)
                tests['futex-park:parked_and_revived'] = (pool is not None and int(pool.group(1)) >= 1
                                                          and int(pool.group(2)) >= 1)
            if mode == 'futex-readonly':
                # diag_admit: the futex read admits no page, the exit word's store admits one.
                base = line(r'^ROOT_FUTEX_RO base=(0x[0-9a-f]+)$', part)
                store = line(r'^ROOT_FUTEX_RO store_base=(0x[0-9a-f]+)$', part)
                touched = re.findall(r'^ROOT_KERNEL:futex-readonly:(?:unpinned )?touch arena (0x[0-9a-f]+)\+',
                                     log, re.M)
                tests['futex-readonly:read_admits_no_page'] = base is not None and base not in touched
                tests['futex-readonly:store_admits_its_page'] = store is not None and store in touched
            if mode == 'pages-enomem':
                aborts = ABORT_REPORT.findall(part)
                tests['pages-enomem:aborts_logged'] = aborts == [('protect', '12')] * (2 if a.root_grants else 1)
            if mode == 'release-fail':
                tests['release-fail:release_failure_logged'] = 'CAPSTONE_VM_RELEASE_FAILED ' in part
                # Block A: the launcher counts the failed releases (CAPSTONE_VM_RELEASES at exit).
                tests['release-fail:release_failure_counted'] = re.search(
                    r'^CAPSTONE_VM_RELEASE_FAILED .* failed=1$', part, re.M) is not None
            releases = re.search(r'^CAPSTONE_VM_RELEASES failed=(\d+) held=(\d+) top_guard=(\d+)$', part, re.M)
            if releases and not (a.launcher_hooks and mode == 'release-fail'):
                tests[f'{mode}:no_release_failed'] = releases.group(1) == '0'
            if releases:
                held_expected = '1' if mode == 'pages-unclear' and a.root_grants and not a.unpinned else '0'
                tests[f'{mode}:held_extents'] = releases.group(2) == held_expected
                tests[f'{mode}:top_page_guarded'] = releases.group(3) == '1'
            if mode == 'top-page':
                # Block A, F4: the launcher holds the topmost page below 2^38, grant_mapping refuses it
                # (ENOMEM) and, past that check, the module's add() (EINVAL).
                tests['top-page:launcher_and_module_refuse_top_page'] = re.search(
                    r'^CAPSTONE_VM_TEST_TOP_PAGE guard=1 launcher=12 module=22$', part, re.M) is not None
                tests['top-page:module_refusal_logged'] = re.search(
                    r'^CAPSTONE_VM_ADD_REFUSED base=0x3ffffff000 bytes=4096 errno=22 ', part, re.M) is not None
            if mode == 'pages-unclear' and a.root_grants and not a.unpinned:
                base = line(r'^ROOT_PAGES_UNCLEAR base=(0x[0-9a-f]+)$', part)
                ktext = '\n'.join(kernel)
                unheld = re.findall(rf'^test unhold arena {base}\+0x[0-9a-f]+ page 0 pfn (0x[0-9a-f]+)$',
                                    ktext, re.M) if base else []
                refused_pfns = re.findall(r'^clear in place refused pfn (0x[0-9a-f]+) at ', ktext, re.M)
                closed = re.findall(rf'^page operation closed: arena {base}\+0x[0-9a-f]+ retire preparation '
                                    r'failed \(-?\d+\), unheld pages cleared in place, (-?\d+) left as they were$',
                                    ktext, re.M) if base else []
                tests['pages-unclear:unheld_page_left_uncleared'] = (len(unheld) == 1 and
                                                                     refused_pfns == unheld and closed == ['1'])
                tests['pages-unclear:extent_held_logged'] = re.search(
                    rf'^CAPSTONE_VM_RELEASE_HELD base={base} bytes=\d+ errno=0 held=1$', part, re.M) is not None
            if mode == 'realloc-exhausted':
                tries = re.findall(r'^ROOT_REALLOC_TRY freed=(\d+) null errno=(\d+)$', part, re.M)
                summary = line(r'^ROOT_REALLOC_EXHAUSTED (filled=\d+ nulls=\d+ moved_at=-?\d+)$', part)
                frames_reused[mode] = {'tries': tries, 'summary': summary, 'status': status,
                                       'causes': mode_causes}
            if mode == 'pages-tagged':
                base = line(r'^ROOT_PAGES_TAGGED base=(0x[0-9a-f]+)$', part)
                counts = re.search(r'^ROOT_PAGES_TAGGED cleared_before=(-?\d+) cleared_after=(-?\d+)$', part, re.M)
                if a.control == 'no-check-pin' and base:
                    removed = re.findall(rf'^remove arena {base}\+0x[0-9a-f]+ page 1 ', '\n'.join(kernel), re.M)
                    unheld = re.findall(rf'^test unhold arena {base}\+0x[0-9a-f]+ page 1 ', '\n'.join(kernel), re.M)
                    delta = (int(counts.group(2)) - int(counts.group(1))) if counts else -1
                    tests['pages-tagged:control_unheld_page_reaches_linux_tagged'] = (
                        len(unheld) == 1 and not removed and delta >= 1)
                elif a.root_grants and not a.unpinned and base:
                    removed = re.findall(rf'^remove arena {base}\+0x[0-9a-f]+ page 1 pfn (0x[0-9a-f]+) '
                                         r'tags (\d+) after (\d+)$', '\n'.join(kernel), re.M)
                    tests['pages-tagged:census_tags_cleared'] = (len(removed) == 1 and int(removed[0][1]) >= 1
                                                                 and removed[0][2] == '0')
                    if a.module_hooks:
                        unheld = re.findall(rf'^test unhold arena {base}\+0x[0-9a-f]+ page 1 pfn (0x[0-9a-f]+)$',
                                            '\n'.join(kernel), re.M)
                        scanned = re.findall(rf'^scan arena {base}\+0x[0-9a-f]+ page 1 pfn (0x[0-9a-f]+) ',
                                             '\n'.join(kernel), re.M)
                        tests['pages-tagged:unheld_page_pinned_by_check'] = (
                            len(unheld) == 1 and scanned == unheld and bool(removed) and removed[0][0] == unheld[0])
                if counts and int(counts.group(1)) >= 0 and int(counts.group(2)) >= 0 and \
                        a.control != 'no-check-pin':
                    delta = int(counts.group(2)) - int(counts.group(1))
                    if a.unpinned:
                        tests['pages-tagged:kernel_cleared_tagged_frame'] = delta >= 1
                    elif a.root_grants:
                        tests['pages-tagged:no_tagged_frame_freed'] = delta == 0
            if mode == 'pages-closed' and a.root_grants and not a.unpinned:
                # Block A: the fail-closed commit whose retire preparation failed clears the unheld
                # page in place (its frame in a "clear in place" line), and no tagged frame reaches
                # Linux (captag's count of tagged frames cleared at their free does not grow).
                base = line(r'^ROOT_PAGES_CLOSED base=(0x[0-9a-f]+)$', part)
                counts = re.search(r'^ROOT_PAGES_CLOSED fixed=\d+ errno=\d+ registered=\d+ '
                                   r'cleared_before=(-?\d+) cleared_after=(-?\d+)$', part, re.M)
                ktext = '\n'.join(kernel)
                unheld = re.findall(rf'^test unhold arena {base}\+0x[0-9a-f]+ page 0 pfn (0x[0-9a-f]+)$',
                                    ktext, re.M) if base else []
                cleared = re.findall(r'^clear in place pfn (0x[0-9a-f]+) at ', ktext, re.M)
                closed = re.findall(rf'^page operation closed: arena {base}\+0x[0-9a-f]+ retire preparation '
                                    r'failed \(-?\d+\), unheld pages cleared in place, (-?\d+) left as they were$',
                                    ktext, re.M) if base else []
                delta = (int(counts.group(2)) - int(counts.group(1))
                         if counts and int(counts.group(1)) >= 0 and int(counts.group(2)) >= 0 else None)
                if a.control == 'no-close-clear':
                    tests['pages-closed:control_unheld_page_reaches_linux_tagged'] = (
                        len(unheld) == 1 and not cleared and delta is not None and delta >= 1)
                else:
                    tests['pages-closed:retire_preparation_failed'] = closed == ['0']
                    tests['pages-closed:extent_released'] = 'CAPSTONE_VM_RELEASE_HELD' not in part
                    tests['pages-closed:unheld_page_cleared_in_place'] = len(unheld) == 1 and unheld[0] in cleared
                    tests['pages-closed:no_tagged_frame_freed'] = delta == 0
            if mode == 'pages-discard' and a.root_grants and a.launcher_control != 'discard-abort':
                tests['pages-discard:discard_failure_logged'] = re.search(
                    r'^CAPSTONE_VM_PAGES_FAILED op=2 base=0x[0-9a-f]+ pages=2 step=discard errno=11$',
                    part, re.M) is not None
            if mode == 'pages-admit' and a.root_grants and not a.unpinned:
                base = line(r'^ROOT_PAGES_ADMIT base=(0x[0-9a-f]+)$', part)
                order = re.findall(rf'^(admit|remove) arena {base}\+0x[0-9a-f]+ page 2 .*?pfn (0x[0-9a-f]+)',
                                   '\n'.join(kernel), re.M) if base else []
                admitted, unpinned_mapped = set(), False
                for kind, pfn in order:
                    if kind == 'admit':
                        admitted.add(pfn)
                    elif pfn in admitted:
                        unpinned_mapped = True
                if a.control == 'no-mark':
                    tests['pages-admit:control_frame_unpinned_while_mapped'] = unpinned_mapped
                else:
                    tests['pages-admit:no_frame_unpinned_while_mapped'] = bool(base) and not unpinned_mapped
                    tests['pages-admit:page_admitted_after_commit'] = any(k == 'admit' for k, _ in order)
            if mode == 'partial-fixed' and a.root_grants and not a.unpinned:
                base = line(r'^ROOT_PARTIAL_FIXED base=(0x[0-9a-f]+)$', part)
                removed = re.findall(rf'^ROOT_KERNEL:partial-fixed:remove arena {base}\+0x[0-9a-f]+ page (\d+) '
                                     r'pfn (0x[0-9a-f]+) tags (\d+) after (\d+)$', log, re.M) if base else []
                admitted = re.findall(rf'^ROOT_KERNEL:partial-fixed:admit arena {base}\+0x[0-9a-f]+ page (\d+) '
                                      r'cause \d+ pc 0x[0-9a-f]+ pfn (0x[0-9a-f]+)$', log, re.M) if base else []
                tagged = [r for r in removed if r[0] == '1' and int(r[2]) >= 1]
                census = bool(tagged) and all(int(r[3]) == 0 for r in removed)
                if a.control == 'no-tag-clear':
                    tests['partial-fixed:control_census_shows_tags'] = any(int(r[3]) >= 1 for r in tagged)
                else:
                    tests['partial-fixed:census_tags_cleared'] = census
                reused = sorted({pfn for page, pfn, *_ in removed} & {pfn for page, pfn in admitted})
                frames_reused[mode] = reused
                if a.block_frames:
                    tests['partial-fixed:refill_admits_reused_frame'] = bool(reused)
            if mode in STACKS_KEPT and (a.root_grants or mode != 'exit-victim'):
                stacks_kept = len(re.findall(r'^CAPSTONE_VM_STACK_UNMAP_REFUSED ', part, re.M))
                tests[f'{mode}:launcher_stacks_kept'] = stacks_kept == STACKS_KEPT[mode]
            if mode == 'refuse' and a.root_grants and a.ids != 63:
                collections = line(r'^CAPSTONE_VM_STATS .*\bcollections=(\d+)', part)
                tests['refuse:collected'] = collections is not None and int(collections) >= 1
            if a.root_grants:
                victim_reason = None
                if mode == 'exit-victim':
                    operand = line(r'^ROOT_OPERAND own_stack_at_victim type=(\d+) adjacent=1 tries=\d+$', part)
                    victim_reason = 'not a reference' if operand == '7' else 'not the mapping'
                    tests['exit-victim:operand_is_a_reference'] = operand == '1'
                expected = expected_refusals(mode, a.control, a.launcher_bypass,
                                             a.launcher_frame_floor, victim_reason,
                                             a.launcher_range_bypass, a.pages_run)
                tests[f'{mode}:kernel_refusals'] = refusals_match(seen, expected)
            else:
                tests[f'{mode}:kernel_refuses_nothing'] = seen == []
        default_launcher = (a.adapter / 'capstone-vexec').read_bytes()
        tests['default_launcher_has_no_test_hooks'] = b'CAPSTONE_VM_TEST_' not in default_launcher
        if a.launcher_hooks and a.launcher:
            tests['hooks_launcher_has_test_hooks'] = b'CAPSTONE_VM_TEST_EINTR' in a.launcher.read_bytes()
        default_module = (a.adapter / 'module/capstone_vm.ko').read_bytes()
        tests['default_module_has_no_test_hooks'] = b'test_hold_ms' not in default_module
        if a.module_hooks:
            tests['hooks_module_has_test_hooks'] = b'test_hold_ms' in a.module.read_bytes()
        refused_count = line(r'^ROOT_UNMAP_REFUSED:(\w+)$')
        if halting:
            # The machine halted: the gate's closing lines never come.
            tests['gate_did_not_end'] = 'VIRTUAL_STAGED_DONE' not in log
        else:
            tests['kernel_count_matches_reports'] = refused_count == str(len(reports))
            tests['kernel_kept_count_matches_reports'] = \
                line(r'^ROOT_EXIT_KEPT:(\w+)$') == str(len(kept_reports))
            tests['module_unloaded'] = 'ROOT_CLEANUP:' + ('skipped' if a.no_unload else '0') in log
        tests['no_kernel_fault'] = 'Oops' not in log and 'Kernel panic' not in log

    result = dict(status='PASS' if all(tests.values()) else 'FAIL', root_grants=a.root_grants,
                  control=a.control, launcher_control=a.launcher_control,
                  libc_control=a.libc_control, launcher_small_table=a.launcher_small_table,
                  launcher=str(a.launcher) if a.launcher else None,
                  launcher_bypass=a.launcher_bypass, launcher_frame_floor=a.launcher_frame_floor,
                  launcher_hooks=a.launcher_hooks,
                  no_refs=a.no_refs, unpinned=a.unpinned,
                  passed=sum(tests.values()), total=len(tests), tests=tests,
                  insmod=insmod, exits=exits, fault_causes=causes, refusals=refusals_seen,
                  retire_refusals=retires_seen if not a.no_refs else None,
                  retire_refused=line(r'^ROOT_RETIRE_REFUSED:(\w+)$'),
                  exit_kept=line(r'^ROOT_EXIT_KEPT:(\w+)$'), no_refs_outcome=outcome,
                  module_hooks=a.module_hooks, block_frames=a.block_frames,
                  launcher_range_bypass=a.launcher_range_bypass, frames_reused=frames_reused,
                  pages_run=a.pages_run, ids=a.ids, module_args=a.module_args, no_unload=a.no_unload)
    (a.work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'total', 'exits', 'fault_causes')}))
    for name, ok in tests.items():
        if not ok:
            print('FAIL', name)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
