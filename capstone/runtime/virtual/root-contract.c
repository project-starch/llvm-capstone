/* Root grants (module parameter root_grants=1): each mapping is a block of depth 1, a child of the
 * one reference over the address space, R, which only the kernel holds; the kernel keeps no reference
 * to the block. Every request of the program to end a block or to change its pages comes with a
 * reference: munmap, the stack at thread exit, the frame's hand-over at thread creation, mprotect,
 * madvise, MAP_FIXED and mremap (stages 2 to 4f and the stage "drop M" of
 * docs/plans/2026-11-12-address-space-root-adapter.md). The kernel accepts a reference only if it is
 * live, its record lies one level below R (a block's, never an object's inside the block) and its
 * bounds cover the page range that the call names; a call that removes the block's last mapped pages,
 * a call that names the whole block included, ends the block with CREVOKE on that reference. The
 * kernel-owned blocks (the image, the main stack, the shared region, the startup block) refuse every
 * such request, in both modes. run-root-grants.py runs this program once per mode (argv[1]) in one
 * boot. Build with the virtual SDK.
 *
 * Every mode first reads the module's parameter root_grants (ROOT_GRANT_TYPE: 1 with root grants, 0
 * without; until block A it read the type of a grant, non-linear or linear, which the 63-bit format
 * no longer stores) and prints the type LCC answers for a grant (ROOT_GRANT_LCC). A check that holds
 * only with root grants prints ROOT_SKIP with the reason without them, so the same program runs in
 * both forms.
 * A check prints ROOT_CHECK:<name>:PASS; a failed requirement prints ROOT_FAIL and exits 1.
 *
 * unmap (the default; ends with a fault):
 *   grant_reads_and_writes          a mapped page reads zero and keeps what the program writes
 *   libc_accepts_copyable_grant     malloc of a small and of a large block and a second mmap work
 *   unmap_through_mapping_reference munmap through the mapping's reference succeeds
 *   other_mapping_survives_unmap    the other mappings keep their contents and take writes
 *   second_unmap_refused            a second munmap of the same mapping is refused
 *   stale_reference_at_reused_address_refused (root grants)
 *                                   the program maps the same address again; munmap through the
 *                                   old reference is refused and the new mapping still works
 *   stale access                    the last line reads through an object derived in the
 *                                   unmapped mapping: it must fault with cause 25, so the program
 *                                   must not reach ROOT_STALE_READ.
 * refuse (exit 0; nothing faults):
 *   address_refused (root grants)   munmap with the mapping's address as an integer
 *   nested_object_refused (root grants)
 *                                   munmap through an object derived over the whole page, one
 *                                   level below the mapping; the object and the mapping still work
 *   unmap_beyond_block_refused      munmap of a range that reaches beyond a two-page mapping (since
 *                                   stage 4e a page range inside the mapping is a partial munmap);
 *                                   the whole mapping then unmaps through its reference
 *   stale_reference_refused (root grants)
 *                                   the program ends a mapping itself with CREVOKE: munmap
 *                                   through the stale reference is refused, twice, without a fault
 *   stale_reference_after_collection_refused (root grants)
 *                                   the same after enough lifetimes ended for the adapter to
 *                                   collect identities (CAPSTONE_VM_STATS collections)
 * narrow (exit 0, or 1 under the negative control):
 *   narrowed_reference_refused (root grants)
 *                                   munmap through a reference narrowed to the first page of a
 *                                   two-page mapping is refused, and both pages still work
 *   whole_reference_after_narrowed  the whole mapping then unmaps through its own reference
 * renew (ends with a fault):
 *   renewed_old_reference_refused (root grants)
 *                                   after CRENEW, munmap through the reference from before it is
 *                                   refused, and the renewed mapping still works
 *   renewed_mapping_unmaps          munmap through the renewed reference succeeds; the last line
 *                                   reads through an object derived in the renewed mapping and
 *                                   must fault with 25
 * renew-ref (ends with a fault): as renew, and the last line reads through the renewed reference
 *                                   itself.
 * mremap (ends with a fault): the libc's mremap (map, copy, unmap) of a two-page mapping that
 *                                   holds data and, at offset 64, a reference to another mapping.
 *                                   Its mapping check refuses before anything is copied.
 *   mremap_beyond_block_refused     an old range that reaches beyond the mapping: mremap fails with
 *                                   EINVAL, and the old mapping keeps its contents and takes writes
 *                                   (since stage 4e any page range inside a block may move)
 *   mremap_oversized_old_refused    the same with an old length beyond the mapping: EINVAL, no
 *                                   fault (a copy before the check reads past the reference)
 *   mremap_nested_object_refused (root grants)
 *                                   the same through an object over the whole mapping
 *   mremap_same_size_returns_reference
 *                                   a resize to the same page count returns the reference
 *   mremap_same_size_stale_refused (root grants)
 *                                   the same through the reference from before CRENEW of another
 *                                   mapping: EINVAL; the renewed reference then succeeds
 *   mremap_read_only_refused        the mapping made read-only: ENOTSUP, the mapping stays
 *                                   read-only with its contents; then write rights are restored
 *   public_mremap_grows             mremap to three pages moves the data and the stored reference,
 *                                   which still reads its object; the third page reads zero
 *   mremap_old_reference_refused    munmap through the old reference afterwards is refused
 *                                   (the old mapping is gone)
 *   public_mremap_shrinks           mremap of the grown mapping to one page shrinks it in place
 *                                   (stage 4e): the reference comes back and keeps the data of that
 *                                   page and the stored reference; the munmap of that last page ends
 *                                   the mapping, and the last line reads through its reference after
 *                                   the program mapped its address again, and must fault with 25.
 * exit-victim (exit 0; root grants):
 *   exit_with_own_stack_at_victim_refused
 *                                   a thread exits naming a victim mapping with its own stack's
 *                                   reference moved to the victim's address, which is the stack's
 *                                   end (the stack lies right below the victim), so the operand is
 *                                   a live reference to another mapping; the victim still works
 *                                   afterwards
 *   exit_with_victim_address_refused
 *                                   the same with the victim's address as an integer
 *   victim_unmaps_afterwards        the victim then unmaps through its own reference
 * exit-renew (ends with a fault):
 *   stack_renewed_before_exit       a thread renews its own stack's mapping with CRENEW and exits
 *                                   naming the renewed reference; the main thread kept a copy.
 *                                   The last line reads through that copy after the main thread
 *                                   mapped the stack's address again: it must fault with 25.
 * create (ends with a fault):
 *   create_with_stale_frame_refused (root grants)
 *                                   the creator renews the frame's mapping and passes the
 *                                   reference from before: the create is refused, and the renewed
 *                                   reference still works
 *   create_with_renewed_frame       the create through the renewed reference succeeds; the last
 *                                   line reads the frame through it and must fault with 25.
 * create-fail (exit 0): contexts that wait until the main thread releases them are created
 *                                   until a create fails. An exit frees its table entry, so the
 *                                   launcher's table (32 entries) fills only with live threads,
 *                                   as the module's limit (32 threads) does:
 *   module_limit_keeps_frame        with the launcher's own table the module refuses first
 *                                   (ENOSPC), and the frame stays the program's and unmaps
 *   failed_create_releases_frame    behind a launcher with a smaller table
 *                                   (run-root-grants.py --launcher-small-table) the launcher fails
 *                                   the create after the module took the frame: EAGAIN, and the
 *                                   frame's mapping is no longer registered (registered())
 *   failed_create_id_refused        either way the failed create issued no id: a join of the id
 *                                   after the last created one answers ESRCH
 * join-two (exit 0):
 *   two_joiners_return              two contexts join one context that waits until released;
 *                                   after the release both joins return 0
 *   thread_after_two_joiners        a new context is then created and joined (the main thread's
 *                                   table census at exit, CAPSTONE_VM_THREADS, shows count=1)
 * join-after (exit 0):
 *   join_after_word_exit            a join of a context that exited with an exit word, after the
 *                                   launcher wrote the word, returns 0
 *   join_after_raw_exit             the same for a context that exited without an exit word
 *   join_unknown_id_refused         a second join of both returns 0; an id never issued: ESRCH
 * exit-many (exit 0):
 *   exit_unjoined_many_then_create  40 contexts exit without an exit word one after the other and
 *                                   none is joined; then one more is created and joined
 * exit-word (exit 0): contexts exit naming an exit word the launcher must not write. Each exit
 *                                   completes, the join returns 0, and the word's bytes keep
 *                                   their value. The launcher logs each refusal
 *                                   (CAPSTONE_VM_EXIT_WORD_REFUSED reason=...)
 *   unaligned_exit_word_not_written a word one byte into an int of the image's data
 *   straddling_exit_word_not_written
 *                                   a word over the end of one mapping and the start of the
 *                                   mapping right above it, which is the exiting thread's stack
 *                                   and is released at the exit
 *   read_only_exit_word_not_written a word in a page the program made read-only
 *   writable_exit_word_written      the same word after write rights are restored: written
 *   stack_exit_word_dropped         a word inside the stack the exit names (stage 4e): the stack is
 *                                   released and the store dropped without an error
 *   frame_exit_word_not_written     a word in the exiting thread's own frame, which the exit
 *                                   releases, named through the frame's reference from before the
 *                                   create (stale with root grants)
 *   Every exit word goes to the exit service as a reference (stage 4c).
 * exit-word-refs (exit 0): exit words through references (stage 4c)
 *   other_frame_word_not_written    a word in another thread's frame (its saved return address),
 *                                   named through the frame's reference from before its create;
 *                                   that thread runs on and is joined
 *   middle_protected_word_written   a word in the third page of a mapping whose middle page was
 *                                   made inaccessible and restored: written, and a futex waiter on
 *                                   it returns
 *   stale_reference_word_not_written (root grants)
 *                                   a word through the reference from before CRENEW
 *   narrow_reference_word_not_written (root grants)
 *                                   a word through a reference that covers two of its bytes
 * futex (exit 0): a wait reads the word and needs its reference; a wake takes an integer
 *   futex_wait_through_reference    a wait through the word's reference times out (its value) or
 *                                   answers EAGAIN (another value)
 *   futex_wait_integer_refused, futex_wait_stale_refused, futex_wait_narrow_refused (root grants)
 *                                   a wait on an integer address, through the reference from
 *                                   before CRENEW, through a reference covering half the word:
 *                                   EFAULT
 *   futex_wait_frame_refused        a wait on a word in another thread's frame: EFAULT
 *   futex_wake_integer_answers      a wake by integer address answers 0, outside every mapping
 *                                   EFAULT
 * advise (exit 0): madvise(MADV_DONTNEED) and MADV_FREE need a reference that covers the range
 *   advise_integer_refused, advise_stale_refused, advise_narrowed_refused,
 *   advise_nested_object_refused (root grants)
 *                                   EINVAL, and the data stays
 *   advise_zeroes_covered_page      through a reference narrowed to the second page of two: that
 *                                   page reads zero, a reference stored in it is gone, the first
 *                                   page keeps its data
 *   advise_free_zeroes              MADV_FREE through the mapping's reference: all of it reads zero
 *   advise_prot_none_zeroes         madvise of a page that held data and a stored reference and was
 *                                   made PROT_NONE first: after mprotect back to read-write the page
 *                                   reads zero, the reference is gone, the first page keeps its data
 *                                   (stage 4d)
 * Stage 4d, check and access in one step. The modes marked (hooks) need a launcher built with
 * -DCAPSTONE_VM_TEST_HOOKS and the delays run-root-grants.py sets for them (--launcher-hooks):
 * futex-race (exit 0; hooks: the futex wait sleeps 2 s after its compare, service_lock released):
 *   futex_wait_never_reads_new_mapping
 *                                   a context waits on a word that holds 5; inside the window the
 *                                   main thread unmaps the word's whole mapping, maps a new one at
 *                                   the same address and writes 6 there: the unmap wakes the waiter
 *                                   (stage 4e), which answers 0, since it compared before the unmap
 *                                   (a read of the new mapping answers EAGAIN)
 *   futex_wake_in_window_caught     inside the window the main thread stores 0 and wakes: the wake
 *                                   counts the waiter, which returns 0
 * futex-remap (exit 0):
 *   futex_wake_reaches_only_its_mapping
 *                                   a context sleeps on a word; the main thread unmaps its mapping,
 *                                   which wakes the waiter (0, stage 4e), maps a new one at the same
 *                                   address and wakes that address: the wake counts nobody
 * exit-race (exit 0; root grants; hooks: the exit sleeps 3 s after the launcher's own checks):
 *   exit_word_after_revoke_not_written
 *                                   a context exits naming an exit word inside an object; inside the
 *                                   window the main thread ends the object with CREVOKE, derives a
 *                                   new object at the same address and writes 0x7777 into it: the
 *                                   word is refused (not live) and the new object keeps its value
 * join-withdrawn (exit 0; hooks: a join sleeps 1 s after releasing service_lock, and the second
 *                                   create fails after thread_add with a delay of 2 s):
 *   joiner_on_withdrawn_id_returns  a context joins the id of the next create, which fails while the
 *                                   joiner waits on its entry: the join answers ESRCH, and the next
 *                                   create takes the id
 * rollback-fail (exit 125; hooks: the first create fails and its rollback is refused):
 *   rollback_create_started         printed before the create; the launcher ends the program
 *                                   (capstone-vexec: thread rollback), ROOT_ROLLBACK_RETURNED never
 *                                   prints
 * die-blocked (exit 125):
 *   die_reader_blocks               the main thread blocks in a read of an empty pipe on the process
 *                                   transport; a pthread names the launcher's exchange as its
 *                                   transport, and the launcher ends the program with 125 without a
 *                                   crash; ROOT_DIE_READ never prints
 * word-metadata (exit 0):
 *   metadata_futex_wait_refused     a futex wait on a word in a mapping of the allocator's metadata:
 *                                   EFAULT
 *   metadata_exit_word_not_written  an exit word there is refused and keeps its value
 * join-burned (exit 0):
 *   join_failed_create_refused      a create on a read-only frame fails; a join of the id after the
 *                                   last created one answers ESRCH
 *   failed_create_consumes_no_id    the next create takes that id
 *   main_self_join_deadlock         the main thread's join of itself (id 0) answers EDEADLK
 * attach-exchange (exit 125):
 *   launcher_exchange_reference     the program holds a reference to the launcher's exchange region
 *                                   (_start's saved a1); a pthread that detached its transport then
 *                                   names it as its own exchange region: the launcher refuses the
 *                                   attachment and ends the program, ROOT_ATTACH_RETURNED never prints
 * attach-address (exit 125 with root grants, else a skip and exit 0): the same with a fresh mapping
 *                                   named by its plain address
 * attach-readonly (exit 125):
 *   attach_protect_before_attach    a pthread detaches its transport (the launcher's hold ends) and
 *                                   makes its exchange region read-only; its read() then attaches
 *                                   the transport again, the launcher refuses the attachment
 *                                   (CAPSTONE_VM_TRANSPORT_REFUSED) and ends the program, so
 *                                   ROOT_ATTACH_READ never prints
 * detached (exit 0):
 *   detached_exit_unmaps_stack      a detached pthread exits through musl's __unmapself: its stack
 *                                   mapping is registered before the exit and gone after it.
 * exit-length (exit 0):
 *   exit_with_wrong_length_completes
 *                                   a thread exits naming its own stack's mapping with a length
 *                                   that is not the mapping's: the joiner returns and the launcher
 *                                   wrote the exit word
 *   wrong_length_stack_stays_mapped the stack's mapping is still registered and works through the
 *                                   reference the main thread kept
 *   wrong_length_stack_unmaps_afterwards
 *                                   munmap through that reference then releases it
 * exit-huge (exit 0): the same with the length 2^64 - 1 (exit_with_huge_length_completes,
 *                                   huge_length_stack_stays_mapped,
 *                                   huge_length_stack_unmaps_afterwards). The exit word lies
 *                                   outside the stack, below the named base plus that length.
 * exit-frame (exit 0):
 *   exit_with_frame_as_stack_completes
 *                                   a thread exits naming its own frame's mapping as its stack:
 *                                   the joiner returns and the launcher wrote the exit word
 *   frame_as_stack_released_once    the frame's mapping is no longer registered, and the thread's
 *                                   stack still works and unmaps
 * protect (exit 0, or 1 under the negative control):
 *   protect_through_mapping_reference
 *                                   mprotect through the mapping's reference changes the
 *                                   protection of the whole mapping and of one page of it, as
 *                                   /proc/self/maps shows
 *   protect_integer_address_refused (root grants)
 *                                   mprotect with the mapping's address as an integer is refused
 *                                   and the protection is unchanged
 *   protect_stale_reference_refused (root grants)
 *                                   the same through the reference from before CRENEW
 *   protect_narrowed_reference_refused (root grants)
 *                                   the same through a reference narrowed to the first page of a
 *                                   two-page mapping, for both pages
 *   protect_nested_object_refused (root grants)
 *                                   the same through an object derived over the whole page
 *   protect_narrowed_covering_range a narrowed reference changes the page it covers: an mprotect
 *                                   may name part of a mapping (musl opens a stack past its guard)
 * protect-held (exit 0, or 1 under the negative control):
 *   protect_held_mapping_refused    a pthread's mprotect of its own exchange region, which the
 *                                   launcher holds while the thread is attached, is refused
 *                                   with EBUSY
 *   protect_after_detach            after the thread detaches its transport, the same mprotect
 *                                   succeeds
 * create-gate (exit 0, or 1 under the negative control):
 *   create_metadata_frame_refused   a frame in a one-page mapping of the allocator's metadata is
 *                                   refused (EINVAL), and the page still works
 *   create_launcher_page_refused    the launcher's own stack mapping, named at its base, is refused
 *   create_mismatched_frame_refused a frame named with its reference moved inside the page is
 *                                   refused, and the frame still works. The launcher refuses it;
 *                                   a launcher that names the mapping around the cursor
 *                                   (run-root-grants.py --launcher-frame-floor) leaves it to the
 *                                   module, which compares the frame with a0's cursor
 *   create_after_refusals           the create through the frame's own reference then succeeds
 *
 * Stage 4e, the follow-ups of stage 4d:
 * futex-readonly (exit 0):
 *   futex_wait_read_only_compares   a wait on a word of a read-only page never written compares
 *                                   (ETIMEDOUT with 0, EAGAIN with 1): the read admits no page
 *   futex_wait_protected_compares   the same for a page that held data and was made read-only
 * futex-park (exit 0; the runner turns the mapping pool on):
 *   futex_waiter_woken_at_park      a waiter on a word of a heap mapping returns 0 when the program
 *                                   unmaps (parks) the mapping, and a wake of the mapping revived at
 *                                   the same address finds nobody
 * futex-end (ends with a fault):
 *   futex_spurious_wake_rewaits     a waiter in the libc's loop reads the word again after a wake
 *                                   that left it unchanged and waits again; then the munmap of the
 *                                   word's mapping wakes it, and its read through the stale reference
 *                                   must fault with 25
 * futex-flags (exit 0):
 *   futex_clock_realtime_refused    FUTEX_CLOCK_REALTIME with FUTEX_WAIT or FUTEX_WAKE: ENOSYS
 *   futex_unknown_flag_refused      an unknown flag bit: ENOSYS
 *   futex_huge_timeout_waits        a wait with a relative timeout of 2^63 - 1 seconds sleeps until
 *                                   woken (the deadline overflowed before)
 * eintr-word, eintr-exit (exit 0; a hooks launcher and a module built with CV_TEST_HOOKS whose
 *                                   test_hold_ms the runner sets: another launcher thread holds
 *                                   vm_lock while a signal interrupts the wait for it):
 *   eintr_futex_wait_retries        the futex wait compares as without the signal
 *   eintr_thread_exit_retries       the exit completes and writes its word
 * advise-refuse (exit 0; a module built with CV_TEST_HOOKS, test_advise_refuse=3):
 *   advise_refused_changes_nothing  a madvise whose third page cannot be cleared fails with EIO and
 *                                   no page of the range changes
 * die-stderr (exit 125; the runner sends stderr into a pipe nobody reads):
 *   die_stderr_blocks               the main thread blocks in a write to the full stderr pipe; a
 *                                   pthread's malformed transport (the process meta block with a
 *                                   wrong exchange region) ends the program with 125 at once
 * protect-exec (exit 0):
 *   image_protect_refused           mprotect and madvise of the image's text and data: EINVAL
 *   image_unmap_refused             munmap of a page of the image or of the whole image: EINVAL
 *   map_exec_refused, protect_exec_refused, fixed_exec_refused
 *                                   PROT_EXEC at mmap, mprotect and MAP_FIXED: EACCES
 *   startup_stays_read_only         (block A, F4) the startup block's reference, as _start left it on
 *                                   the main stack: mprotect to read-write, MAP_FIXED, madvise and
 *                                   munmap of it are refused, and its page stays read-only
 * Stage 4e, partial munmap as a page operation:
 * partial-hole (ends with a page fault, cause 13):
 *   partial_unmap_middle, partial_unmap_wakes_waiter
 *                                   munmap of the middle page of three; a futex waiter on a word of
 *                                   that page returns 0
 *   outer_pages_work                the outer pages keep their data and take writes through the old
 *                                   reference
 *   no_mapping_in_hole              16 new mappings all lie outside the block
 *   hole_unmap_repeats              munmap of the hole twice more succeeds and changes nothing
 *   hole_requests_refused           mprotect and madvise of the hole: ENOMEM; a futex wait there:
 *                                   EFAULT; a wake answers EFAULT in the hole only
 *   the last line reads the hole through the live reference: a page fault, not a lifetime fault
 * partial-all (ends with a fault):
 *   hole_unmap_idempotent           the second page three more times: the live pages still work
 *   prot_none_page_counts           with the first page PROT_NONE and the others removed, the block
 *                                   lives
 *   count_exact_with_mprotect       the first page restored keeps its data
 *   last_page_ends_block            its munmap ends the block (CREVOKE on the passed reference); a copy
 *                                   of the block's reference that the program stored in another mapping
 *                                   must fault with 25 at the reused address
 * partial-narrow (exit 0): a reference acts only on the pages its bounds cover.
 *   partial_narrowed_beyond_refused, partial_nested_object_refused, partial_address_refused (root grants)
 *                                   a reference narrowed to the middle page cannot remove the middle
 *                                   and the last page; an object over the whole block (depth 2) and an
 *                                   integer address cannot remove a page
 *   partial_beyond_block_refused    a range beyond the block: EINVAL
 *   partial_narrowed_removes_own_page
 *                                   the reference narrowed to the middle page removes that page, and
 *                                   the block lives on
 *   partial_through_moved_reference the block's reference moved to the last page removes it
 *   whole_unmap_with_hole           munmap of the whole block with its holes ends it
 * partial-fixed (exit 0; the refill checks need root grants):
 *   fixed_refill_reads_zero         MAP_FIXED into the hole returns the reference; the page reads zero
 *   removed_page_tags_cleared       a reference stored in the page before its removal is gone
 *   fixed_replaces_mapped_page      MAP_FIXED over a mapped page leaves it reading zero
 *   fixed_refused_with_object       with an object derived below the block's reference: EINVAL
 *   fixed_after_object_released     after that object ends, the refill works again
 * partial-mremap (exit 0):
 *   mremap_shrink_in_place          mremap from three pages to one keeps the reference and an object
 *                                   derived in the first page; the tail is removed
 *   mremap_grow_refused_with_object (root grants) growth into the removed tail while that object
 *                                   exists: ENOMEM, the refill rule of MAP_FIXED (stage 4f)
 *   mremap_grow_into_tail (root grants) after the object ends, without MREMAP_MAYMOVE it grows back
 *                                   in place, page by page (without root grants growth in place is
 *                                   never possible: ENOMEM)
 *   mremap_grow_beyond_enomem       beyond the block's extent without MREMAP_MAYMOVE: ENOMEM
 *   mremap_grow_beyond_moves        with it the range moves and the old block ends
 * Stage 4f, page operations in one step:
 * mremap-validate (exit 0): every path of mremap validates the whole old range first
 *   mremap_shrink_over_hole_refused a shrink of an old range whose first pages were removed: EFAULT,
 *                                   and the block keeps its last page (stage 4e removed it and ended
 *                                   the block, answering success)
 *   mremap_grow_over_hole_refused   a growth whose old range holds a removed page: EFAULT, with and
 *                                   without MREMAP_MAYMOVE
 *   mremap_cursor_outside_refused   the block's reference with its cursor a page below the block:
 *                                   EINVAL for a growth and a shrink
 * pages-enomem (exit 0; vm.max_map_count lowered to the map count, so no VMA can be split):
 *   partial_unmap_enomem_changes_nothing
 *                                   a partial munmap of the middle page answers ENOMEM, and the page
 *                                   keeps its data and takes writes
 *   fixed_enomem_changes_nothing (root grants)
 *                                   MAP_FIXED with PROT_READ over that page answers ENOMEM, and the
 *                                   page keeps its data and stays writable
 *   partial_unmap_after_limit       with the limit restored the partial munmap works
 * pages-tagged (exit 0): a reference stored in the middle page, then its partial munmap; the runner
 *                                   reads the module's census (pinned, root grants: the tags before
 *                                   and after the clearing; with the hooks module's test_unhold_pages
 *                                   the page was unheld and the check pinned it again) and the
 *                                   kernel's count of tagged frames it cleared at their free
 *                                   (captag cleared_pages: unpinned at least one, pinned none)
 *   tagged_page_removed
 * futex-stale (exit 0; root grants):
 *   futex_stale_waiter_not_counted  a waiter on an object that the program then frees, a second
 *                                   waiter on a new object at the same address: a wake of one
 *                                   reaches the second (the first is woken too, uncounted)
 * Hook modes of stage 4f (a hooks launcher; the derive, fault and admit modes need root grants):
 * pages-derive (exit 0; with the hooks module's test_pages_run, ends with a fault 25):
 *   pages_derive_after_refill       a context derives an object over a mapped page while a MAP_FIXED
 *                                   of that page waits between its check and its Linux steps: it runs
 *                                   only after the refill, and no live object sees its memory
 *                                   replaced (the object reads what it read at its derivation)
 *   pages_derive_fails_closed       (test_pages_run) the program runs in the window: the commit finds
 *                                   the object below the block and ends the block; MAP_FIXED answers
 *                                   EINVAL, and the last line reads the object: fault 25
 * pages-fault (exit 0):
 *   pages_fault_waits_then_proceeds a context stores into a removed page while MAP_FIXED refills it;
 *                                   the store lands after the refill (test_pages_run: its fault was
 *                                   deferred, "resolve refused at moving")
 * pages-admit (exit 0):
 *   pages_admit_store_after_refill  a context stores into an untouched page while MAP_FIXED of it
 *                                   waits before its commit; the store lands after the refill, and the
 *                                   runner reads that no page admitted in the window was unpinned at
 *                                   the commit
 * release-fail (ends with a fault 25; the hooks launcher's munmap of a released block fails):
 *   release_failure_keeps_range_unused
 *                                   munmap of a whole block answers 0, the launcher drops it, 16 new
 *                                   mappings stay out of its range, and the old reference faults 25
 * Block A, the follow-ups of stage 4f (hook modes, root grants):
 * pages-closed (exit 0; the hooks module with test_pages_run and test_close_unheld, pinned):
 *   pages_closed_fails_closed       a reference stored in the first page of a block, a MAP_FIXED of its
 *                                   other two pages, and a context that derives an object below the
 *                                   block inside the window: the commit fails closed (EINVAL, the
 *                                   block has ended), and its retire preparation fails (the hook
 *                                   unholds the first page); the runner reads that the module cleared
 *                                   the unheld page in place (census) and that no tagged frame reached
 *                                   Linux (captag cleared_pages unchanged)
 * pages-discard (ends with a fault 25; the hooks launcher's discard of a refill fails part way):
 *   pages_discard_fails_closed      MAP_FIXED of two pages of a block answers EINVAL and the block has
 *                                   ended; the last line reads through the block's reference: fault 25
 * Block A, F4 part 2:
 * pages-unclear (hooks launcher and module, test_close_unheld=2; root grants): as pages-closed, and the
 *   in-place clearing leaves the unheld page as it is:
 *   pages_unclear_extent_held       MAP_FIXED answers EINVAL, the block has ended, its extent stays
 *                                   mapped PROT_NONE (never given back to Linux) and the module counts
 *                                   the commit in closed_unclear
 * top-page (hooks launcher, CAPSTONE_VM_TEST_TOP_PAGE): the launcher's first mapping service grants the
 *   topmost page below 2^38 first, which the launcher and then the module refuse (the runner reads the
 *   launcher's line):
 *   top_page_mapping_below          the mapping that follows works and ends a page or more below 2^38
 * The growth modes of the 63-bit format (they skip in the older formats, which have no failure result
 * and end the context at exhaustion):
 * grow-maps (exit 0):
 *   grow_maps_through_grants        5,000 one-page mmaps (5,000 records, one each, more than the first
 *                                   batch of 4,096 slots) all succeed: the module grows the table before
 *                                   its own CDERIVEs (table_growths rises), and no pressure escape is
 *                                   needed
 * grow-objects (exit 0):
 *   grow_objects_through_escape     6,000 objects of 48 bytes all come from malloc: the table grows,
 *                                   through the pressure escape too (pressure_escapes and table_growths
 *                                   rise), and every object keeps its value
 * nogrow-map (exit 0; module loaded with node_max_pages=16, one batch):
 *   nogrow_map_enomem               mmap answers ENOMEM once the table is full, without a loop (a few
 *                                   escapes at most), and NOGROW is set
 *   nogrow_map_one_record           then malloc takes the records that are left and the free of one
 *                                   object reserved before the fill gives one back: one mmap fits into
 *                                   it and works, and the next one answers ENOMEM (an mmap needs one
 *                                   record, the block's; the F4 part 2 module needs two and fails here)
 *   nogrow_map_recovers             after 300 munmaps an mmap succeeds again
 * nogrow-malloc (exit 0; node_max_pages=16):
 *   nogrow_malloc_null              malloc returns NULL with ENOMEM once the table is full, without a
 *                                   loop, and NOGROW is set
 *   nogrow_malloc_recovers          after half the objects are freed malloc succeeds again
 * realloc-exhausted (exit 0; node_max_pages=16):
 *   realloc_exhausted_keeps_object  a large object (256 KiB), the table then filled by small objects:
 *                                   realloc to 384 KiB (the same granule of 512 bytes, so the F4 part 1
 *                                   libc took its moving path), which needs a larger mapping, answers
 *                                   NULL with ENOMEM and the object intact; then after each step of 64
 *                                   frees it answers NULL with the object intact or, once the frees
 *                                   made room, a new object with the contents, which it must reach; a
 *                                   read of a lost object would fault 25
 * The stage that dropped the reservation record M (blocks of depth 1):
 * partial-renew (ends with a fault 25; root grants, skipped without):
 *   renewed_narrow_keeps_block      the block's reference narrowed to the middle page of three and
 *                                   renewed: it keeps the narrowed bounds; munmap of two pages through
 *                                   it and of the whole block through the reference from before are
 *                                   refused; it removes the middle page, the outer pages stay mapped and
 *                                   the block keeps its range (16 new mappings lie outside it)
 *   renewed_narrow_ends_block       a block of two pages loses its first page, its reference narrowed to
 *                                   the second page is renewed and an object derived in it; the
 *                                   reference from before cannot remove the page, the renewed one can,
 *                                   and that ends the block; the last line reads through the object at
 *                                   the reused address and must fault with 25
 * mremap-move (ends with a fault 25):
 *   mremap_move_part_removes_pages  mremap of the middle page of three to two pages moves it: the new
 *                                   mapping holds its data, the old page is a hole, the block lives on
 *   mremap_move_whole_ends_block    mremap of a whole block of two pages to three moves it and ends the
 *                                   old block: a copy of its reference cannot unmap it; the last line
 *                                   reads through an object derived in it at the reused address and must
 *                                   fault with 25
 * kernel-owned (exit 0): the image (gp) and the main stack (sp):
 *   owned_image_unmap_refused       munmap of the whole image and of its last page: EINVAL
 *   owned_image_mremap_refused      mremap shrinking, growing (MREMAP_MAYMOVE) and to its own size: EINVAL
 *   owned_image_mprotect_refused    mprotect of its last page: EINVAL, the protection unchanged
 *   owned_stack_unmap_refused, owned_stack_mremap_refused, owned_stack_mprotect_refused
 *                                   the same for the main stack (its lowest page)
 *   owned_flag_refuses_each         (root grants and the hooks launcher's CAPSTONE_VM_TEST_OWNED_BYPASS,
 *                                   skipped otherwise) every one of those requests reached the module,
 *                                   whose kernel-owned flag refused it: owned_refused rose at each
 *
 * registered() asks the launcher whether an address lies in a registered mapping, by integer: a
 * futex wake of no waiter there answers 0, outside every mapping EFAULT.
 *
 * Before each stale read the program maps the same address again (ROOT_REMAP same_address=1).
 * An access that the processor lets through then reads the new page and does not fault at all,
 * so a missing CREVOKE shows as a read, not as a page fault. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <capstone/capability.h>
#include <capstone/delegate.h>
#include <capstone/virtual.h>
#include "vm.h"

#define PAGE 4096UL
#define LARGE (1UL << 20)
#define ROUNDS 12000UL
#define STACK (64UL * 1024)
/* The launcher's thread table holds 32 contexts, the first one included. */
#define IDLE_THREADS 31

extern long __capstone_vm_thread_create_frame(void *frame);
extern long __capstone_vm_thread_join(long thread);
extern long __capstone_vm_futex(void *word, int op, int value, long seconds, long nanoseconds,
                                int bitset);
extern _Noreturn void __capstone_vm_thread_exit_clear(void *clear, void *stack, size_t bytes,
                                                      long status);
extern uintptr_t __capstone_context_identity_take(void);
extern void __capstone_context_identity_give(uintptr_t identity);
#ifndef MREMAP_MAYMOVE
#define MREMAP_MAYMOVE 1
#endif

static char *volatile stale;
static int root_grants;

static void fail(const char *what)
{
    fprintf(stderr, "ROOT_FAIL:%s\n", what);
    fflush(stderr);
    exit(1);
}

static void require(int ok, const char *what)
{
    if (!ok) fail(what);
}

static void pass(const char *check)
{
    printf("ROOT_CHECK:%s:PASS\n", check);
    fflush(stdout);
}

/* The reason for nearly every skip: without root grants the launcher resolves the address. */
#define BY_ADDRESS "root_grants=0 names a mapping by its address, so this unmap would succeed"
/* The same for the requests of stage 4c that are no unmap: madvise, futex waits, exit words and
 * transport attachments. */
#define ADDRESS_AUTHORITY "root_grants=0 takes the address as the authority, so this request would be accepted"

static void skip(const char *check, const char *reason)
{
    printf("ROOT_SKIP:%s:%s\n", check, reason);
    fflush(stdout);
}

static unsigned long cursor(void *p)
{
    return __builtin_capstone_cap_get_cursor(p);
}

/* munmap refused the call, with EINVAL. */
static int refused(int rc)
{
    return rc == -1 && errno == EINVAL;
}

static void *untagged(unsigned long address)
{
    void *p;
    __asm__("mv %0, %1" : "=r"(p) : "r"(address));
    return p;
}

/* Does address lie in a mapping the launcher has registered? */
static int registered(unsigned long address)
{
    return __capstone_vm_futex(untagged(address), 1 /* FUTEX_WAKE */, 0, -1, 0, 0) != -EFAULT;
}

static char *map(unsigned long bytes)
{
    char *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(p != MAP_FAILED, "mmap");
    p[0] = 1;
    return p;
}

/* The module's parameter root_grants, read through the delegated calls: 1 (Y), 0 (N), or -1. */
static int module_root_grants(void)
{
    char text[8];
    int fd = open("/sys/module/capstone_vm/parameters/root_grants", O_RDONLY);
    if (fd < 0)
        return -1;
    long n = read(fd, text, sizeof text);
    close(fd);
    return n <= 0 ? -1 : text[0] == 'Y' ? 1 : text[0] == 'N' ? 0 : -1;
}

/* A grant as the adapter issues it, as a copyable reference. The mode (root grants or not) comes from
 * the module's parameter, printed and remembered; the type LCC answers for the grant is printed too. */
static char *grant(void)
{
    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP), "grant");
    int mode = module_root_grants();
    require(mode >= 0, "read the module's root_grants parameter");
    printf("ROOT_GRANT_TYPE:%d\nROOT_GRANT_LCC:%lu\n", mode, sublet_type(&slot));
    fflush(stdout);
    root_grants = mode == 1;
    return cap_vm_copyable(&slot);
}

/* Map the address of the mapping of `bytes` that just ended again, then read through the stale
 * reference. The read must fault with 25 and never print. */
static void stale_read_bytes(unsigned long address, unsigned long bytes)
{
    char *again = map(bytes);
    again[64] = 0x77;
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
}

static void stale_read(unsigned long address)
{
    stale_read_bytes(address, PAGE);
}

static void pause_ms(long ms)
{
    struct timespec t = {ms / 1000, (ms % 1000) * 1000000};
    nanosleep(&t, NULL);
}

static void *thread_pointer(void)
{
    void *tp;
    __asm__ volatile("movc %0, tp" : "=r"(tp));
    return tp;
}

/* The type LCC answers for a reference in a register: 1 non-linear, 7 for an integer. */
static unsigned long type_of(void *p)
{
    unsigned long type;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(type) : "r"(p));
    return type;
}

/* p with its cursor set to address (SCC). Outside what the bounds can represent the result is
 * an integer. */
static void *at_address(void *p, unsigned long address)
{
    void *q;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x05, %0, %1, %2" : "=r"(q) : "r"(p), "r"(address));
    return q;
}

static int mode_unmap(void)
{
    char *page = grant();
    unsigned long address = cursor(page);

    require(page[0] == 0 && page[PAGE - 1] == 0, "a fresh page reads zero");
    memset(page, 0x5a, PAGE);
    require(page[0] == 0x5a && page[PAGE - 1] == 0x5a, "the page keeps what the grant wrote");
    pass("grant_reads_and_writes");

    char *small = malloc(48);
    char *large = malloc(LARGE);
    char *other = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(small && large, "malloc");
    require(other != MAP_FAILED, "second mmap");
    memset(small, 0x11, 48);
    memset(large, 0x22, LARGE);
    memset(other, 0x44, PAGE);
    pass("libc_accepts_copyable_grant");

    char *object = capstone_cap_derive(page, 64, 64);
    object[0] = 0x33;
    require(page[64] == 0x33, "the object lies in the page");
    stale = object;

    require(!munmap(page, PAGE), "munmap through the mapping's reference");
    pass("unmap_through_mapping_reference");
    require(small[47] == 0x11 && large[LARGE - 1] == 0x22 && other[PAGE - 1] == 0x44,
            "the other mappings keep their contents");
    other[0] = 0x45;
    large[0] = 0x23;
    require(other[0] == 0x45 && large[0] == 0x23, "the other mappings take a write");
    pass("other_mapping_survives_unmap");

    require(refused(munmap(page, PAGE)), "a second munmap of the same mapping");
    pass("second_unmap_refused");

    /* The launcher places the next mapping of this size at the address it unmapped last. */
    char *again = map(PAGE);
    printf("ROOT_REMAP same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    again[64] = 0x77;
    if (root_grants) {
        require(refused(munmap(page, PAGE)), "munmap through the old reference at the reused address");
        again[65] = 0x78;
        require(again[64] == 0x77 && again[65] == 0x78, "the new mapping after the refusal");
        pass("stale_reference_at_reused_address_refused");
    } else {
        skip("stale_reference_at_reused_address_refused", BY_ADDRESS);
    }

    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    return 0;
}

static int mode_refuse(void)
{
    char *keep = grant();
    keep[0] = 1;

    if (root_grants) {
        char *page = map(PAGE);
        require(refused(munmap(untagged(cursor(page)), PAGE)), "munmap with an integer address");
        page[1] = 2;
        require(page[0] == 1 && page[1] == 2, "the mapping after the refusal of an address");
        pass("address_refused");

        char *object = capstone_cap_derive(page, 0, PAGE);
        require(refused(munmap(object, PAGE)), "munmap through an object over the whole page");
        object[2] = 3;
        page[3] = 4;
        require(page[2] == 3 && object[3] == 4, "the object and the mapping after the refusal");
        pass("nested_object_refused");
    } else {
        skip("address_refused", BY_ADDRESS);
        skip("nested_object_refused", BY_ADDRESS);
    }

    char *two = map(2 * PAGE);
    two[PAGE] = 2;
    require(refused(munmap(two + PAGE, 2 * PAGE)), "munmap of a range beyond the mapping");
    require(two[0] == 1 && two[PAGE] == 2, "the mapping after the refused munmap");
    require(!munmap(two, 2 * PAGE), "munmap of the whole mapping afterwards");
    pass("unmap_beyond_block_refused");

    if (root_grants) {
        char *page = map(PAGE);
        capstone_cap_release(page);
        require(refused(munmap(page, PAGE)), "munmap through a reference the program ended");
        require(refused(munmap(page, PAGE)), "a second munmap through that reference");
        require(keep[0] == 1, "the other mapping after those refusals");
        pass("stale_reference_refused");

        page = map(PAGE);
        capstone_cap_release(page);
        for (unsigned long i = 0; i < ROUNDS; ++i)
            capstone_cap_release(capstone_cap_derive(keep, 64, 64));
        require(refused(munmap(page, PAGE)), "munmap through an ended reference after a collection");
        keep[1] = 2;
        require(keep[0] == 1 && keep[1] == 2, "the other mapping after the collection");
        pass("stale_reference_after_collection_refused");
    } else {
        skip("stale_reference_refused", BY_ADDRESS);
        skip("stale_reference_after_collection_refused", BY_ADDRESS);
    }
    return 0;
}

static int mode_narrow(void)
{
    (void)grant();
    if (!root_grants) {
        skip("narrowed_reference_refused", BY_ADDRESS);
        skip("whole_reference_after_narrowed", "follows the narrowed case");
        return 0;
    }

    char *two = map(2 * PAGE);
    unsigned long base = cursor(two);
    two[PAGE] = 2;
    char *narrow = __builtin_capstone_cap_shrink(two, base, base + PAGE);
    require(cursor(narrow) == base, "the narrowed reference starts at the mapping");
    require(refused(munmap(narrow, 2 * PAGE)), "munmap through a narrowed reference");
    two[PAGE + 1] = 3;
    require(two[0] == 1 && two[PAGE] == 2 && two[PAGE + 1] == 3, "the mapping after the refusal");
    pass("narrowed_reference_refused");

    require(!munmap(two, 2 * PAGE), "munmap through the whole reference afterwards");
    pass("whole_reference_after_narrowed");
    return 0;
}

/* through_object: the last read goes through an object derived in the renewed mapping, else
 * through the renewed reference itself. */
static int mode_renew(int through_object)
{
    (void)grant();
    char *page = map(PAGE);
    unsigned long address = cursor(page);
    char *renewed = capstone_cap_renew(page);

    if (root_grants) {
        require(refused(munmap(page, PAGE)), "munmap through the reference from before CRENEW");
        renewed[1] = 0x56;
        require(renewed[0] == 1 && renewed[1] == 0x56, "the renewed mapping after the refusal");
        pass("renewed_old_reference_refused");
    } else {
        skip("renewed_old_reference_refused", BY_ADDRESS);
    }

    char *object = capstone_cap_derive(renewed, 64, 64);
    object[0] = 0x55;
    require(renewed[64] == 0x55, "the object lies in the renewed mapping");
    stale = through_object ? object : renewed;
    require(!munmap(renewed, PAGE), "munmap through the renewed reference");
    pass("renewed_mapping_unmaps");
    stale_read(address);
    return 0;
}

static int protected_as(unsigned long address, const char *rights);

/* mremap: the libc's mremap asks the mapping check, then maps a new block, copies the
 * contents with their tags and unmaps the old block through the program's reference
 * (mapping.c). The stored reference at offset 64 names another mapping. */
static int mode_mremap(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *p = map(2 * PAGE);
    unsigned long address = cursor(p);
    memset(p, 0x61, 2 * PAGE);
    *(char **)(p + 64) = target;

    errno = 0;
    require(mremap(p + PAGE, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
            "mremap of a range that reaches beyond the mapping");
    p[2] = 0x62;
    require(p[0] == 0x61 && p[2] == 0x62 && p[2 * PAGE - 1] == 0x61 &&
            (*(char **)(p + 64))[1] == 0x71, "the old mapping after the refused mremap");
    pass("mremap_beyond_block_refused");

    printf("ROOT_MREMAP case=oversized\n");
    fflush(stdout);
    errno = 0;
    require(mremap(p, 3 * PAGE, 4 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
            "mremap with an old length beyond the mapping");
    p[4] = 0x65;
    require(p[0] == 0x61 && p[4] == 0x65 && p[2 * PAGE - 1] == 0x61,
            "the old mapping after the oversized mremap");
    pass("mremap_oversized_old_refused");

    if (root_grants) {
        char *object = capstone_cap_derive(p, 0, 2 * PAGE);
        errno = 0;
        require(mremap(object, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
                "mremap through an object over the whole mapping");
        object[3] = 0x63;
        require(p[3] == 0x63 && p[2] == 0x62 && p[2 * PAGE - 1] == 0x61,
                "the mapping and the object after the refused mremap");
        pass("mremap_nested_object_refused");
    } else {
        skip("mremap_nested_object_refused", BY_ADDRESS);
    }

    char *same = mremap(p, 2 * PAGE, 2 * PAGE - 100, MREMAP_MAYMOVE);
    require(same != MAP_FAILED && cursor(same) == address && type_of(same) == 1 && same[4] == 0x65,
            "mremap to the same page count returns the reference");
    pass("mremap_same_size_returns_reference");

    if (root_grants) {
        char *s = map(PAGE);
        s[1] = 0x66;
        char *renewed = capstone_cap_renew(s);
        errno = 0;
        require(mremap(s, PAGE, PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EINVAL,
                "mremap to the same size through the reference from before CRENEW");
        char *kept = mremap(renewed, PAGE, PAGE, MREMAP_MAYMOVE);
        require(kept != MAP_FAILED && cursor(kept) == cursor(renewed) && renewed[1] == 0x66,
                "mremap to the same size through the renewed reference");
        pass("mremap_same_size_stale_refused");
    } else {
        skip("mremap_same_size_stale_refused", BY_ADDRESS);
    }

    require(!mprotect(p, 2 * PAGE, PROT_READ), "make the mapping read-only");
    errno = 0;
    require(mremap(p, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == ENOTSUP,
            "mremap of a read-only mapping");
    require(protected_as(address, "r--") && protected_as(address + PAGE, "r--") &&
            p[0] == 0x61 && p[4] == 0x65, "the read-only mapping after the refused mremap");
    require(!mprotect(p, 2 * PAGE, PROT_READ | PROT_WRITE), "restore write rights");
    pass("mremap_read_only_refused");

    char *q = mremap(p, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE);
    require(q != MAP_FAILED && cursor(q) != address, "mremap to three pages moves the mapping");
    char *moved = *(char **)(q + 64);
    printf("ROOT_MREMAP grown moved_type=%lu\n", type_of(moved));
    fflush(stdout);
    require(q[0] == 0x61 && q[2] == 0x62 && q[2 * PAGE - 1] == 0x61,
            "the grown mapping keeps the data");
    require(type_of(moved) == 1 && cursor(moved) == cursor(target) && moved[1] == 0x71,
            "the grown mapping keeps the stored reference");
    require(q[2 * PAGE] == 0 && q[3 * PAGE - 1] == 0, "the third page reads zero");
    q[3 * PAGE - 1] = 0x64;
    require(q[3 * PAGE - 1] == 0x64, "the third page takes a write");
    pass("public_mremap_grows");

    require(refused(munmap(p, 2 * PAGE)), "munmap through the old reference after mremap");
    pass("mremap_old_reference_refused");

    unsigned long grown = cursor(q);
    char *r = mremap(q, 3 * PAGE, PAGE, MREMAP_MAYMOVE);
    require(r != MAP_FAILED && cursor(r) == grown && type_of(r) == 1,
            "mremap to one page shrinks the mapping in place (stage 4e)");
    moved = *(char **)(r + 64);
    require(q[0] == 0x61 && q[2] == 0x62 && q[PAGE - 1] == 0x61,
            "the shrunk mapping keeps the data of its page");
    require(type_of(moved) == 1 && moved[1] == 0x71, "the shrunk mapping keeps the stored reference");
    pass("public_mremap_shrinks");

    /* The last page that is still mapped goes: the mapping ends, and q with it. */
    require(!munmap(q, PAGE), "munmap of the last page of the shrunk mapping");
    stale = q;
    stale_read_bytes(grown, 3 * PAGE);
    return 0;
}

/* ---- Threads ----
 *
 * The thread cases use raw contexts (__capstone_virtual_context_create, as the pthread contract
 * does) that share the main thread's TLS and touch none of it, so each case controls exactly
 * what its exit service names. */

/* exit-victim: the exit service's stack operand and its length, chosen by the main thread. */
static void *volatile exit_operand;
static volatile unsigned long exit_bytes;

static void *exit_naming_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear(NULL, exit_operand, exit_bytes, 0);
}

/* One raw context on its own stack mapping (own, own_bytes) that exits naming exit_operand;
 * joined. */
static void exit_naming(void *operand, unsigned long bytes, char *own, unsigned long own_bytes)
{
    exit_operand = operand;
    exit_bytes = bytes;
    long id = __capstone_virtual_context_create(exit_naming_worker, NULL, own + own_bytes,
                                                thread_pointer());
    require(id > 0, "create the exiting context");
    require(!capstone_virtual_thread_join(id), "join the exiting context");
}

static int mode_exit_victim(void)
{
    (void)grant();
    if (!root_grants) {
        skip("exit_with_own_stack_at_victim_refused", BY_ADDRESS);
        skip("exit_with_victim_address_refused", BY_ADDRESS);
        skip("victim_unmaps_afterwards", "follows the refused cases");
        return 0;
    }

    /* Two heap grants in a row: the launcher places each one right below the one before where
     * that range is free, so the thread's stack ends where the victim begins, and the stack's
     * reference with its cursor at the victim's address stays a reference (a cursor at the end
     * is inside what the bounds represent). A pair that did not land so is left mapped and the
     * next pair tried. */
    char *victim = NULL, *own = NULL;
    unsigned long address = 0;
    unsigned tries = 0;
    for (; tries < 8; ++tries) {
        sublet_cap slot;
        victim = grant();
        address = cursor(victim);
        require(!cap_vm_acquire(&slot, 2 * PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
                "the thread's stack");
        own = cap_vm_copyable(&slot);
        if (cursor(own) + 2 * PAGE == address) break;
    }
    victim[0] = 1;
    void *moved = at_address(own, address);
    printf("ROOT_OPERAND own_stack_at_victim type=%lu adjacent=%d tries=%u\n", type_of(moved),
           cursor(own) + 2 * PAGE == address, tries + 1);
    fflush(stdout);
    exit_naming(moved, PAGE, own, 2 * PAGE);
    victim[1] = 2;
    require(victim[0] == 1 && victim[1] == 2, "the victim after an exit with the own stack's reference");
    pass("exit_with_own_stack_at_victim_refused");

    char *other = map(STACK);
    exit_naming(untagged(address), PAGE, other, STACK);
    victim[2] = 3;
    require(victim[0] == 1 && victim[2] == 3, "the victim after an exit with its address");
    pass("exit_with_victim_address_refused");

    require(!munmap(victim, PAGE), "munmap of the victim through its own reference");
    pass("victim_unmaps_afterwards");
    return 0;
}

/* exit-renew: the worker renews the reference to its own stack's mapping and exits naming the
 * renewed reference, of which the main thread keeps a copy. From the CRENEW on the worker's sp
 * names the old lifetime and is stale, so everything after it is one asm block that touches no
 * stack. */
static void *volatile exit_stack;
static void *volatile kept;

static void *renew_exit_worker(void *arg)
{
    (void)arg;
    __asm__ volatile(".insn i 0x5b, 0x3, t0, 0(%[from])\n"       /* LDC t0, the stack's mapping */
                     ".insn r 0x5b, 0x1, 0x54, a1, t0, x0\n"     /* CRENEW a1, t0 */
                     ".insn s 0x5b, 0x4, a1, 0(%[keep])\n"       /* STC a1, the main thread's copy */
                     "li a0, 0\n"
                     "mv a2, %[bytes]\n"
                     "li a3, 0\n"
                     "li a7, %[service]\n"
                     "ecall\n"
                     "ebreak\n"
                     :
                     : [from] "r"(&exit_stack), [keep] "r"(&kept), [bytes] "r"(STACK),
                       [service] "i"(CV_SERVICE_PTHREAD_EXIT)
                     : "t0", "a0", "a1", "a2", "a3", "a7", "memory");
    __builtin_unreachable();
}

static int mode_exit_renew(void)
{
    (void)grant();
    char *stack = map(STACK);
    unsigned long address = cursor(stack);

    exit_stack = stack;
    long id = __capstone_virtual_context_create(renew_exit_worker, NULL, stack + STACK,
                                                thread_pointer());
    require(id > 0, "create the worker");
    require(!capstone_virtual_thread_join(id), "join the worker");
    char *renewed = kept;
    require(type_of(renewed) == 1 && cursor(renewed) == address,
            "the main thread holds the worker's renewed stack reference");
    pass("stack_renewed_before_exit");
    stale = renewed;
    stale_read_bytes(address, STACK);
    return 0;
}

/* create and create-fail: a start frame as thread.c builds it, on its own one-page mapping. */
static volatile int child_ran;

static void *frame_child(void *arg)
{
    __atomic_store_n(&child_ran, 1, __ATOMIC_RELEASE);
    return arg;
}

static void *frame_return(void *unused)
{
    (void)unused;
    capstone_virtual_thread_exit(NULL);
}

static unsigned char *frame_fill(unsigned char *frame, void *(*entry)(void *), void *stack_top,
                                 uintptr_t *identity);

static unsigned char *frame_new(void *(*entry)(void *), void *stack_top, uintptr_t *identity)
{
    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
            "the frame's grant");
    return frame_fill(cap_vm_copyable(&slot), entry, stack_top, identity);
}

/* The page frame as a start frame for entry on stack_top, with a new context identity. */
static unsigned char *frame_fill(unsigned char *frame, void *(*entry)(void *), void *stack_top,
                                 uintptr_t *identity)
{
    memset(frame, 0, PAGE);
    ((uint64_t *)frame)[7] = 3;
    *identity = __capstone_context_identity_take();
    require(*identity != 0, "an identity for the context");
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16), (void *)frame_return);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack_top);
    __asm__ volatile("scc t0, gp, %1\n\tstc t0, 0(%0)"
                     : : "r"(frame + 64 + 3 * 16), "r"(*identity) : "t0", "memory");
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 4 * 16), thread_pointer());
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 10 * 16), NULL);
    capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)entry);
    return frame;
}

static int mode_create(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    unsigned char *frame = frame_new(frame_child, stack + STACK, &identity);
    unsigned char *renewed = capstone_cap_renew(frame);

    if (root_grants) {
        long id = __capstone_vm_thread_create_frame(frame);
        printf("ROOT_CREATE stale=%ld\n", id);
        fflush(stdout);
        require(id == -EINVAL, "thread create with the frame's reference from before CRENEW");
        renewed[PAGE - 1] = 0x5a;
        require(renewed[PAGE - 1] == 0x5a && ((uint64_t *)renewed)[7] == 3,
                "the renewed frame after the refused create");
        pass("create_with_stale_frame_refused");
    } else {
        skip("create_with_stale_frame_refused", BY_ADDRESS);
    }

    long id = __capstone_vm_thread_create_frame(renewed);
    printf("ROOT_CREATE renewed=%ld\n", id);
    fflush(stdout);
    require(id > 0, "thread create with the renewed frame reference");
    pass("create_with_renewed_frame");
    stale = (char *)renewed;
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
    capstone_virtual_thread_join(id);
    return 0;
}

static volatile int idle_done;

static void *idle_entry(void *arg)
{
    __atomic_fetch_add(&idle_done, 1, __ATOMIC_RELEASE);
    return arg;
}

/* Contexts that wait until the main thread releases them (release_blocked), on a futex of the
 * launcher's: the futex service is not a delegated call, so it needs no transport. */
static volatile int release_word;

static void *blocked_entry(void *arg)
{
    while (!__atomic_load_n(&release_word, __ATOMIC_ACQUIRE))
        __capstone_vm_futex((void *)&release_word, 0 /* FUTEX_WAIT */, 0, -1, 0, 0);
    return arg;
}

static void release_blocked(void)
{
    __atomic_store_n(&release_word, 1, __ATOMIC_RELEASE);
    __capstone_vm_futex((void *)&release_word, 1 /* FUTEX_WAKE */, 1 << 30, -1, 0, 0);
}

/* Wait until *flag reaches value, for at most about ms milliseconds. */
static int wait_for(volatile int *flag, int value, long ms)
{
    for (long waited = 0; __atomic_load_n(flag, __ATOMIC_ACQUIRE) < value; waited += 10) {
        if (waited >= ms)
            return 0;
        pause_ms(10);
    }
    return 1;
}

static int mode_create_fail(void)
{
    (void)grant();
    char *stacks = map((IDLE_THREADS + 2) * STACK);
    long ids[IDLE_THREADS + 1];
    unsigned live = 0;
    unsigned char *frame;
    uintptr_t identity;
    long id;

    /* Blocked contexts until a create fails. Each exit frees its entry, so only live contexts
     * fill the table. */
    for (;;) {
        frame = frame_new(blocked_entry, stacks + (live + 1) * STACK, &identity);
        id = __capstone_vm_thread_create_frame(frame);
        if (id <= 0)
            break;
        require(live < IDLE_THREADS + 1, "a create fails once the thread table is full");
        ids[live++] = id;
    }
    __capstone_context_identity_give(identity);
    int probe = registered(cursor(frame));
    printf("ROOT_CREATE_FAIL live=%u result=%ld registered=%d\n", live, id, probe);
    fflush(stdout);
    if (id == -EAGAIN) {
        require(!probe, "the frame's mapping is no longer registered");
        pass("failed_create_releases_frame");
    } else if (id == -ENOSPC) {
        require(probe && !munmap(frame, PAGE), "the frame stays the program's and unmaps");
        pass("module_limit_keeps_frame");
    } else {
        fail("the create fails with EAGAIN or ENOSPC");
    }

    /* The failed create issued no id: the module gives back the id of a create that the launcher
     * rolls back, and refuses before it issues one. */
    long rc = __capstone_vm_thread_join(ids[live - 1] + 1);
    printf("ROOT_CREATE_FAIL join_next=%ld\n", rc);
    fflush(stdout);
    require(rc == -ESRCH, "a join of the failed create's id");
    pass("failed_create_id_refused");

    release_blocked();
    for (unsigned i = 0; i < live; ++i)
        require(!capstone_virtual_thread_join(ids[i]), "join a released context");
    return 0;
}

/* join-two: two contexts join one context. The launcher keeps the joined context's table entry
 * while joiners wait, and the last joiner to leave frees it. */
static volatile long join_target;
static volatile long join_result[2] = {1, 1};
static volatile int joins_returned;

static void *joiner_entry(void *arg)
{
    long rc = __capstone_vm_thread_join(join_target);
    unsigned k = __atomic_fetch_add(&joins_returned, 1, __ATOMIC_ACQ_REL);
    if (k < 2)
        join_result[k] = rc;
    return arg;
}

static long context(void *(*entry)(void *), char *stack_top)
{
    long id = __capstone_virtual_context_create(entry, NULL, stack_top, thread_pointer());
    require(id > 0, "create a context");
    return id;
}

static int mode_join_two(void)
{
    (void)grant();
    char *stacks = map(4 * STACK);

    long target = context(blocked_entry, stacks + STACK);
    join_target = target;
    long first = context(joiner_entry, stacks + 2 * STACK);
    long second = context(joiner_entry, stacks + 3 * STACK);
    /* Both joiners reach the launcher's join long before this pause ends. */
    pause_ms(500);
    require(!__atomic_load_n(&joins_returned, __ATOMIC_ACQUIRE), "no join returns before the exit");
    release_blocked();
    int both = wait_for(&joins_returned, 2, 5000);
    printf("ROOT_JOIN_TWO returned=%d first=%ld second=%ld\n", joins_returned, join_result[0],
           join_result[1]);
    fflush(stdout);
    require(both, "both joiners return");
    require(join_result[0] == 0 && join_result[1] == 0, "both joins answer 0");
    require(!capstone_virtual_thread_join(first) && !capstone_virtual_thread_join(second),
            "join the joiners");
    pass("two_joiners_return");

    idle_done = 0;
    long next = context(idle_entry, stacks + STACK);
    require(!capstone_virtual_thread_join(next) && idle_done == 1, "a new context after the joins");
    pass("thread_after_two_joiners");
    return 0;
}

/* join-after: joins that arrive after the exit. */
static volatile int exit_word;

static void *exit_word_worker(void *arg);

static int mode_join_after(void)
{
    (void)grant();
    char *stacks = map(3 * STACK);

    exit_operand = NULL;
    exit_bytes = 0;
    exit_word = 1;
    long worded = context(exit_word_worker, stacks + STACK);
    for (unsigned spins = 0; __atomic_load_n(&exit_word, __ATOMIC_ACQUIRE); ++spins) {
        require(spins < 500, "the launcher writes the exit word");
        pause_ms(10);
    }
    pause_ms(200);
    long rc = capstone_virtual_thread_join(worded) ? -errno : 0;
    printf("ROOT_JOIN_AFTER worded=%ld\n", rc);
    fflush(stdout);
    require(rc == 0, "a join after an exit with an exit word");
    pass("join_after_word_exit");

    idle_done = 0;
    long raw = context(idle_entry, stacks + 2 * STACK);
    require(wait_for(&idle_done, 1, 5000), "the context runs");
    pause_ms(200);
    rc = capstone_virtual_thread_join(raw) ? -errno : 0;
    printf("ROOT_JOIN_AFTER raw=%ld\n", rc);
    fflush(stdout);
    require(rc == 0, "a join after an exit without an exit word");
    pass("join_after_raw_exit");

    require(!capstone_virtual_thread_join(worded) && !capstone_virtual_thread_join(raw),
            "a second join of both");
    rc = __capstone_vm_thread_join(raw + 1000);
    printf("ROOT_JOIN_AFTER unknown=%ld\n", rc);
    fflush(stdout);
    require(rc == -ESRCH, "a join of an id never issued");
    pass("join_unknown_id_refused");
    return 0;
}

/* exit-many: more contexts than the launcher's table holds exit without an exit word, one after
 * the other, and none is joined. */
#define MANY_THREADS 40

static int mode_exit_many(void)
{
    (void)grant();
    char *stacks = map((MANY_THREADS + 1) * STACK);
    long first = 0;

    idle_done = 0;
    for (unsigned i = 0; i < MANY_THREADS; ++i) {
        long id = __capstone_virtual_context_create(idle_entry, NULL, stacks + (i + 1) * STACK,
                                                    thread_pointer());
        if (id <= 0) {
            printf("ROOT_EXIT_MANY failed_at=%u errno=%d\n", i, errno);
            fflush(stdout);
            fail("create a context while earlier ones exited unjoined");
        }
        if (!first)
            first = id;
        require(wait_for(&idle_done, (int)i + 1, 5000), "the context runs");
        pause_ms(5);
    }
    long last = context(idle_entry, stacks + (MANY_THREADS + 1) * STACK);
    require(!capstone_virtual_thread_join(last), "join the context after the unjoined ones");
    require(!capstone_virtual_thread_join(first), "a late join of the first unjoined context");
    printf("ROOT_EXIT_MANY contexts=%d\n", MANY_THREADS + 1);
    fflush(stdout);
    pass("exit_unjoined_many_then_create");
    return 0;
}

/* detached: a pthread's stack mapping is registered while it runs and gone after its exit, which
 * unmaps it through musl's __unmapself (pthread.c) with the mapping's reference. */
static volatile unsigned long detached_page;
static volatile int detached_ready, detached_go;

static void *detached_worker(void *arg)
{
    pthread_attr_t attr;
    void *base = NULL;
    size_t bytes = 0;
    if (!pthread_getattr_np(pthread_self(), &attr) && !pthread_attr_getstack(&attr, &base, &bytes))
        detached_page = (cursor(base) + bytes / 2) & ~(PAGE - 1);
    __atomic_store_n(&detached_ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&detached_go, __ATOMIC_ACQUIRE)) pause_ms(5);
    return arg;
}

static int stack_probe(void)
{
    return registered(detached_page) ? 0 : EINVAL;
}

static int mode_detached(void)
{
    (void)grant();
    pthread_attr_t attr;
    pthread_t thread;
    require(!pthread_attr_init(&attr) &&
            !pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED), "detached attribute");
    require(!pthread_create(&thread, &attr, detached_worker, NULL), "pthread_create");
    for (unsigned spins = 0; !__atomic_load_n(&detached_ready, __ATOMIC_ACQUIRE); ++spins) {
        require(spins < 1000, "the detached thread starts");
        pause_ms(10);
    }
    require(detached_page, "the detached thread's stack");
    int before = stack_probe();
    __atomic_store_n(&detached_go, 1, __ATOMIC_RELEASE);
    int after = 0;
    for (unsigned spins = 0; spins < 200 && !after; ++spins) {
        pause_ms(20);
        after = stack_probe();
    }
    printf("ROOT_DETACHED page=0x%lx before=%d after=%d\n", detached_page, before, after);
    fflush(stdout);
    require(before == 0, "the stack's mapping is registered while the thread runs");
    require(after == EINVAL, "the stack's mapping is gone after the thread's exit");
    pass("detached_exit_unmaps_stack");
    return 0;
}

/* exit-length and exit-huge: a thread exits naming its own stack's mapping with a length that is
 * not the mapping's, STACK - PAGE or 2^64 - 1. The exit completes and writes the exit word; the
 * mapping stays until the program unmaps it through the reference it kept. The exit word lies
 * outside the stack, so a launcher that checks it against the base and the length the program
 * names (and not against the mapping it found) stops at the huge length. exit_word is defined
 * with join-after. */
static void *exit_word_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear((void *)&exit_word, exit_operand, exit_bytes, 0);
}

static int mode_exit_length(unsigned long bytes, const char *tag)
{
    char name[80];
    (void)grant();
    char *stack = map(STACK);
    unsigned long address = cursor(stack);

    printf("ROOT_EXIT_WORD in_stack=%d length=0x%lx\n",
           cursor((void *)&exit_word) - address < STACK, bytes);
    fflush(stdout);
    require(cursor((void *)&exit_word) - address >= STACK, "the exit word lies outside the stack");
    exit_operand = stack;
    exit_bytes = bytes;
    exit_word = 1;
    long id = __capstone_virtual_context_create(exit_word_worker, NULL, stack + STACK,
                                                thread_pointer());
    require(id > 0, "create the exiting context");
    require(!capstone_virtual_thread_join(id), "join the context that named a wrong length");
    require(exit_word == 0, "the launcher wrote the exit word");
    snprintf(name, sizeof name, "exit_with_%s_length_completes", tag);
    pass(name);

    require(registered(address), "the stack's mapping is still registered");
    stack[0] = 0x31;
    stack[STACK - 1] = 0x32;
    require(stack[0] == 0x31 && stack[STACK - 1] == 0x32,
            "the stack's mapping works through the kept reference");
    snprintf(name, sizeof name, "%s_length_stack_stays_mapped", tag);
    pass(name);

    require(!munmap(stack, STACK), "munmap of the stack through the kept reference");
    require(!registered(address), "the stack's mapping is gone after the munmap");
    snprintf(name, sizeof name, "%s_length_stack_unmaps_afterwards", tag);
    pass(name);
    return 0;
}

/* exit-frame: a thread exits naming its own frame's mapping as its stack, through the frame's
 * reference from before the create (stale since the hand-over). The module refuses the stack
 * because it is the thread's frame and releases the arena once, as the frame. */
static void *volatile frame_operand;

static void *frame_as_stack_child(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear((void *)&exit_word, frame_operand, PAGE, 0);
}

static int mode_exit_frame(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    unsigned char *frame = frame_new(frame_as_stack_child, stack + STACK, &identity);
    unsigned long address = cursor(frame);

    frame_operand = frame;
    exit_word = 1;
    long id = __capstone_vm_thread_create_frame(frame);
    require(id > 0, "create the thread on its frame");
    require(!capstone_virtual_thread_join(id), "join the thread that named its frame as its stack");
    require(exit_word == 0, "the launcher wrote the exit word");
    pass("exit_with_frame_as_stack_completes");

    int probe = registered(address);
    printf("ROOT_FRAME_PROBE registered=%d\n", probe);
    fflush(stdout);
    require(!probe, "the frame's mapping is released");
    stack[0] = 0x33;
    require(stack[0] == 0x33 && !munmap(stack, STACK), "the thread's stack afterwards");
    pass("frame_as_stack_released_once");
    return 0;
}

/* The protection of the page at address as /proc/self/maps shows it ("rw-", "r--", "---"), or
 * "?" where no line holds it. The launcher's process is the program's address space. */
static void protection(unsigned long address, char out[4])
{
    char line[256];
    FILE *maps = fopen("/proc/self/maps", "r");

    strcpy(out, "?");
    if (!maps)
        return;
    while (fgets(line, sizeof line, maps)) {
        unsigned long low, high;
        char rights[8];
        if (sscanf(line, "%lx-%lx %7s", &low, &high, rights) == 3 && low <= address &&
            address < high) {
            memcpy(out, rights, 3);
            out[3] = 0;
            break;
        }
    }
    fclose(maps);
}

/* Is the page at address protected as rights says? Printed for the record. */
static int protected_as(unsigned long address, const char *rights)
{
    char seen[4];
    protection(address, seen);
    printf("ROOT_PROTECTION 0x%lx %s\n", address, seen);
    fflush(stdout);
    return !strcmp(seen, rights);
}

static int mode_protect(void)
{
    (void)grant();
    char *two = map(2 * PAGE);
    unsigned long base = cursor(two);
    two[PAGE] = 2;

    require(!mprotect(two, 2 * PAGE, PROT_READ), "mprotect through the mapping's reference");
    require(protected_as(base, "r--") && protected_as(base + PAGE, "r--"),
            "the whole mapping is read-only");
    require(two[0] == 1 && two[PAGE] == 2, "the read-only mapping reads");
    require(!mprotect(two, 2 * PAGE, PROT_READ | PROT_WRITE), "restore write rights");
    require(!mprotect(two + PAGE, PAGE, PROT_NONE), "mprotect of the second page");
    require(protected_as(base, "rw-") && protected_as(base + PAGE, "---"),
            "only the second page is inaccessible");
    require(!mprotect(two + PAGE, PAGE, PROT_READ | PROT_WRITE), "restore the second page");
    two[1] = 3;
    two[PAGE + 1] = 4;
    require(two[1] == 3 && two[PAGE + 1] == 4, "the mapping after the restores");
    pass("protect_through_mapping_reference");

    if (root_grants) {
        char *page = map(PAGE);
        unsigned long address = cursor(page);
        require(refused(mprotect(untagged(address), PAGE, PROT_NONE)),
                "mprotect with an integer address");
        require(protected_as(address, "rw-"), "the protection after the refusal of an address");
        page[1] = 5;
        require(page[0] == 1 && page[1] == 5, "the mapping after the refusal of an address");
        pass("protect_integer_address_refused");

        char *renewed = capstone_cap_renew(page);
        require(refused(mprotect(page, PAGE, PROT_NONE)),
                "mprotect through the reference from before CRENEW");
        require(protected_as(address, "rw-"), "the protection after the refusal of a stale reference");
        renewed[2] = 6;
        require(renewed[1] == 5 && renewed[2] == 6, "the renewed mapping after the refusal");
        pass("protect_stale_reference_refused");

        char *narrow = __builtin_capstone_cap_shrink(two, base, base + PAGE);
        require(refused(mprotect(narrow, 2 * PAGE, PROT_NONE)),
                "mprotect of two pages through a reference narrowed to one");
        require(protected_as(base, "rw-") && protected_as(base + PAGE, "rw-"),
                "the protection after the refusal of a narrowed reference");
        pass("protect_narrowed_reference_refused");

        char *object = capstone_cap_derive(renewed, 0, PAGE);
        require(refused(mprotect(object, PAGE, PROT_NONE)),
                "mprotect through an object over the whole page");
        require(protected_as(address, "rw-"), "the protection after the refusal of an object");
        object[3] = 7;
        require(renewed[3] == 7, "the object and the mapping after the refusal");
        pass("protect_nested_object_refused");
    } else {
        skip("protect_integer_address_refused", BY_ADDRESS);
        skip("protect_stale_reference_refused", BY_ADDRESS);
        skip("protect_narrowed_reference_refused", BY_ADDRESS);
        skip("protect_nested_object_refused", BY_ADDRESS);
    }

    char *first = __builtin_capstone_cap_shrink(two, base, base + PAGE);
    require(!mprotect(first, PAGE, PROT_READ), "mprotect of the page a narrowed reference covers");
    require(protected_as(base, "r--") && protected_as(base + PAGE, "rw-"),
            "only the covered page is read-only");
    require(!mprotect(two, PAGE, PROT_READ | PROT_WRITE), "restore the first page");
    pass("protect_narrowed_covering_range");
    return 0;
}

/* protect-held: a pthread's exchange region is a mapping of its own that the launcher holds
 * while the thread is attached, since the launcher writes it through Linux. mprotect of it through
 * the thread's own reference is refused with EBUSY. After the thread detaches its transport, the
 * same mprotect succeeds. The worker restores each protection it changed before its next
 * delegated call writes the region again. */
extern void *__capstone_virtual_exchange(void);
extern long __capstone_vm_transport_detach(void);
static int held_errno = -1, detached_errno = -1;

static void *held_worker(void *arg)
{
    printf("ROOT_HELD_WORKER attached\n");      /* a delegated call: the worker attaches */
    fflush(stdout);
    char *exchange = __capstone_virtual_exchange();
    if (!exchange)
        return arg;
    held_errno = mprotect(exchange, PAGE, PROT_READ) ? errno : 0;
    if (!held_errno)
        mprotect(exchange, PAGE, PROT_READ | PROT_WRITE);
    __capstone_vm_transport_detach();
    detached_errno = mprotect(exchange, PAGE, PROT_READ) ? errno : 0;
    if (!detached_errno)
        mprotect(exchange, PAGE, PROT_READ | PROT_WRITE);
    return arg;
}

static int mode_protect_held(void)
{
    (void)grant();
    pthread_t thread;
    require(!pthread_create(&thread, NULL, held_worker, NULL), "pthread_create");
    require(!pthread_join(thread, NULL), "pthread_join");
    printf("ROOT_HELD held_errno=%d detached_errno=%d\n", held_errno, detached_errno);
    fflush(stdout);
    require(held_errno == EBUSY, "mprotect of a held mapping");
    pass("protect_held_mapping_refused");
    require(detached_errno == 0, "mprotect of the same mapping after the detach");
    pass("protect_after_detach");
    return 0;
}

/* exit-word: exits naming an exit word that the launcher must not write. Each context runs on
 * its own stack and exits through the exit service with the word, the stack operand and its
 * length set here; the main thread joins it. */
static void *volatile word_ref;
static void *volatile word_stack;
static volatile unsigned long word_stack_bytes;
static volatile int words[2];

static void *word_exit_worker(void *arg)
{
    (void)arg;
    __capstone_vm_thread_exit_clear(word_ref, word_stack, word_stack_bytes, 0);
}

static void exit_with_word(void *word, void *stack, unsigned long bytes, char *own_top)
{
    word_ref = word;
    word_stack = stack;
    word_stack_bytes = bytes;
    long id = context(word_exit_worker, own_top);
    require(!capstone_virtual_thread_join(id), "join the context");
}

static int mode_exit_word(void)
{
    (void)grant();
    char *own = map(STACK);

    words[0] = words[1] = 0x11111111;
    printf("ROOT_EXIT_WORD case=unaligned word=0x%lx\n", cursor((void *)words) + 1);
    fflush(stdout);
    exit_with_word((char *)words + 1, NULL, 0, own + STACK);
    require(words[0] == 0x11111111 && words[1] == 0x11111111, "the unaligned word keeps its bytes");
    pass("unaligned_exit_word_not_written");

    /* Two heap grants in a row: the launcher places each one right below the one before where
     * that range is free, so the second (low) ends where the first (the thread's stack) begins.
     * A pair that did not land so is left mapped and the next pair tried. */
    char *stack = NULL, *low = NULL;
    unsigned tries = 0;
    for (; tries < 8; ++tries) {
        sublet_cap slot;
        require(!cap_vm_acquire(&slot, STACK, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
                "the thread's stack");
        stack = cap_vm_copyable(&slot);
        require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
                "the mapping below the stack");
        low = cap_vm_copyable(&slot);
        if (cursor(low) + PAGE == cursor(stack)) break;
    }
    unsigned long stack_address = cursor(stack);
    printf("ROOT_STRADDLE adjacent=%d tries=%u\n", cursor(low) + PAGE == stack_address, tries + 1);
    fflush(stdout);
    low[PAGE - 2] = 0x5a;
    low[PAGE - 1] = 0x5b;
    exit_with_word(low + PAGE - 2, stack, STACK, stack + STACK);
    require(low[PAGE - 2] == 0x5a && low[PAGE - 1] == 0x5b, "the straddling word keeps its bytes");
    require(!registered(stack_address), "the stack is released");
    pass("straddling_exit_word_not_written");

    char *page = map(PAGE);
    *(volatile int *)(page + 64) = 0x2222;
    require(!mprotect(page, PAGE, PROT_READ), "make the word's page read-only");
    printf("ROOT_EXIT_WORD case=read-only\n");
    fflush(stdout);
    exit_with_word(page + 64, NULL, 0, own + STACK);
    require(*(volatile int *)(page + 64) == 0x2222, "the read-only word keeps its value");
    pass("read_only_exit_word_not_written");
    require(!mprotect(page, PAGE, PROT_READ | PROT_WRITE), "restore write rights");
    exit_with_word(page + 64, NULL, 0, own + STACK);
    require(*(volatile int *)(page + 64) == 0, "the launcher writes the word again");
    pass("writable_exit_word_written");

    /* Stage 4e: a word inside the stack that the exit releases no longer keeps the stack; the
     * store is dropped without an error (CAPSTONE_VM_EXIT_WORD_DROPPED), as Linux drops a
     * clear_child_tid store into a stack that is gone. */
    sublet_cap held_slot;
    require(!cap_vm_acquire(&held_slot, STACK, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
            "a stack that holds the exit word");
    char *held = cap_vm_copyable(&held_slot);
    unsigned long held_address = cursor(held);
    *(volatile int *)(held + 128) = 0x6666;
    printf("ROOT_EXIT_WORD case=in-stack word=0x%lx\n", held_address + 128);
    fflush(stdout);
    exit_with_word(held + 128, held, STACK, own + STACK);
    require(!registered(held_address), "the stack that holds the exit word is released");
    pass("stack_exit_word_dropped");

    uintptr_t identity;
    unsigned char *frame = frame_new(word_exit_worker, own + STACK, &identity);
    unsigned long frame_address = cursor(frame);
    word_ref = frame + 64;
    word_stack = NULL;
    word_stack_bytes = 0;
    long id = __capstone_vm_thread_create_frame(frame);
    require(id > 0, "create the thread on its frame");
    require(!capstone_virtual_thread_join(id), "join the thread that named its frame");
    require(!registered(frame_address), "the frame is released");
    pass("frame_exit_word_not_written");
    return 0;
}

/* attach-readonly: a pthread detaches its transport, which ends the launcher's hold on its
 * blocks, and makes its exchange region read-only. Its next delegated call, a read(), attaches
 * the transport again, and the launcher, which would write the read's data into the exchange
 * region, refuses the attachment and ends the program. */
static volatile int attach_errno = -1;
static volatile int attach_ready, attach_go;
static int attach_fds[2];
static volatile long attach_read = -1000;

static void *attach_worker(void *arg)
{
    char *exchange = __capstone_virtual_exchange();
    __capstone_vm_transport_detach();
    attach_errno = !exchange ? -2 : mprotect(exchange, LARGE, PROT_READ) ? errno : 0;
    __atomic_store_n(&attach_ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&attach_go, __ATOMIC_ACQUIRE))
        __capstone_vm_futex((void *)&attach_go, 0 /* FUTEX_WAIT */, 0, -1, 0, 0);
    char buffer[8];
    attach_read = read(attach_fds[0], buffer, sizeof buffer);
    return arg;
}

static int mode_attach_readonly(void)
{
    (void)grant();
    pthread_t thread;
    require(!pipe(attach_fds) && write(attach_fds[1], "transpor", 8) == 8, "a pipe with data");
    require(!pthread_create(&thread, NULL, attach_worker, NULL), "pthread_create");
    require(wait_for(&attach_ready, 1, 10000), "the worker starts");
    printf("ROOT_ATTACH protect_errno=%d\n", attach_errno);
    fflush(stdout);
    require(attach_errno == 0, "mprotect of the exchange region before the worker attaches");
    pass("attach_protect_before_attach");
    __atomic_store_n(&attach_go, 1, __ATOMIC_RELEASE);
    __capstone_vm_futex((void *)&attach_go, 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    for (unsigned spins = 0; spins < 500 && attach_read == -1000; ++spins)
        pause_ms(10);
    printf("ROOT_ATTACH_READ result=%ld\n", attach_read);
    fflush(stdout);
    return 0;
}

/* The base of a reference's bounds (LCC selector 3). */
static unsigned long base_of(void *p)
{
    unsigned long base;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x3" : "=r"(base) : "r"(p));
    return base;
}

static void *stack_pointer(void)
{
    void *sp;
    __asm__ volatile("movc %0, sp" : "=r"(sp));
    return sp;
}

/* create-gate: the launcher hands the module only a frame that the program may end, a whole
 * one-page heap mapping named at its address, and the module takes only the frame that a0
 * names. */
static int mode_create_gate(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    sublet_cap slot;

    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_METADATA),
            "a page of the allocator's metadata");
    unsigned char *metadata = frame_fill(cap_vm_copyable(&slot), frame_child, stack + STACK,
                                         &identity);
    long id = __capstone_vm_thread_create_frame(metadata);
    printf("ROOT_CREATE metadata=%ld\n", id);
    fflush(stdout);
    require(id == -EINVAL, "thread create with a frame in a metadata page");
    __capstone_context_identity_give(identity);
    metadata[PAGE - 1] = 0x5b;
    require(metadata[PAGE - 1] == 0x5b && ((uint64_t *)metadata)[7] == 3,
            "the metadata page after the refused create");
    pass("create_metadata_frame_refused");

    void *sp = stack_pointer();
    void *launcher_stack = at_address(sp, base_of(sp));
    id = __capstone_vm_thread_create_frame(launcher_stack);
    printf("ROOT_CREATE launcher_stack=%ld base=0x%lx type=%lu\n", id, base_of(sp),
           type_of(launcher_stack));
    fflush(stdout);
    require(id == -EINVAL, "thread create with the launcher's stack mapping");
    pass("create_launcher_page_refused");

    unsigned char *frame = frame_new(frame_child, stack + STACK, &identity);
    void *inside = at_address(frame, cursor(frame) + 64);
    id = __capstone_vm_thread_create_frame(inside);
    printf("ROOT_CREATE inside=%ld\n", id);
    fflush(stdout);
    require(id == -EINVAL,
            "thread create with a frame reference whose cursor is not the frame's address");
    frame[PAGE - 1] = 0x5c;
    require(frame[PAGE - 1] == 0x5c && ((uint64_t *)frame)[7] == 3,
            "the frame after the refused create");
    pass("create_mismatched_frame_refused");

    id = __capstone_vm_thread_create_frame(frame);
    printf("ROOT_CREATE own=%ld\n", id);
    fflush(stdout);
    require(id > 0, "thread create through the frame's own reference afterwards");
    require(!capstone_virtual_thread_join(id) && __atomic_load_n(&child_ran, __ATOMIC_ACQUIRE),
            "the created thread ran");
    pass("create_after_refusals");
    return 0;
}

/* ---- Stage 4c: words, madvise, ids and transports through references ---- */

/* The end of a reference's bounds (LCC selector 4). */
static unsigned long end_of(void *p)
{
    unsigned long end;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x4" : "=r"(end) : "r"(p));
    return end;
}

/* A context on a frame of its own that waits until the main thread releases it (release_blocked)
 * and then sets other_done. Its frame belongs to the module from the create on; the program keeps
 * only the stale reference from before. */
static volatile int other_done;

static void *other_entry(void *arg)
{
    while (!__atomic_load_n(&release_word, __ATOMIC_ACQUIRE))
        __capstone_vm_futex((void *)&release_word, 0 /* FUTEX_WAIT */, 0, -1, 0, 0);
    __atomic_store_n(&other_done, 1, __ATOMIC_RELEASE);
    return arg;
}

/* Create the waiting context on a new frame; *frame_out is the frame's reference from before the
 * create. */
static long other_thread(char *stack_top, unsigned char **frame_out)
{
    uintptr_t identity;
    unsigned char *frame = frame_new(other_entry, stack_top, &identity);
    long id = __capstone_vm_thread_create_frame(frame);
    require(id > 0, "create the other thread on its frame");
    *frame_out = frame;
    return id;
}

/* The slot of x1 (ra) in a frame: frame + 64 + 16 * 1. The escape saves the paused context's
 * registers there and the resume loads them, so a store of 0 clears the tag of its return address. */
#define FRAME_RA 80

/* A context that waits on *waited_word with FUTEX_WAIT until the word is 0, then sets waiter_done. */
static int *volatile waited_word;
static volatile int waiter_done;

static void *word_waiter_entry(void *arg)
{
    while (__atomic_load_n(waited_word, __ATOMIC_ACQUIRE))
        __capstone_vm_futex(waited_word, 0 /* FUTEX_WAIT */, 1, -1, 0, 0);
    __atomic_store_n(&waiter_done, 1, __ATOMIC_RELEASE);
    return arg;
}

/* exit-word-refs: exit words that come as references. Each refused word keeps its value and the
 * exit completes; the written one releases its futex waiter. */
static int mode_exit_word_refs(void)
{
    (void)grant();
    char *stacks = map(3 * STACK);
    char *own = stacks + STACK;
    unsigned char *frame;

    long other = other_thread(stacks + 2 * STACK, &frame);
    pause_ms(200);
    unsigned long target = cursor(frame) + FRAME_RA;
    printf("ROOT_EXIT_WORD case=other-frame word=0x%lx\n", target);
    fflush(stdout);
    exit_with_word(at_address(frame, target), NULL, 0, own);
    release_blocked();
    require(wait_for(&other_done, 1, 5000), "the other thread runs on after the exit");
    require(!capstone_virtual_thread_join(other), "join the other thread");
    pass("other_frame_word_not_written");

    char *three = map(3 * PAGE);
    require(!mprotect(three + PAGE, PAGE, PROT_NONE), "make the middle page inaccessible");
    require(!mprotect(three + PAGE, PAGE, PROT_READ | PROT_WRITE), "restore the middle page");
    int *word = (int *)(three + 2 * PAGE + 64);
    *word = 1;
    waited_word = word;
    long waiter = context(word_waiter_entry, stacks + 3 * STACK);
    pause_ms(200);
    printf("ROOT_EXIT_WORD case=middle-protected word=0x%lx\n", cursor(word));
    fflush(stdout);
    exit_with_word(word, NULL, 0, own);
    require(*(volatile int *)word == 0, "the launcher writes the word in the third page");
    require(wait_for(&waiter_done, 1, 5000), "the futex waiter on the word returns");
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("middle_protected_word_written");

    if (!root_grants) {
        skip("stale_reference_word_not_written", ADDRESS_AUTHORITY);
        skip("narrow_reference_word_not_written", ADDRESS_AUTHORITY);
        return 0;
    }
    char *page = map(PAGE);
    *(volatile int *)(page + 64) = 0x3333;
    char *renewed = capstone_cap_renew(page);
    exit_with_word(page + 64, NULL, 0, own);
    require(*(volatile int *)(renewed + 64) == 0x3333, "the word behind a stale reference");
    pass("stale_reference_word_not_written");

    char *other_page = map(PAGE);
    unsigned long base = cursor(other_page);
    *(volatile int *)(other_page + 128) = 0x4444;
    char *narrow = __builtin_capstone_cap_shrink(other_page, base + 128, base + 130);
    exit_with_word(narrow, NULL, 0, own);
    require(*(volatile int *)(other_page + 128) == 0x4444, "the word behind a narrowed reference");
    pass("narrow_reference_word_not_written");
    return 0;
}

/* futex: a wait reads the word, so it needs a live reference that covers the word, outside every
 * thread's frame. A wake reads nothing and takes an integer. A wait that the launcher accepts with
 * the word's own value times out (1 ms); a refused one answers EFAULT without waiting. */
static long futex_wait_ms(void *word, int value)
{
    return __capstone_vm_futex(word, 0 /* FUTEX_WAIT */, value, 0, 1000000, 0);
}

static int mode_futex(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    char *page = map(PAGE);
    int *word = (int *)(page + 64);
    *word = 5;

    long timed = futex_wait_ms(word, 5), other_value = futex_wait_ms(word, 6);
    printf("ROOT_FUTEX reference timed=%ld other_value=%ld\n", timed, other_value);
    fflush(stdout);
    require(timed == -ETIMEDOUT && other_value == -EAGAIN, "a wait through the word's reference");
    pass("futex_wait_through_reference");

    if (root_grants) {
        long rc = futex_wait_ms(untagged(cursor(word)), 5);
        require(rc == -EFAULT, "a wait on an integer address");
        pass("futex_wait_integer_refused");

        char *stale = map(PAGE);
        *(volatile int *)(stale + 64) = 5;
        char *renewed = capstone_cap_renew(stale);
        rc = futex_wait_ms(stale + 64, 5);
        require(rc == -EFAULT && *(volatile int *)(renewed + 64) == 5,
                "a wait through the reference from before CRENEW");
        pass("futex_wait_stale_refused");

        unsigned long base = cursor(page);
        rc = futex_wait_ms(__builtin_capstone_cap_shrink(page, base + 64, base + 66), 5);
        require(rc == -EFAULT, "a wait through a reference that covers half the word");
        pass("futex_wait_narrow_refused");
    } else {
        skip("futex_wait_integer_refused", ADDRESS_AUTHORITY);
        skip("futex_wait_stale_refused", ADDRESS_AUTHORITY);
        skip("futex_wait_narrow_refused", ADDRESS_AUTHORITY);
    }

    unsigned char *frame;
    long other = other_thread(stacks + STACK, &frame);
    pause_ms(200);
    long rc = futex_wait_ms(at_address(frame, cursor(frame) + FRAME_RA), 0);
    printf("ROOT_FUTEX frame=%ld\n", rc);
    fflush(stdout);
    release_blocked();
    require(rc == -EFAULT, "a wait on a word in the other thread's frame");
    require(wait_for(&other_done, 1, 5000) && !capstone_virtual_thread_join(other),
            "the other thread runs on");
    pass("futex_wait_frame_refused");

    long woken = __capstone_vm_futex(untagged(cursor(word)), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    long outside = __capstone_vm_futex(untagged(4096), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    printf("ROOT_FUTEX wake_integer=%ld wake_outside=%ld\n", woken, outside);
    fflush(stdout);
    require(woken == 0 && outside == -EFAULT, "a wake by integer address");
    pass("futex_wake_integer_answers");
    return 0;
}

/* advise: madvise(MADV_DONTNEED) and MADV_FREE destroy contents, so they need a reference to the
 * mapping that covers the range, as mprotect does. A refused call changes nothing; an accepted one
 * leaves the range reading zero, the tags of stored references included. */
static int mode_advise(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *two = map(2 * PAGE);
    unsigned long base = cursor(two);
    memset(two, 0x5a, 2 * PAGE);
    *(char **)(two + PAGE + 64) = target;

    if (root_grants) {
        require(refused(madvise(untagged(base), 2 * PAGE, MADV_DONTNEED)),
                "madvise with an integer address");
        require(two[0] == 0x5a && two[2 * PAGE - 1] == 0x5a, "the mapping after the refusal of an address");
        pass("advise_integer_refused");

        char *page = map(PAGE);
        page[1] = 0x6a;
        char *renewed = capstone_cap_renew(page);
        require(refused(madvise(page, PAGE, MADV_DONTNEED)),
                "madvise through the reference from before CRENEW");
        require(renewed[1] == 0x6a, "the renewed mapping after the refusal");
        pass("advise_stale_refused");

        char *narrow = __builtin_capstone_cap_shrink(two, base, base + PAGE);
        require(refused(madvise(narrow, 2 * PAGE, MADV_DONTNEED)),
                "madvise of two pages through a reference narrowed to one");
        require(two[0] == 0x5a && two[PAGE] == 0x5a, "the mapping after the refusal of a narrowed reference");
        pass("advise_narrowed_refused");

        char *object = capstone_cap_derive(two, 0, 2 * PAGE);
        require(refused(madvise(object, 2 * PAGE, MADV_DONTNEED)),
                "madvise through an object over the whole mapping");
        require(two[1] == 0x5a && type_of(*(char **)(two + PAGE + 64)) == 1,
                "the mapping after the refusal of an object");
        pass("advise_nested_object_refused");
    } else {
        skip("advise_integer_refused", ADDRESS_AUTHORITY);
        skip("advise_stale_refused", ADDRESS_AUTHORITY);
        skip("advise_narrowed_refused", ADDRESS_AUTHORITY);
        skip("advise_nested_object_refused", ADDRESS_AUTHORITY);
    }

    char *second = __builtin_capstone_cap_shrink(two, base + PAGE, base + 2 * PAGE);
    require(!madvise(second, PAGE, MADV_DONTNEED), "madvise of the page a narrowed reference covers");
    printf("ROOT_ADVISE first=0x%x second=0x%x stored_type=%lu\n", two[0], two[PAGE + 1],
           type_of(*(char **)(two + PAGE + 64)));
    fflush(stdout);
    require(two[0] == 0x5a && two[PAGE - 1] == 0x5a, "the first page keeps its data");
    require(two[PAGE] == 0 && two[2 * PAGE - 1] == 0 && type_of(*(char **)(two + PAGE + 64)) == 7,
            "the second page reads zero, and the stored reference is gone");
    two[PAGE] = 0x5c;
    require(two[PAGE] == 0x5c, "the second page takes a write");
    pass("advise_zeroes_covered_page");

    require(!madvise(two, 2 * PAGE, MADV_FREE), "madvise(MADV_FREE) through the mapping's reference");
    require(two[0] == 0 && two[PAGE - 1] == 0 && two[PAGE] == 0, "the whole mapping reads zero");
    pass("advise_free_zeroes");

    /* A page that the program made inaccessible after it wrote data and a reference into it: the
     * advise zeroes it all the same, and the restored page reads zero (stage 4d). */
    char *none = map(2 * PAGE);
    memset(none, 0x6b, 2 * PAGE);
    *(char **)(none + PAGE + 64) = target;
    require(!mprotect(none + PAGE, PAGE, PROT_NONE), "make the second page inaccessible");
    require(!madvise(none + PAGE, PAGE, MADV_DONTNEED), "madvise of the inaccessible page");
    require(!mprotect(none + PAGE, PAGE, PROT_READ | PROT_WRITE), "restore the second page");
    printf("ROOT_ADVISE none first=0x%x second=0x%x stored_type=%lu\n", none[0], none[PAGE + 1],
           type_of(*(char **)(none + PAGE + 64)));
    fflush(stdout);
    require(none[0] == 0x6b && none[PAGE - 1] == 0x6b, "the first page keeps its data");
    require(none[PAGE] == 0 && none[2 * PAGE - 1] == 0 && type_of(*(char **)(none + PAGE + 64)) == 7,
            "the inaccessible page reads zero after the restore, and the stored reference is gone");
    pass("advise_prot_none_zeroes");
    return 0;
}

/* join-burned: a create that fails after the kernel checked the frame issues no id, and the main
 * thread joining itself answers EDEADLK. The frame is made read-only, so the module cannot pin it
 * and refuses the create after it checked the reference. */
static int mode_join_burned(void)
{
    (void)grant();
    char *stacks = map(3 * STACK);
    long first = context(idle_entry, stacks + STACK);
    require(!capstone_virtual_thread_join(first), "join the first context");

    uintptr_t identity;
    unsigned char *frame = frame_new(frame_child, stacks + 2 * STACK, &identity);
    require(!mprotect(frame, PAGE, PROT_READ), "make the frame read-only");
    long failed = __capstone_vm_thread_create_frame(frame);
    __capstone_context_identity_give(identity);
    long rc = __capstone_vm_thread_join(first + 1);
    printf("ROOT_JOIN_BURNED first=%ld failed=%ld join=%ld\n", first, failed, rc);
    fflush(stdout);
    require(failed < 0 && failed != -EAGAIN, "a create on a read-only frame fails before the kernel takes it");
    require(rc == -ESRCH, "a join of the failed create's id");
    pass("join_failed_create_refused");

    require(!mprotect(frame, PAGE, PROT_READ | PROT_WRITE) && !munmap(frame, PAGE),
            "the frame stays the program's and unmaps");
    long next = context(idle_entry, stacks + 3 * STACK);
    printf("ROOT_JOIN_BURNED next=%ld\n", next);
    fflush(stdout);
    require(next == first + 1, "the next create takes the id that the failed create did not use");
    require(!capstone_virtual_thread_join(next), "join it");
    pass("failed_create_consumes_no_id");

    rc = __capstone_vm_thread_join(0);
    printf("ROOT_JOIN_BURNED self=%ld\n", rc);
    fflush(stdout);
    require(rc == -EDEADLK, "the main thread joins itself");
    pass("main_self_join_deadlock");
    return 0;
}

/* attach-exchange and attach-address: a pthread gives its transport back to the launcher and then
 * attaches a transport that names a mapping it may not: the launcher's own exchange region through
 * the reference the program holds to it, or a fresh mapping by its plain address. The launcher
 * refuses the attachment (CAPSTONE_VM_TRANSPORT_REFUSED) and ends the program, so
 * ROOT_ATTACH_RETURNED never prints. */
extern void __capstone_vm_delegate_transport(void *meta, void *exchange, size_t bytes);
static void *volatile named_exchange;
static volatile unsigned long named_bytes;
static volatile int named_by_address, attach_returned;

static void *attach_named_worker(void *arg)
{
    void *meta = mmap(NULL, CAPSTONE_DELEGATE_META_BYTES, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void *exchange = named_exchange;
    if (named_by_address) {
        void *fresh = mmap(NULL, LARGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        exchange = fresh == MAP_FAILED ? NULL : untagged(cursor(fresh));
    }
    if (meta == MAP_FAILED || !exchange)
        return arg;
    /* No delegated call from here on: the next one would attach the thread's own transport. */
    __capstone_vm_transport_detach();
    __capstone_vm_delegate_transport(meta, exchange, named_bytes);
    __atomic_store_n(&attach_returned, 1, __ATOMIC_RELEASE);
    return arg;
}

/* The launcher's grant that start.S's _start received in register a<index> (0 the meta block,
 * 1 the exchange region): _start stores a0 to a3 at the top of the main stack, at top - 80 + 16 *
 * index, before it calls anything, and that frame lasts as long as the program. */
static void *launcher_grant(unsigned index)
{
    void *sp = stack_pointer();
    return *(void **)at_address(sp, end_of(sp) - 80 + 16 * index);
}

static int attach_named(void *exchange, unsigned long bytes, int by_address)
{
    pthread_t thread;
    named_exchange = exchange;
    named_bytes = bytes;
    named_by_address = by_address;
    require(!pthread_create(&thread, NULL, attach_named_worker, NULL), "pthread_create");
    for (unsigned spins = 0; spins < 500 && !__atomic_load_n(&attach_returned, __ATOMIC_ACQUIRE); ++spins)
        pause_ms(10);
    printf("ROOT_ATTACH_RETURNED\n");
    fflush(stdout);
    return 0;
}

static int mode_attach_exchange(void)
{
    (void)grant();
    void *exchange = launcher_grant(1);
    unsigned long type = type_of(exchange);
    unsigned long bytes = type == 1 ? end_of(exchange) - base_of(exchange) : 0;
    printf("ROOT_ATTACH launcher_exchange type=%lu address=0x%lx bytes=%lu\n", type,
           type == 1 ? cursor(exchange) : 0UL, bytes);
    fflush(stdout);
    require(type == 1 && bytes, "the launcher's exchange region as the program holds it");
    pass("launcher_exchange_reference");
    return attach_named(exchange, bytes, 0);
}

static int mode_attach_address(void)
{
    (void)grant();
    if (!root_grants) {
        skip("attach_address_refused", ADDRESS_AUTHORITY);
        return 0;
    }
    return attach_named(NULL, LARGE, 1);
}

/* ---- Stage 4d: check and access in one step ---- */

/* A busy wait of about ms milliseconds that makes no service call: a thread that must act while
 * another thread's service holds service_lock cannot sleep through the launcher. spin_calibrate
 * measures the loop once against the clock, before the race. */
static unsigned long spins_per_ms;

static void spin_calibrate(void)
{
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (volatile unsigned long i = 0; i < 4000000UL; ++i)
        ;
    clock_gettime(CLOCK_MONOTONIC, &b);
    long ms = (b.tv_sec - a.tv_sec) * 1000L + (b.tv_nsec - a.tv_nsec) / 1000000L;
    spins_per_ms = 4000000UL / (ms > 0 ? (unsigned long)ms : 1UL);
    printf("ROOT_SPIN per_ms=%lu\n", spins_per_ms);
    fflush(stdout);
}

static void spin_ms(unsigned long ms)
{
    for (volatile unsigned long i = 0; i < spins_per_ms * ms; ++i)
        ;
}

/* A context that waits on race_word for race_value with a timeout of race_seconds and records the
 * answer in race_rc. */
static int *volatile race_word;
static volatile int race_value, race_started, race_done;
static volatile long race_rc, race_seconds;
/* With race_hold set the waiter, once it has answered, waits until the main thread sets
 * race_release, so its exit (which releases its frame's mapping, and with it the address the
 * launcher reuses next) comes only after the main thread mapped the old address again. */
static volatile int race_hold;
static int race_release;

static void *race_wait_entry(void *arg)
{
    __atomic_store_n(&race_started, 1, __ATOMIC_RELEASE);
    race_rc = __capstone_vm_futex(race_word, 0 /* FUTEX_WAIT */, race_value, race_seconds, 0, 0);
    __atomic_store_n(&race_done, 1, __ATOMIC_RELEASE);
    while (race_hold && !__atomic_load_n(&race_release, __ATOMIC_ACQUIRE))
        __capstone_vm_futex(&race_release, 0 /* FUTEX_WAIT */, 0, -1, 0, 0);
    return arg;
}

/* Let a held waiter exit. */
static void race_let_go(void)
{
    __atomic_store_n(&race_release, 1, __ATOMIC_RELEASE);
    __capstone_vm_futex(untagged(cursor(&race_release)), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
}

static long race_waiter(int *word, int value, long seconds, char *stack_top)
{
    race_word = word;
    race_value = value;
    race_seconds = seconds;
    race_started = race_done = 0;
    race_release = 0;
    race_rc = 1;
    return context(race_wait_entry, stack_top);
}

/* futex-race: the futex window of stage 4c, forced open by the launcher's test hook. */
static int mode_futex_race(void)
{
    (void)grant();
    char *stacks = map(3 * STACK);

    char *page = map(PAGE);
    unsigned long address = cursor(page);
    *(volatile int *)(page + 64) = 5;
    long waiter = race_waiter((int *)(page + 64), 5, 6, stacks + STACK);
    require(wait_for(&race_started, 1, 5000), "the waiter starts");
    pause_ms(500);
    require(!munmap(page, PAGE), "munmap of the waited word's whole mapping");
    char *again = map(PAGE);
    *(volatile int *)(again + 64) = 6;
    printf("ROOT_FUTEX_RACE remap same_address=%d\n", cursor(again) == address);
    fflush(stdout);
    require(wait_for(&race_done, 1, 12000), "the waiter returns");
    printf("ROOT_FUTEX_RACE remap rc=%ld\n", race_rc);
    fflush(stdout);
    require(cursor(again) == address, "the new mapping lies at the old address");
    /* Stage 4e: the unmap woke the waiter, which compared the old mapping's word (EAGAIN would mean
     * the new one's) and then answers 0. */
    require(race_rc == 0, "the waiter compared the old mapping's word, never the new one's");
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("futex_wait_never_reads_new_mapping");

    int *word = (int *)(again + 128);
    *word = 7;
    waiter = race_waiter(word, 7, 6, stacks + 2 * STACK);
    require(wait_for(&race_started, 1, 5000), "the second waiter starts");
    pause_ms(500);
    __atomic_store_n(word, 0, __ATOMIC_RELEASE);
    long woken = __capstone_vm_futex(untagged(cursor(word)), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    require(wait_for(&race_done, 1, 12000), "the second waiter returns");
    printf("ROOT_FUTEX_RACE wake woken=%ld rc=%ld\n", woken, race_rc);
    fflush(stdout);
    require(woken == 1 && race_rc == 0, "the wake inside the window reaches the waiter");
    require(!capstone_virtual_thread_join(waiter), "join the second waiter");
    pass("futex_wake_in_window_caught");
    return 0;
}

/* futex-remap: a wake names a mapping, not an address. */
static int mode_futex_remap(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    char *page = map(PAGE);
    unsigned long address = cursor(page);

    *(volatile int *)(page + 64) = 3;
    race_hold = 1;
    long waiter = race_waiter((int *)(page + 64), 3, 3, stacks + STACK);
    require(wait_for(&race_started, 1, 5000), "the waiter starts");
    pause_ms(500);
    require(!munmap(page, PAGE), "munmap of the waited word's whole mapping");
    char *again = map(PAGE);
    long woken = __capstone_vm_futex(untagged(address + 64), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    require(wait_for(&race_done, 1, 10000), "the waiter returns");
    printf("ROOT_FUTEX_REMAP same_address=%d woken=%ld rc=%ld\n", cursor(again) == address, woken,
           race_rc);
    fflush(stdout);
    require(cursor(again) == address, "the new mapping lies at the old address");
    /* Stage 4e: the unmap woke the waiter (0), and the wake of the new mapping finds nobody. */
    require(woken == 0 && race_rc == 0,
            "a wake of the new mapping does not reach the old mapping's waiter");
    race_let_go();
    race_hold = 0;
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("futex_wake_reaches_only_its_mapping");
    return 0;
}

/* exit-race: the exit-word window of stage 4c, forced open by the launcher's test hook. */
static int *volatile exit_race_word;
static volatile int exit_race_started;

static void *exit_race_entry(void *arg)
{
    (void)arg;
    __atomic_store_n(&exit_race_started, 1, __ATOMIC_RELEASE);
    __capstone_vm_thread_exit_clear(exit_race_word, NULL, 0, 0);
}

static int mode_exit_race(void)
{
    (void)grant();
    if (!root_grants) {
        skip("exit_word_after_revoke_not_written",
             "root_grants=0 has no lifetime that a CREVOKE ends; the address is the authority");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *page = map(PAGE);
    char *object = capstone_cap_derive(page, 0, 64);

    exit_race_word = (int *)(object + 16);
    *exit_race_word = 1;
    long id = context(exit_race_entry, stacks + STACK);
    while (!__atomic_load_n(&exit_race_started, __ATOMIC_ACQUIRE))
        ;
    /* The exiting context passes the launcher's own checks within a few milliseconds and then
     * sleeps in the hook with service_lock held: no service call from here until the join. */
    spin_ms(500);
    capstone_cap_release(object);
    char *reused = capstone_cap_derive(page, 0, 64);
    *(volatile int *)(reused + 16) = 0x7777;
    require(!capstone_virtual_thread_join(id), "join the exiting context");
    int value = *(volatile int *)(page + 16);
    printf("ROOT_EXIT_RACE value=0x%x\n", value);
    fflush(stdout);
    require(value == 0x7777, "the exit word's store does not reach the object reused after the revoke");
    pass("exit_word_after_revoke_not_written");
    return 0;
}

/* join-withdrawn: a joiner waits on the entry of a create that then fails. */
static volatile long withdrawn_target, withdrawn_rc = 1;
static volatile int withdrawn_go, withdrawn_started, withdrawn_done;

static void *withdrawn_joiner(void *arg)
{
    while (!__atomic_load_n(&withdrawn_go, __ATOMIC_ACQUIRE))
        ;
    __atomic_store_n(&withdrawn_started, 1, __ATOMIC_RELEASE);
    withdrawn_rc = __capstone_vm_thread_join(withdrawn_target);
    __atomic_store_n(&withdrawn_done, 1, __ATOMIC_RELEASE);
    return arg;
}

static int mode_join_withdrawn(void)
{
    (void)grant();
    char *stacks = map(3 * STACK);
    uintptr_t identity;

    long joiner = context(withdrawn_joiner, stacks + STACK);
    withdrawn_target = joiner + 1;
    __atomic_store_n(&withdrawn_go, 1, __ATOMIC_RELEASE);
    require(wait_for(&withdrawn_started, 1, 5000), "the joiner starts");
    pause_ms(300);
    unsigned char *frame = frame_new(frame_child, stacks + 2 * STACK, &identity);
    unsigned long frame_address = cursor(frame);
    long failed = __capstone_vm_thread_create_frame(frame);
    __capstone_context_identity_give(identity);
    require(wait_for(&withdrawn_done, 1, 6000), "the joiner of the failed create's id returns");
    printf("ROOT_JOIN_WITHDRAWN target=%ld failed=%ld join=%ld\n", withdrawn_target, failed,
           withdrawn_rc);
    fflush(stdout);
    require(failed == -EAGAIN, "the create fails after the launcher took the frame");
    require(!registered(frame_address), "the failed create's frame is released");
    require(withdrawn_rc == -ESRCH, "the join of the withdrawn id answers ESRCH");
    long next = context(idle_entry, stacks + 2 * STACK);
    printf("ROOT_JOIN_WITHDRAWN next=%ld\n", next);
    fflush(stdout);
    require(next == withdrawn_target, "the next create takes the withdrawn id");
    require(!capstone_virtual_thread_join(next) && !capstone_virtual_thread_join(joiner),
            "join the next context and the joiner");
    pass("joiner_on_withdrawn_id_returns");
    return 0;
}

/* rollback-fail: the launcher's rollback of a failed create is refused; the program ends. */
static int mode_rollback_fail(void)
{
    (void)grant();
    char *stack = map(STACK);
    uintptr_t identity;
    unsigned char *frame = frame_new(frame_child, stack + STACK, &identity);

    pass("rollback_create_started");
    long id = __capstone_vm_thread_create_frame(frame);
    printf("ROOT_ROLLBACK_RETURNED id=%ld\n", id);
    fflush(stdout);
    return 0;
}

/* die-blocked: the launcher ends the program while the main thread blocks in a delegated read. The
 * pthread sleeps half a second without a delegated call (one would attach its own transport), then
 * names the launcher's exchange region as its transport, which the launcher refuses. */
static int die_fds[2];

static void *die_attach_worker(void *arg)
{
    int sleeper = 0;
    void *meta = mmap(NULL, CAPSTONE_DELEGATE_META_BYTES, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    __capstone_vm_futex(&sleeper, 0 /* FUTEX_WAIT */, 0, 0, 500000000, 0);
    if (meta == MAP_FAILED)
        return arg;
    __capstone_vm_transport_detach();
    __capstone_vm_delegate_transport(meta, named_exchange, named_bytes);
    __atomic_store_n(&attach_returned, 1, __ATOMIC_RELEASE);
    return arg;
}

static int mode_die_blocked(void)
{
    (void)grant();
    pthread_t thread;
    void *exchange = launcher_grant(1);

    require(type_of(exchange) == 1, "the launcher's exchange region as the program holds it");
    named_exchange = exchange;
    named_bytes = end_of(exchange) - base_of(exchange);
    require(!pipe(die_fds), "an empty pipe");
    require(!pthread_create(&thread, NULL, die_attach_worker, NULL), "pthread_create");
    pass("die_reader_blocks");
    char byte;
    long n = read(die_fds[0], &byte, 1);
    printf("ROOT_DIE_READ result=%ld\n", n);
    fflush(stdout);
    return 0;
}

/* word-metadata: no word of the allocator's metadata reaches the kernel. */
static int mode_word_metadata(void)
{
    (void)grant();
    char *own = map(STACK);
    sublet_cap slot;

    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_METADATA),
            "a page of the allocator's metadata");
    char *metadata = cap_vm_copyable(&slot);
    int *word = (int *)(metadata + 64);
    *word = 0x5151;
    long rc = futex_wait_ms(word, 0x5151);
    printf("ROOT_WORD_METADATA futex=%ld\n", rc);
    fflush(stdout);
    require(rc == -EFAULT, "a futex wait on a word in the allocator's metadata");
    pass("metadata_futex_wait_refused");
    exit_with_word(word, NULL, 0, own + STACK);
    require(*(volatile int *)word == 0x5151, "the exit word in the allocator's metadata keeps its value");
    pass("metadata_exit_word_not_written");
    return 0;
}

/* ---- Stage 4e: the follow-ups of stage 4d ---- */

/* futex-readonly: a futex wait reads the word without admitting its page, so a word in a page the
 * program may only read compares as any other. */
static int mode_futex_readonly(void)
{
    (void)grant();
    char *fresh = mmap(NULL, PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(fresh != MAP_FAILED, "a read-only mapping");
    printf("ROOT_FUTEX_RO base=0x%lx\n", cursor(fresh));
    fflush(stdout);
    long same = futex_wait_ms(fresh + 64, 0), other = futex_wait_ms(fresh + 64, 1);
    printf("ROOT_FUTEX_RO fresh same=%ld other=%ld\n", same, other);
    fflush(stdout);
    require(same == -ETIMEDOUT && other == -EAGAIN, "a wait on a word of a page never written");
    pass("futex_wait_read_only_compares");

    char *page = map(PAGE);
    *(volatile int *)(page + 64) = 7;
    require(!mprotect(page, PAGE, PROT_READ), "make the word's page read-only");
    same = futex_wait_ms(page + 64, 7);
    other = futex_wait_ms(page + 64, 8);
    printf("ROOT_FUTEX_RO protected same=%ld other=%ld\n", same, other);
    fflush(stdout);
    require(same == -ETIMEDOUT && other == -EAGAIN, "a wait on a word of a page made read-only");
    pass("futex_wait_protected_compares");

    /* The positive control of the runner's admission check: an exit word's store into a page never
     * touched admits that page (one touch line under diag_admit). */
    char *own = map(STACK);
    char *untouched = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(untouched != MAP_FAILED, "a mapping never touched");
    printf("ROOT_FUTEX_RO store_base=0x%lx\n", cursor(untouched));
    fflush(stdout);
    exit_with_word(untouched + 64, NULL, 0, own + STACK);
    require(*(volatile int *)(untouched + 64) == 0, "the exit word's store into the untouched page");
    return 0;
}

static long elapsed_ms(const struct timespec *from)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (now.tv_sec - from->tv_sec) * 1000L + (now.tv_nsec - from->tv_nsec) / 1000000L;
}

/* futex-park (run with the mapping pool on): a waiter on a word of a heap mapping that the program
 * unmaps is woken when the mapping is parked, and a wake of the mapping revived at the same address
 * finds nobody. Stage 4d keyed waiters by the arena id, which the revive changes: the waiter slept
 * until its timeout. */
static int mode_futex_park(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    sublet_cap slot;
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP), "a heap mapping");
    char *heap = cap_vm_copyable(&slot);
    unsigned long address = cursor(heap);
    *(volatile int *)(heap + 64) = 9;
    long waiter = race_waiter((int *)(heap + 64), 9, 8, stacks + STACK);
    require(wait_for(&race_started, 1, 5000), "the waiter starts");
    pause_ms(300);
    struct timespec unmapped;
    clock_gettime(CLOCK_MONOTONIC, &unmapped);
    require(!munmap(heap, PAGE), "munmap of the heap mapping (parked)");
    int returned = wait_for(&race_done, 1, 3000);
    long after = elapsed_ms(&unmapped);
    require(!cap_vm_acquire(&slot, PAGE, PAGE, PROT_READ | PROT_WRITE, CAP_VM_HEAP),
            "a heap mapping of the same size (revived)");
    char *again = cap_vm_copyable(&slot);
    long woken = __capstone_vm_futex(untagged(address + 64), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    printf("ROOT_FUTEX_PARK same_address=%d returned=%d after_ms=%ld woken=%ld rc=%ld\n",
           cursor(again) == address, returned, after, woken, race_rc);
    fflush(stdout);
    require(returned && race_rc == 0, "the waiter returns when its mapping is parked");
    /* Without the pool (the unpinned module refuses a park) the mapping is retired, the next one lies
     * elsewhere and the old address belongs to no mapping (EFAULT); the runner checks the park and
     * the revive from the pool's counters where the pool works. */
    require(woken == 0 || (cursor(again) != address && woken == -EFAULT),
            "a wake of the revived mapping finds nobody");
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("futex_waiter_woken_at_park");
    return 0;
}

/* futex-end (ends with a fault): a waiter in the libc's loop (read the word through its own
 * reference, wait while it holds the value). A wake that leaves the word unchanged makes it read
 * and wait again; the end of the word's mapping wakes it, and its read through the stale reference
 * must fault with 25. */
static int *volatile end_word;
static volatile int end_passes, end_done;

static void *end_waiter_entry(void *arg)
{
    while (__atomic_load_n(end_word, __ATOMIC_ACQUIRE) == 1) {
        __atomic_add_fetch(&end_passes, 1, __ATOMIC_RELEASE);
        __capstone_vm_futex(end_word, 0 /* FUTEX_WAIT */, 1, -1, 0, 0);
    }
    __atomic_store_n(&end_done, 1, __ATOMIC_RELEASE);
    return arg;
}

static int mode_futex_end(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    char *page = map(PAGE);
    end_word = (int *)(page + 64);
    *end_word = 1;
    long waiter = context(end_waiter_entry, stacks + STACK);
    require(wait_for(&end_passes, 1, 5000), "the waiter reads the word and waits");
    pause_ms(300);
    long woken = __capstone_vm_futex(untagged(cursor(end_word)), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    require(woken == 1 && wait_for(&end_passes, 2, 5000) && !end_done,
            "a wake that leaves the word unchanged: the waiter reads it and waits again");
    pass("futex_spurious_wake_rewaits");
    pause_ms(300);
    printf("ROOT_FUTEX_END unmap passes=%d\n", end_passes);
    fflush(stdout);
    require(!munmap(page, PAGE), "munmap of the word's mapping");
    pause_ms(3000);
    printf("ROOT_STALE_READ waiter survived done=%d passes=%d\n", end_done, end_passes);
    fflush(stdout);
    (void)waiter;
    return 0;
}

/* futex-flags: the operations and flags that Linux refuses are refused, and a relative timeout that
 * overflows the clock waits without one. */
static int mode_futex_flags(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    char *page = map(PAGE);
    int *word = (int *)(page + 64);
    *word = 3;
    long wait_rt = __capstone_vm_futex(word, 0 | 256 /* FUTEX_WAIT | FUTEX_CLOCK_REALTIME */, 3, 0,
                                       1000000, 0);
    long wake_rt = __capstone_vm_futex(untagged(cursor(word)), 1 | 256, 1, -1, 0, 0);
    long unknown = __capstone_vm_futex(word, 0 | 1024 /* an unknown flag */, 3, 0, 1000000, 0);
    long bitset_rt = __capstone_vm_futex(word, 9 | 256 /* FUTEX_WAIT_BITSET | REALTIME */, 3, 1, 0, -1);
    printf("ROOT_FUTEX_FLAGS wait_rt=%ld wake_rt=%ld unknown=%ld bitset_rt=%ld\n", wait_rt, wake_rt,
           unknown, bitset_rt);
    fflush(stdout);
    require(wait_rt == -ENOSYS && wake_rt == -ENOSYS, "FUTEX_CLOCK_REALTIME without FUTEX_WAIT_BITSET");
    pass("futex_clock_realtime_refused");
    require(unknown == -ENOSYS, "an unknown futex flag");
    pass("futex_unknown_flag_refused");
    require(bitset_rt == -ETIMEDOUT, "FUTEX_WAIT_BITSET with FUTEX_CLOCK_REALTIME and a past deadline");

    long waiter = race_waiter(word, 3, 0x7fffffffffffffffL, stacks + STACK);
    require(wait_for(&race_started, 1, 5000), "the waiter with a huge timeout starts");
    pause_ms(500);
    int early = race_done;
    __atomic_store_n(word, 0, __ATOMIC_RELEASE);
    long woken = __capstone_vm_futex(untagged(cursor(word)), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    require(wait_for(&race_done, 1, 5000), "the waiter returns");
    printf("ROOT_FUTEX_FLAGS huge early=%d woken=%ld rc=%ld\n", early, woken, race_rc);
    fflush(stdout);
    require(!early && woken == 1 && race_rc == 0, "a wait with a huge relative timeout sleeps until woken");
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("futex_huge_timeout_waits");
    return 0;
}

/* eintr-word (hooks: CAPSTONE_VM_TEST_EINTR=word and the module's test_hold_ms): the futex wait's
 * CV_WORD waits for vm_lock and a signal interrupts the wait; the launcher repeats the ioctl, and the
 * wait compares as without the signal. */
static int mode_eintr_word(void)
{
    (void)grant();
    char *page = map(PAGE);
    int *word = (int *)(page + 64);
    *word = 5;
    long same = futex_wait_ms(word, 5), other = futex_wait_ms(word, 6);
    printf("ROOT_EINTR word same=%ld other=%ld\n", same, other);
    fflush(stdout);
    require(same == -ETIMEDOUT && other == -EAGAIN, "the futex wait after interrupted waits for vm_lock");
    pass("eintr_futex_wait_retries");
    return 0;
}

/* eintr-exit (hooks: CAPSTONE_VM_TEST_EINTR=exit): the same for a thread's exit, which then
 * completes and writes its exit word. */
static int mode_eintr_exit(void)
{
    (void)grant();
    char *own = map(STACK);
    char *page = map(PAGE);
    *(volatile int *)(page + 64) = 0x5555;
    exit_with_word(page + 64, NULL, 0, own + STACK);
    printf("ROOT_EINTR exit word=0x%x\n", *(volatile int *)(page + 64));
    fflush(stdout);
    require(*(volatile int *)(page + 64) == 0, "the exit after interrupted waits writes its word");
    pass("eintr_thread_exit_retries");
    return 0;
}

/* advise-refuse (module hook test_advise_refuse=3): the module refuses the third page of a madvise,
 * as a page that cannot be cleared: the call fails and no page of the range changes. */
static int mode_advise_refuse(void)
{
    (void)grant();
    char *target = map(PAGE);
    char *three = map(3 * PAGE);
    memset(three, 0x77, 3 * PAGE);
    *(char **)(three + 64) = target;
    errno = 0;
    int rc = madvise(three, 3 * PAGE, MADV_DONTNEED);
    int error = errno;
    printf("ROOT_ADVISE_REFUSE rc=%d errno=%d\n", rc, error);
    fflush(stdout);
    require(rc == -1 && error == EIO, "madvise with a page the kernel cannot clear");
    require(three[0] == 0x77 && three[PAGE] == 0x77 && three[3 * PAGE - 1] == 0x77 &&
            type_of(*(char **)(three + 64)) == 1, "no page of the range changed");
    pass("advise_refused_changes_nothing");
    return 0;
}

/* die-stderr (the runner sends stderr into a pipe that nobody reads): the main thread writes into
 * stderr until the pipe is full and blocks there; a pthread names the process transport with a
 * wrong exchange region, which the launcher refuses without a line of its own (virtual_service
 * answers -1), and the launcher ends the program with 125 at once, its own line dropped. */
static void *volatile die_meta;

static void *die_stderr_worker(void *arg)
{
    int sleeper = 0;
    __capstone_vm_futex(&sleeper, 0 /* FUTEX_WAIT */, 0, 0, 500000000, 0);
    __capstone_vm_transport_detach();
    __capstone_vm_delegate_transport(die_meta, NULL, 0);
    __atomic_store_n(&attach_returned, 1, __ATOMIC_RELEASE);
    return arg;
}

static int mode_die_stderr(void)
{
    (void)grant();
    pthread_t thread;
    static char block[4096];
    die_meta = launcher_grant(0);
    require(type_of(die_meta) == 1, "the launcher's meta block as the program holds it");
    memset(block, 'x', sizeof block);
    require(!pthread_create(&thread, NULL, die_stderr_worker, NULL), "pthread_create");
    pass("die_stderr_blocks");
    for (unsigned i = 0; i < 256; ++i)
        if (write(2, block, sizeof block) < 0) break;
    printf("ROOT_DIE_WRITTEN\n");
    fflush(stdout);
    return 0;
}

/* protect-exec: only the image's text is executable, and the image is no mapping the program may
 * protect, advise or unmap: its data reference (gp) covers the whole image, so only the page's write
 * bit protects the text. */
static int mode_protect_exec(void)
{
    (void)grant();
    void *image;
    __asm__ volatile("movc %0, gp" : "=r"(image));
    unsigned long base = base_of(image), end = end_of(image);
    char *text = at_address(image, base), *data = at_address(image, end - PAGE);
    printf("ROOT_PROTECT_EXEC image=0x%lx+0x%lx type=%lu\n", base, end - base, type_of(image));
    fflush(stdout);
    require(type_of(text) == 1 && type_of(data) == 1, "the image's reference at its text and data");
    errno = 0;
    require(mprotect(text, PAGE, PROT_READ | PROT_WRITE) == -1 && errno == EINVAL,
            "mprotect of the image's text");
    errno = 0;
    require(mprotect(data, PAGE, PROT_READ) == -1 && errno == EINVAL, "mprotect of the image's data");
    errno = 0;
    require(madvise(text, PAGE, MADV_DONTNEED) == -1 && errno == EINVAL, "madvise of the image");
    pass("image_protect_refused");
    require(refused(munmap(text, PAGE)) && refused(munmap(data, PAGE)) &&
            refused(munmap(text, end - base)), "munmap of the image");
    pass("image_unmap_refused");

    errno = 0;
    void *exec = mmap(NULL, PAGE, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    require(exec == MAP_FAILED && errno == EACCES, "mmap with PROT_EXEC");
    pass("map_exec_refused");
    char *page = map(PAGE);
    errno = 0;
    require(mprotect(page, PAGE, PROT_READ | PROT_EXEC) == -1 && errno == EACCES &&
            protected_as(cursor(page), "rw-"), "mprotect with PROT_EXEC");
    pass("protect_exec_refused");
    char *two = map(2 * PAGE);
    errno = 0;
    require(mmap(two + PAGE, PAGE, PROT_READ | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                 -1, 0) == MAP_FAILED && errno == EACCES, "MAP_FIXED with PROT_EXEC");
    pass("fixed_exec_refused");

    /* The startup block (block A): its grant carried the only right a grant ever narrowed (read only),
     * and since block A the page alone protects it. The program still holds its reference: _start
     * stored a2 at 32(sp) after sp -= 80, at the top of the main stack. No service may make it
     * writable, take its pages or end it. */
    void *sp;
    __asm__ volatile("movc %0, sp" : "=r"(sp));
    void *startup = *(void **)at_address(sp, end_of(sp) - 80 + 32);
    unsigned long start = base_of(startup), length = end_of(startup) - start;
    printf("ROOT_STARTUP base=0x%lx bytes=0x%lx type=%lu\n", start, length, type_of(startup));
    fflush(stdout);
    require(type_of(startup) == 1 && length >= PAGE && !(start & (PAGE - 1)) && !(length & (PAGE - 1)) &&
            protected_as(start, "r--"), "the startup block's reference on the main stack");
    char *first = at_address(startup, start);
    errno = 0;
    require(mprotect(first, PAGE, PROT_READ | PROT_WRITE) == -1 && errno == EINVAL,
            "mprotect of the startup block to read-write");
    errno = 0;
    require(mprotect(first, length, PROT_READ | PROT_WRITE) == -1 && errno == EINVAL,
            "mprotect of the whole startup block to read-write");
    require(mmap(first, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) ==
            MAP_FAILED, "MAP_FIXED over the startup block");
    errno = 0;
    require(madvise(first, PAGE, MADV_DONTNEED) == -1 && errno == EINVAL, "madvise of the startup block");
    require(refused(munmap(first, length)) && refused(munmap(first, PAGE)), "munmap of the startup block");
    require(protected_as(start, "r--") && first[0] == ((char *)at_address(startup, start))[0],
            "the startup block stays read-only and mapped");
    pass("startup_stays_read_only");
    return 0;
}

/* ---- Stage 4e: partial munmap as a page operation ---- */

/* partial-hole (ends with a page fault): the middle page of three goes. The outer pages keep working
 * through the old references, no new mapping lands in the hole, requests on the hole are refused as
 * for an unmapped page, and the last line reads the hole through the live reference: a page fault
 * (13), not a lifetime fault. */
static int mode_partial_hole(void)
{
    (void)grant();
    char *stacks = map(2 * STACK);
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x31, 3 * PAGE);
    int *middle = (int *)(three + PAGE + 64);
    *middle = 4;
    long waiter = race_waiter(middle, 4, 10, stacks + STACK);
    require(wait_for(&race_started, 1, 5000), "the waiter on the middle page starts");
    pause_ms(300);
    require(!munmap(three + PAGE, PAGE), "munmap of the middle page");
    pass("partial_unmap_middle");
    require(wait_for(&race_done, 1, 5000) && race_rc == 0, "the waiter on the removed page returns");
    require(!capstone_virtual_thread_join(waiter), "join the waiter");
    pass("partial_unmap_wakes_waiter");

    require(three[0] == 0x31 && three[PAGE - 1] == 0x31 && three[2 * PAGE] == 0x31 &&
            three[3 * PAGE - 1] == 0x31, "the outer pages keep their data");
    three[1] = 0x32;
    three[2 * PAGE + 1] = 0x33;
    require(three[1] == 0x32 && three[2 * PAGE + 1] == 0x33, "the outer pages take writes");
    pass("outer_pages_work");

    for (unsigned i = 0; i < 16; ++i) {
        char *q = map(PAGE);
        require(cursor(q) + PAGE <= base || cursor(q) >= base + 4 * PAGE,
                "a new mapping lies outside the block");
    }
    pass("no_mapping_in_hole");

    require(!munmap(three + PAGE, PAGE) && !munmap(three + PAGE, PAGE),
            "munmap of the hole again, twice");
    require(three[0] == 0x31 && three[1] == 0x32 && three[2 * PAGE + 1] == 0x33,
            "the outer pages after the repeated munmaps");
    pass("hole_unmap_repeats");

    errno = 0;
    require(mprotect(three + PAGE, PAGE, PROT_READ) == -1 && errno == ENOMEM, "mprotect of the hole");
    errno = 0;
    require(madvise(three + PAGE, PAGE, MADV_DONTNEED) == -1 && errno == ENOMEM, "madvise of the hole");
    require(futex_wait_ms(three + PAGE + 64, 0) == -EFAULT, "a futex wait in the hole");
    require(!registered(base + PAGE) && registered(base) && registered(base + 2 * PAGE),
            "a wake answers EFAULT in the hole only");
    pass("hole_requests_refused");

    printf("ROOT_HOLE_READ address=0x%lx\n", base + PAGE);
    fflush(stdout);
    char seen = *(volatile char *)(three + PAGE + 5);
    printf("ROOT_HOLE_SURVIVED value=0x%x\n", seen);
    fflush(stdout);
    return 0;
}

/* partial-all (ends with a fault): the pages of a block go one by one. Unmapping a removed page again
 * changes nothing, a PROT_NONE page still counts as mapped, and the munmap of the last page that is
 * still mapped ends the block with CREVOKE on the reference it passes: every copy of that reference,
 * here one stored in another mapping, must then fault with 25 at the reused address. */
static int mode_partial_all(void)
{
    (void)grant();
    char **saved = (char **)map(PAGE);
    char *four = map(4 * PAGE);
    unsigned long base = cursor(four);
    memset(four, 0x41, 4 * PAGE);
    *saved = four;

    require(!munmap(four + PAGE, PAGE), "munmap of the second page");
    for (unsigned i = 0; i < 3; ++i)
        require(!munmap(four + PAGE, PAGE), "munmap of the second page again");
    printf("ROOT_PARTIAL repeats_done\n");
    fflush(stdout);
    require(four[0] == 0x41 && four[3 * PAGE] == 0x41, "the live pages after the repeated munmaps");
    four[2 * PAGE] = 0x42;
    require(four[2 * PAGE] == 0x42, "a live page takes a write");
    pass("hole_unmap_idempotent");

    require(!mprotect(four, PAGE, PROT_NONE), "make the first page inaccessible");
    require(!munmap(four + 2 * PAGE, PAGE) && !munmap(four + 3 * PAGE, PAGE),
            "munmap of the third and the fourth page");
    require(registered(base), "the block lives while its inaccessible page is mapped");
    pass("prot_none_page_counts");
    require(!mprotect(four, PAGE, PROT_READ | PROT_WRITE), "restore the first page");
    require(four[0] == 0x41, "the first page keeps its data");
    pass("count_exact_with_mprotect");

    require(!munmap(four, PAGE), "munmap of the last page that is still mapped");
    require(!registered(base), "the block has ended");
    pass("last_page_ends_block");
    stale = *saved;
    stale_read_bytes(base, 4 * PAGE);
    return 0;
}

/* partial-narrow (exit 0): a reference acts only on the pages its bounds cover, whatever its cursor. A
 * reference narrowed to the middle page removes that page and no other; an object over the whole block
 * and an integer address remove none. */
static int mode_partial_narrow(void)
{
    (void)grant();
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x21, 3 * PAGE);
    char *narrow = __builtin_capstone_cap_shrink(three, base + PAGE, base + 2 * PAGE);
    if (root_grants) {
        require(refused(munmap(narrow, 2 * PAGE)),
                "munmap of two pages through a reference narrowed to the middle page");
        require(three[PAGE] == 0x21 && three[2 * PAGE] == 0x21, "the pages after the refusal");
        pass("partial_narrowed_beyond_refused");
        char *object = capstone_cap_derive(three, 0, 3 * PAGE);
        require(refused(munmap(object + PAGE, PAGE)), "munmap through an object over the whole block");
        require(three[PAGE] == 0x21 && object[PAGE] == 0x21, "the page after the refusal");
        pass("partial_nested_object_refused");
        require(refused(munmap(untagged(base + PAGE), PAGE)), "munmap with an integer address");
        pass("partial_address_refused");
        capstone_cap_release(object);
    } else {
        skip("partial_narrowed_beyond_refused", BY_ADDRESS);
        skip("partial_nested_object_refused", BY_ADDRESS);
        skip("partial_address_refused", BY_ADDRESS);
    }
    require(refused(munmap(three + 2 * PAGE, 2 * PAGE)), "munmap of a range beyond the block");
    require(three[2 * PAGE] == 0x21, "the last page after the refusal");
    pass("partial_beyond_block_refused");
    require(!munmap(narrow, PAGE), "munmap of the middle page through the reference narrowed to it");
    require(three[0] == 0x21 && three[2 * PAGE] == 0x21 && registered(base) && !registered(base + PAGE),
            "the outer pages and the block after the removal");
    pass("partial_narrowed_removes_own_page");
    require(!munmap(three + 2 * PAGE, PAGE), "munmap through the block's reference moved to the last page");
    require(three[0] == 0x21 && registered(base), "the first page and the block");
    pass("partial_through_moved_reference");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block with its holes");
    require(!registered(base), "the block has ended");
    pass("whole_unmap_with_hole");
    return 0;
}

/* partial-fixed (exit 0): MAP_FIXED inside a block refills pages of the same object with zeros, while
 * the program derived no object below the block's reference. A reference stored in a removed page is
 * gone after the refill. */
static int mode_partial_fixed(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x51, 3 * PAGE);
    *(char **)(three + PAGE + 64) = target;
    require(type_of(*(char **)(three + PAGE + 64)) == 1, "a reference stored in the middle page");
    printf("ROOT_PARTIAL_FIXED base=0x%lx\n", base);
    fflush(stdout);
    require(!munmap(three + PAGE, PAGE), "munmap of the middle page");
    if (!root_grants) {
        errno = 0;
        require(mmap(three + PAGE, PAGE, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) == MAP_FAILED &&
                errno == EOPNOTSUPP, "MAP_FIXED without root grants");
        skip("fixed_refill_reads_zero", "root_grants=0 cannot read the reference's objects");
        skip("removed_page_tags_cleared", "follows the refill");
        skip("fixed_replaces_mapped_page", "follows the refill");
        skip("fixed_refused_with_object", "follows the refill");
        skip("fixed_after_object_released", "follows the refill");
        return 0;
    }
    char *q = mmap(three + PAGE, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    require(q != MAP_FAILED && cursor(q) == base + PAGE && type_of(q) == 1,
            "MAP_FIXED into the hole returns the reference");
    unsigned long nonzero = 0;
    for (unsigned long i = 0; i < PAGE; ++i)
        nonzero += three[PAGE + i] != 0;
    printf("ROOT_PARTIAL_FIXED refill nonzero=%lu stored_type=%lu\n", nonzero,
           type_of(*(char **)(three + PAGE + 64)));
    fflush(stdout);
    require(!nonzero, "the refilled page reads zero");
    three[PAGE + 1] = 0x52;
    require(three[PAGE + 1] == 0x52 && three[0] == 0x51 && three[2 * PAGE] == 0x51,
            "the refilled page takes a write, the outer pages keep their data");
    pass("fixed_refill_reads_zero");
    require(type_of(*(char **)(three + PAGE + 64)) == 7, "the reference stored before the removal is gone");
    pass("removed_page_tags_cleared");

    require(mmap(three, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0) ==
            three && three[0] == 0 && three[PAGE + 1] == 0x52, "MAP_FIXED over a mapped page");
    pass("fixed_replaces_mapped_page");

    char *object = capstone_cap_derive(three, 2 * PAGE, 64);
    object[0] = 0x53;
    require(!munmap(three + PAGE, PAGE), "munmap of the middle page again");
    errno = 0;
    require(mmap(three + PAGE, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                 -1, 0) == MAP_FAILED && errno == EINVAL, "MAP_FIXED while an object exists below");
    require(object[0] == 0x53 && three[2 * PAGE] == 0x53, "the object after the refusal");
    pass("fixed_refused_with_object");
    capstone_cap_release(object);
    require(mmap(three + PAGE, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                 -1, 0) == three + PAGE && three[PAGE] == 0, "MAP_FIXED after the object ended");
    three[PAGE] = 0x54;
    require(three[PAGE] == 0x54, "the page refilled again takes a write");
    pass("fixed_after_object_released");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block");
    return 0;
}

/* partial-mremap (exit 0): mremap shrinks in place, and the references stay valid; growth into the
 * block's own removed tail is a refill, under the rule of MAP_FIXED (stage 4f): refused while an object
 * is derived below the block's reference, in place once it ended; beyond the block's extent it answers
 * ENOMEM without MREMAP_MAYMOVE and moves with it. Without root grants growth in place is never
 * possible (the refill rule needs the reference) and the range moves. */
static int mode_partial_mremap(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *p = map(3 * PAGE);
    unsigned long base = cursor(p);
    memset(p, 0x61, 3 * PAGE);
    *(char **)(p + 64) = target;
    char *object = capstone_cap_derive(p, 128, 64);

    char *r = mremap(p, 3 * PAGE, PAGE, 0);
    require(r != MAP_FAILED && cursor(r) == base, "mremap shrinks in place");
    require(p[0] == 0x61 && object[0] == 0x61 && (*(char **)(p + 64))[1] == 0x71,
            "the references after the shrink");
    require(registered(base) && !registered(base + PAGE), "the tail is removed");
    pass("mremap_shrink_in_place");

    errno = 0;
    require(mremap(p, PAGE, 2 * PAGE, 0) == MAP_FAILED && errno == ENOMEM,
            "mremap grows while an object exists below the block");
    require(p[0] == 0x61 && object[0] == 0x61 && !registered(base + PAGE),
            "the block after the refused growth");
    if (root_grants)
        pass("mremap_grow_refused_with_object");
    else
        skip("mremap_grow_refused_with_object", "root_grants=0 refuses every growth in place");
    capstone_cap_release(object);

    char *q;
    if (root_grants) {
        r = mremap(p, PAGE, 2 * PAGE, 0);
        require(r == p && p[PAGE] == 0 && p[2 * PAGE - 1] == 0, "mremap grows by one page in place");
        r = mremap(p, 2 * PAGE, 3 * PAGE, 0);
        require(r == p && p[2 * PAGE] == 0 && p[0] == 0x61, "mremap grows into the rest of the tail");
        p[3 * PAGE - 1] = 0x62;
        require(p[3 * PAGE - 1] == 0x62, "the grown pages take writes");
        pass("mremap_grow_into_tail");

        errno = 0;
        require(mremap(p, 3 * PAGE, 4 * PAGE, 0) == MAP_FAILED && errno == ENOMEM,
                "mremap beyond the block without MREMAP_MAYMOVE");
        require(p[0] == 0x61 && p[3 * PAGE - 1] == 0x62, "the block after the refusal");
        pass("mremap_grow_beyond_enomem");

        q = mremap(p, 3 * PAGE, 5 * PAGE, MREMAP_MAYMOVE);
        require(q != MAP_FAILED && cursor(q) != base && q[0] == 0x61 && q[3 * PAGE - 1] == 0x62 &&
                (*(char **)(q + 64))[1] == 0x71 && q[4 * PAGE] == 0, "mremap beyond the block moves");
    } else {
        skip("mremap_grow_into_tail", "root_grants=0 refuses every growth in place");
        errno = 0;
        require(mremap(p, PAGE, 4 * PAGE, 0) == MAP_FAILED && errno == ENOMEM,
                "mremap beyond the block without MREMAP_MAYMOVE");
        require(p[0] == 0x61, "the block after the refusal");
        pass("mremap_grow_beyond_enomem");

        q = mremap(p, PAGE, 5 * PAGE, MREMAP_MAYMOVE);
        require(q != MAP_FAILED && cursor(q) != base && q[0] == 0x61 &&
                (*(char **)(q + 64))[1] == 0x71 && q[PAGE] == 0, "mremap beyond the block moves");
    }
    require(!registered(base), "the old block has ended");
    pass("mremap_grow_beyond_moves");
    return 0;
}

/* mremap-validate (exit 0, stage 4f): every path of mremap validates the whole old range before it
 * changes anything: the range lies inside one block, every page of it is mapped, and the reference is
 * the block's own (with root grants). */
static int mode_mremap_validate(void)
{
    (void)grant();
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x41, 3 * PAGE);
    require(!munmap(three, 2 * PAGE), "munmap of the first two pages");
    errno = 0;
    char *r = mremap(three, 3 * PAGE, PAGE, 0);
    printf("ROOT_MREMAP_VALIDATE shrink_over_hole failed=%d errno=%d registered=%d\n",
           r == MAP_FAILED, errno, registered(base + 2 * PAGE));
    fflush(stdout);
    require(r == MAP_FAILED && errno == EFAULT, "mremap shrinks an old range with removed pages");
    require(registered(base + 2 * PAGE) && three[2 * PAGE] == 0x41, "the block keeps its last page");
    three[2 * PAGE + 1] = 0x42;
    require(three[2 * PAGE + 1] == 0x42, "the last page takes a write");
    pass("mremap_shrink_over_hole_refused");
    require(!munmap(three + 2 * PAGE, PAGE) && !registered(base), "munmap of the last page ends the block");

    char *q = map(3 * PAGE);
    unsigned long qbase = cursor(q);
    memset(q, 0x43, 3 * PAGE);
    require(!munmap(q + 2 * PAGE, PAGE) && !munmap(q, PAGE), "munmap of the first and the last page");
    errno = 0;
    require(mremap(q, 2 * PAGE, 3 * PAGE, 0) == MAP_FAILED && errno == EFAULT,
            "mremap grows an old range with a removed page");
    errno = 0;
    require(mremap(q, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE) == MAP_FAILED && errno == EFAULT,
            "mremap grows an old range with a removed page, MREMAP_MAYMOVE");
    require(q[PAGE] == 0x43 && !registered(qbase) && registered(qbase + PAGE) &&
            !registered(qbase + 2 * PAGE), "the block after the refused growth");
    pass("mremap_grow_over_hole_refused");
    require(!munmap(q, 3 * PAGE) && !registered(qbase + PAGE), "munmap of the whole block");

    char *two = map(2 * PAGE);
    unsigned long tbase = cursor(two);
    memset(two, 0x44, 2 * PAGE);
    require(!munmap(two + PAGE, PAGE), "munmap of the second page");
    char *below = at_address(two, tbase - PAGE);
    printf("ROOT_MREMAP_VALIDATE below type=%lu cursor=0x%lx\n", type_of(below), cursor(below));
    fflush(stdout);
    errno = 0;
    require(mremap(below, 2 * PAGE, 3 * PAGE, 0) == MAP_FAILED && errno == EINVAL,
            "mremap grows through a cursor below the block");
    errno = 0;
    require(mremap(below, 2 * PAGE, PAGE, 0) == MAP_FAILED && errno == EINVAL,
            "mremap shrinks through a cursor below the block");
    require(two[0] == 0x44 && registered(tbase) && !registered(tbase + PAGE),
            "the block after the refusals");
    pass("mremap_cursor_outside_refused");
    require(!munmap(two, 2 * PAGE), "munmap of the whole block");
    return 0;
}

/* A number read from a file through the delegated calls, or -1. */
static long read_number(const char *path)
{
    char text[64];
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;
    long n = read(fd, text, sizeof text - 1);
    close(fd);
    if (n <= 0)
        return -1;
    text[n] = 0;
    return strtol(text, NULL, 10);
}

/* Write a number into a file through the delegated calls: 0, or -1. */
static int write_number(const char *path, long value)
{
    char text[32];
    int fd = open(path, O_WRONLY);
    if (fd < 0)
        return -1;
    int n = snprintf(text, sizeof text, "%ld\n", value);
    long written = write(fd, text, (size_t)n);
    close(fd);
    return written == n ? 0 : -1;
}

/* The launcher process's mappings: the lines of /proc/self/maps, which the launcher opens. */
static long map_count(void)
{
    char text[4096];
    long lines = 0, n;
    int fd = open("/proc/self/maps", O_RDONLY);
    if (fd < 0)
        return -1;
    while ((n = read(fd, text, sizeof text)) > 0)
        for (long i = 0; i < n; ++i)
            lines += text[i] == '\n';
    close(fd);
    return n < 0 ? -1 : lines;
}

/* pages-enomem (exit 0, stage 4f): Linux refuses the first step of a page operation, the protection
 * change, because no VMA can be split (vm.max_map_count lowered to the map count): the program gets
 * ENOMEM and nothing has changed. */
static int mode_pages_enomem(void)
{
    (void)grant();
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x51, 3 * PAGE);
    /* The launcher's byte per page for this block now, so the requests under the limit allocate
     * nothing; an mprotect to the same protection changes no VMA. */
    require(!mprotect(three, PAGE, PROT_READ | PROT_WRITE), "mprotect of the first page to its protection");
    long limit = read_number("/proc/sys/vm/max_map_count");
    long count = map_count();
    printf("ROOT_PAGES_ENOMEM base=0x%lx max_map_count=%ld map_count=%ld\n", base, limit, count);
    fflush(stdout);
    require(limit > 0 && count > 0, "read the map count and its limit");
    require(!write_number("/proc/sys/vm/max_map_count", count), "lower vm.max_map_count to the map count");
    errno = 0;
    int unmapped = munmap(three + PAGE, PAGE);
    int unmap_errno = errno;
    void *fixed = MAP_FAILED;
    int fixed_errno = 0;
    if (root_grants) {
        errno = 0;
        fixed = mmap(three + PAGE, PAGE, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        fixed_errno = errno;
    }
    int restored = write_number("/proc/sys/vm/max_map_count", limit);
    printf("ROOT_PAGES_ENOMEM unmap=%d errno=%d fixed=%d errno=%d restored=%d\n", unmapped, unmap_errno,
           fixed != MAP_FAILED, fixed_errno, restored);
    fflush(stdout);
    require(!restored, "restore vm.max_map_count");
    require(unmapped == -1 && unmap_errno == ENOMEM, "the partial munmap answers ENOMEM at the limit");
    require(registered(base + PAGE) && three[PAGE] == 0x51 && three[2 * PAGE - 1] == 0x51,
            "the page keeps its data after the refused munmap");
    three[PAGE + 1] = 0x52;
    require(three[PAGE + 1] == 0x52, "the page takes a write after the refused munmap");
    pass("partial_unmap_enomem_changes_nothing");
    if (root_grants) {
        require(fixed == MAP_FAILED && fixed_errno == ENOMEM, "MAP_FIXED answers ENOMEM at the limit");
        require(three[PAGE] == 0x51 && three[PAGE + 1] == 0x52,
                "the page keeps its data after the refused MAP_FIXED");
        three[PAGE + 2] = 0x53;
        require(three[PAGE + 2] == 0x53, "the page stays writable after the refused MAP_FIXED");
        pass("fixed_enomem_changes_nothing");
    } else {
        skip("fixed_enomem_changes_nothing", "root_grants=0 refuses MAP_FIXED before Linux acts");
    }
    require(!munmap(three + PAGE, PAGE) && !registered(base + PAGE) && three[0] == 0x51 &&
            three[2 * PAGE] == 0x51, "the partial munmap below the limit");
    pass("partial_unmap_after_limit");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block");
    return 0;
}

/* pages-tagged (exit 0, stage 4f): the partial munmap of a page that holds a reference; the runner
 * reads the module's census and the kernel's count of tagged frames it cleared at their free. */
static int mode_pages_tagged(void)
{
    (void)grant();
    char *target = map(PAGE);
    target[1] = 0x71;
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x61, 3 * PAGE);
    *(char **)(three + PAGE + 64) = target;
    require(type_of(*(char **)(three + PAGE + 64)) == 1, "a reference stored in the middle page");
    long before = read_number("/sys/module/captag/parameters/cleared_pages");
    printf("ROOT_PAGES_TAGGED base=0x%lx\n", base);
    fflush(stdout);
    require(!munmap(three + PAGE, PAGE), "munmap of the tagged middle page");
    long after = read_number("/sys/module/captag/parameters/cleared_pages");
    printf("ROOT_PAGES_TAGGED cleared_before=%ld cleared_after=%ld\n", before, after);
    fflush(stdout);
    require(three[0] == 0x61 && three[2 * PAGE] == 0x61 && !registered(base + PAGE), "the outer pages");
    pass("tagged_page_removed");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block");
    return 0;
}

/* futex-stale (exit 0, stage 4f, root grants): two waiters at one address, the first on an object the
 * program then frees, the second on a new object there. */
static int *volatile stale_words[2];
static volatile int stale_values[2], stale_started[2], stale_done[2];
static volatile long stale_rc[2] = {1, 1};

static void stale_wait(int i)
{
    __atomic_store_n(&stale_started[i], 1, __ATOMIC_RELEASE);
    stale_rc[i] = __capstone_vm_futex(stale_words[i], 0 /* FUTEX_WAIT */, stale_values[i], 8, 0, 0);
    __atomic_store_n(&stale_done[i], 1, __ATOMIC_RELEASE);
}

static void *stale_entry0(void *arg) { stale_wait(0); return arg; }
static void *stale_entry1(void *arg) { stale_wait(1); return arg; }

static int mode_futex_stale(void)
{
    (void)grant();
    if (!root_grants) {
        skip("futex_stale_waiter_not_counted", "root_grants=0 keeps no lifetime of a futex word");
        return 0;
    }
    char *stacks = map(3 * STACK);
    char *page = map(PAGE);
    unsigned long word = cursor(page) + 64;
    int *old = capstone_cap_derive(page, 64, 16);
    *old = 1;
    stale_words[0] = old;
    stale_values[0] = 1;
    long w0 = context(stale_entry0, stacks + STACK);
    require(wait_for(&stale_started[0], 1, 5000), "the first waiter starts");
    pause_ms(500);
    capstone_cap_release(old);
    int *fresh = capstone_cap_derive(page, 64, 16);
    require(cursor(fresh) == word, "the new object lies at the old one's address");
    *fresh = 2;
    stale_words[1] = fresh;
    stale_values[1] = 2;
    long w1 = context(stale_entry1, stacks + 2 * STACK);
    require(wait_for(&stale_started[1], 1, 5000), "the second waiter starts");
    pause_ms(500);
    long woken = __capstone_vm_futex(untagged(word), 1 /* FUTEX_WAKE */, 1, -1, 0, 0);
    int live_done = wait_for(&stale_done[1], 1, 3000);
    int stale_done_now = wait_for(&stale_done[0], 1, 3000);
    printf("ROOT_FUTEX_STALE woken=%ld live_done=%d live_rc=%ld stale_done=%d stale_rc=%ld\n", woken,
           live_done, stale_rc[1], stale_done_now, stale_rc[0]);
    fflush(stdout);
    require(live_done && stale_rc[1] == 0 && woken == 1,
            "a wake of one reaches the live waiter, the stale one does not count");
    require(stale_done_now && stale_rc[0] == 0, "the stale waiter is woken too");
    require(!capstone_virtual_thread_join(w0) && !capstone_virtual_thread_join(w1), "join the waiters");
    pass("futex_stale_waiter_not_counted");
    return 0;
}

/* The hook modes of stage 4f: a context acts while a page operation of the main thread waits in one of
 * the hooks launcher's windows (CAPSTONE_VM_TEST_PAGES_DELAY_MS after the check,
 * CAPSTONE_VM_TEST_PAGES_LATE_DELAY_MS before the commit). The context spins (no service) so that it
 * runs whenever the module lets the program run. */
static char *volatile derive_parent, *volatile derive_object;
static volatile unsigned long derive_offset, derive_spin;
static volatile int derive_value = -1, derive_done;

static void *derive_entry(void *arg)
{
    spin_ms(derive_spin);
    char *o = capstone_cap_derive(derive_parent, derive_offset, 64);
    derive_value = (unsigned char)o[0];
    derive_object = o;
    __atomic_store_n(&derive_done, 1, __ATOMIC_RELEASE);
    return arg;
}

/* pages-derive: a context derives an object over a mapped page while MAP_FIXED of that page is open. */
static int mode_pages_derive(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_derive_after_refill", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x5a, 3 * PAGE);
    derive_parent = three;
    derive_offset = 2 * PAGE;
    derive_spin = 500;
    long deriver = context(derive_entry, stacks + STACK);
    errno = 0;
    char *q = mmap(three + PAGE, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    int fixed_errno = errno;
    require(wait_for(&derive_done, 1, 15000), "the context derives its object");
    require(!capstone_virtual_thread_join(deriver), "join the context");
    printf("ROOT_PAGES_DERIVE fixed=%d errno=%d value_at_derive=0x%x\n", q != MAP_FAILED, fixed_errno,
           derive_value);
    fflush(stdout);
    if (q != MAP_FAILED) {
        int now = (unsigned char)derive_object[0];
        printf("ROOT_PAGES_DERIVE value_now=0x%x\n", now);
        fflush(stdout);
        require(now == derive_value, "no live object sees its memory replaced");
        require(now == 0 && three[PAGE] == 0 && three[0] == 0x5a, "the object was derived after the refill");
        pass("pages_derive_after_refill");
        return 0;
    }
    require(fixed_errno == EINVAL && !registered(base), "the refill failed closed: the block has ended");
    pass("pages_derive_fails_closed");
    stale = derive_object;
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
    return 0;
}

static char *volatile touch_at;
static volatile unsigned long touch_spin;
static volatile int touch_value, touch_done;

static void *touch_entry(void *arg)
{
    spin_ms(touch_spin);
    *(volatile char *)touch_at = (char)touch_value;
    __atomic_store_n(&touch_done, 1, __ATOMIC_RELEASE);
    return arg;
}

static long toucher(char *at, int value, unsigned long spin, char *stack_top)
{
    touch_at = at;
    touch_value = value;
    touch_spin = spin;
    touch_done = 0;
    return context(touch_entry, stack_top);
}

/* pages-fault: a context stores into a removed page while MAP_FIXED refills it (the window after the
 * check). */
static int mode_pages_fault(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_fault_waits_then_proceeds", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x31, 3 * PAGE);
    require(!munmap(three + PAGE, PAGE), "munmap of the middle page");
    printf("ROOT_PAGES_FAULT base=0x%lx\n", base);
    fflush(stdout);
    long t = toucher(three + PAGE + 8, 0x77, 500, stacks + STACK);
    char *q = mmap(three + PAGE, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    require(q == three + PAGE, "MAP_FIXED refills the middle page");
    require(wait_for(&touch_done, 1, 15000), "the context's store completes");
    require(!capstone_virtual_thread_join(t), "join the context");
    require(three[PAGE + 8] == 0x77 && three[PAGE] == 0 && three[PAGE + 9] == 0 && three[0] == 0x31,
            "the store landed in the refilled page, after the refill");
    pass("pages_fault_waits_then_proceeds");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block");
    return 0;
}

/* pages-admit: a context stores into an untouched page while MAP_FIXED of it waits before its commit
 * (the window after the Linux steps): the scenario in which stage 4e admitted a page of the new range
 * and then unpinned it at the commit while it stayed mapped. */
static int mode_pages_admit(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_admit_store_after_refill", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    printf("ROOT_PAGES_ADMIT base=0x%lx\n", base);
    fflush(stdout);
    long t = toucher(three + 2 * PAGE + 8, 0x88, 500, stacks + STACK);
    char *q = mmap(three + 2 * PAGE, PAGE, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    require(q == three + 2 * PAGE, "MAP_FIXED over the untouched third page");
    require(wait_for(&touch_done, 1, 15000), "the context's store completes");
    require(!capstone_virtual_thread_join(t), "join the context");
    require(three[2 * PAGE + 8] == (char)0x88 && three[2 * PAGE] == 0, "the store landed in the page");
    pass("pages_admit_store_after_refill");
    require(!munmap(three, 3 * PAGE), "munmap of the whole block");
    return 0;
}

/* release-fail (ends with a fault 25): the hooks launcher's munmap of every released block fails. */
static int mode_release_fail(void)
{
    (void)grant();
    char *page = map(PAGE);
    unsigned long base = cursor(page);
    page[1] = 0x21;
    printf("ROOT_RELEASE_FAIL base=0x%lx\n", base);
    fflush(stdout);
    require(!munmap(page, PAGE), "munmap answers 0 although Linux keeps the range");
    require(!registered(base), "the launcher dropped the registration");
    for (unsigned i = 0; i < 16; ++i) {
        char *q = map(PAGE);
        require(cursor(q) + PAGE <= base || cursor(q) >= base + PAGE, "a new mapping outside the kept range");
    }
    pass("release_failure_keeps_range_unused");
    stale = page;
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
    return 0;
}

/* pages-closed (exit 0, block A): a page operation's commit fails closed while its retire preparation
 * fails (the hooks module: test_pages_run lets a context derive an object below the block inside the
 * window, test_close_unheld unholds the block's first page and fails the preparation). The first page
 * holds a reference, so its frame is tagged; the module must clear it in place before Linux gets it. */
static int mode_pages_closed(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_closed_fails_closed", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *target = map(PAGE);
    target[1] = 0x71;
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x5a, 3 * PAGE);
    *(char **)(three + 64) = target;
    require(type_of(*(char **)(three + 64)) == 1, "a reference stored in the first page");
    long before = read_number("/sys/module/captag/parameters/cleared_pages");
    printf("ROOT_PAGES_CLOSED base=0x%lx\n", base);
    fflush(stdout);
    derive_parent = three;
    derive_offset = 2 * PAGE;
    derive_spin = 500;
    long deriver = context(derive_entry, stacks + STACK);
    errno = 0;
    char *q = mmap(three + PAGE, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    int fixed_errno = errno;
    require(wait_for(&derive_done, 1, 15000), "the context derives its object");
    require(!capstone_virtual_thread_join(deriver), "join the context");
    long after = read_number("/sys/module/captag/parameters/cleared_pages");
    printf("ROOT_PAGES_CLOSED fixed=%d errno=%d registered=%d cleared_before=%ld cleared_after=%ld\n",
           q != MAP_FAILED, fixed_errno, registered(base), before, after);
    fflush(stdout);
    require(q == MAP_FAILED && fixed_errno == EINVAL && !registered(base),
            "the refill failed closed: the block has ended");
    pass("pages_closed_fails_closed");
    return 0;
}

/* pages-discard (ends with a fault 25, block A): the hooks launcher's discard of a refill discards the
 * first page of the range and then fails (CAPSTONE_VM_TEST_DISCARD_FAIL): the block must end, never
 * stay as if nothing had changed. */
static int mode_pages_discard(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_discard_fails_closed", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x5a, 3 * PAGE);
    errno = 0;
    char *q = mmap(three + PAGE, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    int fixed_errno = errno, live = registered(base);
    /* With stage 4f's abort the block stays live although its first refilled page was discarded, and the
     * module still holds that page's old frame; the program does not touch the range then. */
    printf("ROOT_PAGES_DISCARD fixed=%d errno=%d registered=%d\n", q != MAP_FAILED, fixed_errno, live);
    fflush(stdout);
    require(q == MAP_FAILED && fixed_errno == EINVAL && !live, "the refill failed closed: the block has ended");
    pass("pages_discard_fails_closed");
    stale = three;
    char seen = *(volatile char *)stale;
    printf("ROOT_STALE_READ value=0x%x\n", seen);
    fflush(stdout);
    return 0;
}

/* ---- Block A, F4 part 2: held extents, the top page, growth and the failure result ---- */

/* pages-unclear (block A, F4): as pages-closed, with test_close_unheld=2, so the in-place clearing
 * leaves the unheld tagged page as it is. The commit still fails closed, and the launcher keeps the
 * block's whole extent mapped PROT_NONE instead of giving it back to Linux. */
static int mode_pages_unclear(void)
{
    (void)grant();
    if (!root_grants) {
        skip("pages_unclear_extent_held", "root_grants=0 refuses MAP_FIXED before Linux acts");
        return 0;
    }
    spin_calibrate();
    char *stacks = map(2 * STACK);
    char *target = map(PAGE);
    target[1] = 0x71;
    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x5a, 3 * PAGE);
    *(char **)(three + 64) = target;
    require(type_of(*(char **)(three + 64)) == 1, "a reference stored in the first page");
    long unclear = read_number("/sys/module/capstone_vm/parameters/closed_unclear");
    printf("ROOT_PAGES_UNCLEAR base=0x%lx\n", base);
    fflush(stdout);
    derive_parent = three;
    derive_offset = 2 * PAGE;
    derive_spin = 500;
    long deriver = context(derive_entry, stacks + STACK);
    errno = 0;
    char *q = mmap(three + PAGE, 2 * PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                   -1, 0);
    int fixed_errno = errno;
    require(wait_for(&derive_done, 1, 15000), "the context derives its object");
    require(!capstone_virtual_thread_join(deriver), "join the context");
    long unclear_after = read_number("/sys/module/capstone_vm/parameters/closed_unclear");
    int held = protected_as(base, "---") && protected_as(base + 2 * PAGE, "---");
    printf("ROOT_PAGES_UNCLEAR fixed=%d errno=%d registered=%d held=%d unclear_before=%ld unclear_after=%ld\n",
           q != MAP_FAILED, fixed_errno, registered(base), held, unclear, unclear_after);
    fflush(stdout);
    require(q == MAP_FAILED && fixed_errno == EINVAL && !registered(base),
            "the refill failed closed: the block has ended");
    require(held, "the ended block's extent stays mapped PROT_NONE");
    require(unclear >= 0 && unclear_after == unclear + 1, "the module counts the commit in closed_unclear");
    pass("pages_unclear_extent_held");
    return 0;
}

/* top-page (block A, F4; hooks launcher with CAPSTONE_VM_TEST_TOP_PAGE): the launcher's first mapping
 * service tries the topmost page below 2^38 first; the runner reads its line. The mapping the program
 * gets works and ends a page or more below 2^38. */
static int mode_top_page(void)
{
    (void)grant();
    char *page = map(PAGE);
    page[100] = 0x42;
    require(page[100] == 0x42 && cursor(page) + PAGE <= (1UL << 38) - PAGE,
            "the mapping works and ends a page or more below 2^38");
    pass("top_page_mapping_below");
    require(!munmap(page, PAGE), "munmap of the mapping");
    return 0;
}

/* The lifetime-id width of the module's build: its parameter ids (absent in the 30-bit build). */
static int sublet_format(void)
{
    return read_number("/sys/module/capstone_vm/parameters/ids") == 63;
}

static long module_number(const char *name)
{
    char path[96];
    snprintf(path, sizeof path, "/sys/module/capstone_vm/parameters/%s", name);
    return read_number(path);
}

/* An array of n pointers in a mapping of its own, made before the table fills. */
static void **pointer_array(unsigned long n)
{
    unsigned long bytes = (n * sizeof(void *) + PAGE - 1) & ~(PAGE - 1);
    void **array = (void **)map(bytes);
    memset(array, 0, bytes);
    return array;
}

#define GROW_MAPS 5000UL
#define GROW_OBJECTS 6000UL

/* grow-maps: more mappings than the first batch holds records for, one record each (the block; two
 * before the stage that dropped the reservation record M, when GROW_MAPS was 2,500). */
static int mode_grow_maps(void)
{
    (void)grant();
    if (!sublet_format()) {
        skip("grow_maps_through_grants", "the older formats: no growth word");
        return 0;
    }
    void **maps = pointer_array(GROW_MAPS);
    long growths = module_number("table_growths"), escapes = module_number("pressure_escapes");
    unsigned long done = 0;
    for (; done < GROW_MAPS; ++done) {
        char *p = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) break;
        p[8] = (char)done;
        maps[done] = p;
    }
    long growths_after = module_number("table_growths"), escapes_after = module_number("pressure_escapes");
    unsigned long kept = 0;
    for (unsigned long i = 0; i < done; ++i) {
        kept += ((char *)maps[i])[8] == (char)i;
        require(!munmap(maps[i], PAGE), "munmap of a grown mapping");
    }
    printf("ROOT_GROW_MAPS maps=%lu kept=%lu growths=%ld escapes=%ld\n", done, kept,
           growths_after - growths, escapes_after - escapes);
    fflush(stdout);
    require(done == GROW_MAPS && kept == GROW_MAPS, "every mmap succeeds and keeps its value");
    require(growths >= 0 && growths_after > growths, "the module grows the table for its own CDERIVEs");
    pass("grow_maps_through_grants");
    return 0;
}

/* grow-objects: more objects than the first batch holds, one record each. */
static int mode_grow_objects(void)
{
    (void)grant();
    if (!sublet_format()) {
        skip("grow_objects_through_escape", "the older formats: no growth word");
        return 0;
    }
    void **objects = pointer_array(GROW_OBJECTS);
    long growths = module_number("table_growths"), escapes = module_number("pressure_escapes");
    unsigned long done = 0, kept = 0;
    for (; done < GROW_OBJECTS; ++done) {
        long *o = malloc(48);
        if (!o) break;
        o[0] = (long)done;
        objects[done] = o;
    }
    long growths_after = module_number("table_growths"), escapes_after = module_number("pressure_escapes");
    for (unsigned long i = 0; i < done; ++i) {
        kept += ((long *)objects[i])[0] == (long)i;
        free(objects[i]);
    }
    printf("ROOT_GROW_OBJECTS objects=%lu kept=%lu growths=%ld escapes=%ld\n", done, kept,
           growths_after - growths, escapes_after - escapes);
    fflush(stdout);
    require(done == GROW_OBJECTS && kept == GROW_OBJECTS, "every malloc succeeds and keeps its value");
    require(growths >= 0 && growths_after > growths && escapes_after > escapes,
            "the table grows, through the pressure escape too");
    pass("grow_objects_through_escape");
    return 0;
}

/* malloc and realloc called through volatile pointers: the compiler treats the named functions as
 * builtins that touch only memory the program cannot see, so it would carry errno = 0 across the call
 * and the failure's ENOMEM would read as 0. */
static void *(*volatile try_malloc)(size_t) = malloc;
static void *(*volatile try_realloc)(void *, size_t) = realloc;

#define NOGROW_LIMIT 8000UL
/* At most this many pressure escapes while the table fills: one answer sets NOGROW. */
#define NOGROW_ESCAPES 4L

/* nogrow-map (node_max_pages=16): mmap until the table is full. Each mmap needs one record, the block's.
 * The test then makes exactly one record free whatever a mapping costs: malloc takes every record the
 * mmaps left, and the free of one object reserved before the fill gives back one. One mmap then fits and
 * the next one does not (the module of F4 part 2 needed two records, M and P, and answers ENOMEM there; a
 * munmap would free two records under that module, so it cannot be the test's one record). */
static int mode_nogrow_map(void)
{
    (void)grant();
    if (!sublet_format()) {
        skip("nogrow_map_enomem", "the older formats end the context at exhaustion");
        skip("nogrow_map_one_record", "the older formats end the context at exhaustion");
        skip("nogrow_map_recovers", "the older formats end the context at exhaustion");
        return 0;
    }
    void **maps = pointer_array(NOGROW_LIMIT);
    long escapes = module_number("pressure_escapes"), sets = module_number("nogrow_sets");
    unsigned long done = 0, filled, rests = 0;
    int error = 0;
    void *rest[16];
    void *spare = try_malloc(48), *sibling = try_malloc(48);
    require(spare && sibling, "two objects before the table fills");
    for (; done < NOGROW_LIMIT; ++done) {
        errno = 0;
        char *p = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) { error = errno; break; }
        maps[done] = p;
    }
    filled = done;
    long escapes_after = module_number("pressure_escapes"), sets_after = module_number("nogrow_sets");
    require(done >= 300, "the first batch holds more than 300 mappings");

    while (rests < 16 && (rest[rests] = try_malloc(48)) != NULL)
        ++rests;
    require(rests < 16, "malloc takes the records that the mmaps left");
    free(spare);
    errno = 0;
    char *one = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int one_errno = errno, one_ok = one != MAP_FAILED, one_reads = 0;
    if (one_ok) {
        one[8] = 0x55;
        one_reads = one[8] == 0x55;
        maps[done++] = one;
    }
    errno = 0;
    char *extra = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int extra_errno = errno, full_again = extra == MAP_FAILED;
    if (!full_again)
        maps[done++] = extra;

    unsigned long keep = done - 300;
    for (unsigned long i = keep; i < done; ++i)
        require(!munmap(maps[i], PAGE), "munmap at exhaustion");
    char *again = mmap(NULL, PAGE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int recovered = again != MAP_FAILED;
    if (recovered) { again[0] = 3; require(!munmap(again, PAGE), "munmap after recovery"); }
    long clears = module_number("nogrow_clears");
    for (unsigned long i = 0; i < keep; ++i)
        require(!munmap(maps[i], PAGE), "munmap of the rest");
    printf("ROOT_NOGROW_MAP maps=%lu errno=%d escapes=%ld nogrow_sets=%ld recovered=%d nogrow_clears=%ld\n",
           filled, error, escapes_after - escapes, sets_after - sets, recovered, clears);
    for (unsigned long i = 0; i < rests; ++i)
        free(rest[i]);
    free(sibling);
    printf("ROOT_NOGROW_ONE rests=%lu one=%d one_errno=%d one_reads=%d extra=%d extra_errno=%d\n", rests,
           one_ok, one_errno, one_reads, !full_again, extra_errno);
    fflush(stdout);
    require(filled < NOGROW_LIMIT && error == ENOMEM, "mmap answers ENOMEM once the table is full");
    require(escapes_after - escapes <= NOGROW_ESCAPES && sets_after > sets,
            "NOGROW is set, without a loop of escapes");
    pass("nogrow_map_enomem");
    require(one_ok && one_reads, "an mmap fits into the one record that the free gave back");
    require(full_again && extra_errno == ENOMEM, "the next mmap answers ENOMEM: the table is full again");
    pass("nogrow_map_one_record");
    require(recovered, "an mmap succeeds again after munmaps");
    pass("nogrow_map_recovers");
    return 0;
}

/* nogrow-malloc (node_max_pages=16): malloc until the table is full. */
static int mode_nogrow_malloc(void)
{
    (void)grant();
    if (!sublet_format()) {
        skip("nogrow_malloc_null", "the older formats end the context at exhaustion");
        skip("nogrow_malloc_recovers", "the older formats end the context at exhaustion");
        return 0;
    }
    void **objects = pointer_array(NOGROW_LIMIT);
    long escapes = module_number("pressure_escapes"), sets = module_number("nogrow_sets");
    unsigned long done = 0;
    int error = 0;
    for (; done < NOGROW_LIMIT; ++done) {
        errno = 0;
        long *o = try_malloc(48);
        if (!o) { error = errno; break; }
        o[0] = (long)done;
        objects[done] = o;
    }
    long escapes_after = module_number("pressure_escapes"), sets_after = module_number("nogrow_sets");
    unsigned long kept = 0;
    for (unsigned long i = done / 2; i < done; ++i) {
        kept += ((long *)objects[i])[0] == (long)i;
        free(objects[i]);
    }
    long *again = malloc(48);
    int recovered = again != NULL;
    free(again);
    for (unsigned long i = 0; i < done / 2; ++i) {
        kept += ((long *)objects[i])[0] == (long)i;
        free(objects[i]);
    }
    printf("ROOT_NOGROW_MALLOC objects=%lu kept=%lu errno=%d escapes=%ld nogrow_sets=%ld recovered=%d\n",
           done, kept, error, escapes_after - escapes, sets_after - sets, recovered);
    fflush(stdout);
    require(done < NOGROW_LIMIT && error == ENOMEM && kept == done,
            "malloc returns NULL with ENOMEM once the table is full, every object intact");
    require(escapes_after - escapes <= NOGROW_ESCAPES && sets_after > sets,
            "NOGROW is set, without a loop of escapes");
    pass("nogrow_malloc_null");
    require(recovered, "malloc succeeds again after frees");
    pass("nogrow_malloc_recovers");
    return 0;
}

#define REALLOC_FROM (256UL * 1024)
/* Larger than REALLOC_FROM within the same representability granule (512 bytes for 256 KiB to 512 KiB),
 * so that a libc that resizes a large group's mapping in place or moves it (F4 part 1: __heap_remapping)
 * takes that path: at 1 MiB the granule is 2 KiB and the object's offset in its mapping does not meet
 * it, so every libc copies through malloc. */
#define REALLOC_TO (384UL * 1024)

static int pattern_kept(const unsigned char *p, unsigned long bytes)
{
    for (unsigned long i = 0; i < bytes; i += 509)
        if (p[i] != (unsigned char)(i * 7 + 3)) return 0;
    return 1;
}

/* Objects freed between two tries of realloc-exhausted. One object frees one slot, and a try whose new
 * mapping lands at a new address keeps the index pages it mapped (malloc-mapping.c never unmaps them),
 * so frees of one object at a time may never catch up with a move. */
#define REALLOC_STEP 64UL

/* realloc-exhausted (node_max_pages=16): realloc of a large object while the table is full, then
 * after each step of frees until it succeeds. */
static int mode_realloc_exhausted(void)
{
    (void)grant();
    if (!sublet_format()) {
        skip("realloc_exhausted_keeps_object", "the older formats end the context at exhaustion");
        return 0;
    }
    void **objects = pointer_array(NOGROW_LIMIT);
    unsigned char *big = malloc(REALLOC_FROM);
    require(big != NULL, "malloc of the large object");
    for (unsigned long i = 0; i < REALLOC_FROM; i += 509)
        big[i] = (unsigned char)(i * 7 + 3);
    unsigned long done = 0;
    while (done < NOGROW_LIMIT && (objects[done] = malloc(48)) != NULL)
        ++done;
    require(done < NOGROW_LIMIT, "the table fills");

    unsigned long filled = done, freed = 0;
    long moved_at = -1;
    int nulls = 0;
    for (;;) {
        errno = 0;
        unsigned char *q = try_realloc(big, REALLOC_TO);
        if (q) {
            require(pattern_kept(q, REALLOC_FROM), "a realloc that succeeds keeps the contents");
            big = q;
            moved_at = (long)freed;
            break;
        }
        int error = errno;
        printf("ROOT_REALLOC_TRY freed=%lu null errno=%d\n", freed, error);
        /* A NULL once frees made room: a libc that ended the object before the move's last step lost
         * it here, and the read below faults 25. */
        if (freed)
            printf("ROOT_REALLOC_NULL_AFTER_FREES freed=%lu\n", freed);
        fflush(stdout);
        require(error == ENOMEM, "a failed realloc answers ENOMEM");
        require(pattern_kept(big, REALLOC_FROM), "a failed realloc keeps the original intact");
        ++nulls;
        if (!done)
            break;
        for (unsigned long i = 0; i < REALLOC_STEP && done; ++i, ++freed)
            free(objects[--done]);
    }

    for (unsigned long i = 0; i < done; ++i)
        free(objects[i]);
    require(pattern_kept(big, REALLOC_FROM), "the object after the frees");
    free(big);
    printf("ROOT_REALLOC_EXHAUSTED filled=%lu nulls=%d moved_at=%ld\n", filled, nulls, moved_at);
    fflush(stdout);
    require(nulls >= 1, "realloc at a full table answers NULL at least once");
    require(moved_at > 0, "realloc succeeds once frees made room, and keeps the contents");
    pass("realloc_exhausted_keeps_object");
    return 0;
}

/* ---- The stage that dropped the reservation record M: blocks of depth 1 ---- */

/* partial-renew (ends with a fault 25; root grants): a reference narrowed to part of a block and renewed
 * is a block's reference with the narrower bounds (CRENEW keeps the record's place and depth), and the
 * reference from before is stale. It removes the pages it covers and ends the block only when they are
 * the last mapped ones; mapped pages outside its bounds it cannot remove, and the block then keeps its
 * address range. */
static int mode_partial_renew(void)
{
    (void)grant();
    if (!root_grants) {
        skip("renewed_narrow_keeps_block", BY_ADDRESS);
        skip("renewed_narrow_ends_block", BY_ADDRESS);
        return 0;
    }

    char *three = map(3 * PAGE);
    unsigned long base = cursor(three);
    memset(three, 0x31, 3 * PAGE);
    char *renewed = capstone_cap_renew(__builtin_capstone_cap_shrink(three, base + PAGE, base + 2 * PAGE));
    printf("ROOT_PARTIAL_RENEW middle type=%lu base=0x%lx end=0x%lx\n", type_of(renewed),
           base_of(renewed), end_of(renewed));
    fflush(stdout);
    require(type_of(renewed) == 1 && base_of(renewed) == base + PAGE && end_of(renewed) == base + 2 * PAGE &&
            renewed[0] == 0x31, "the renewed reference keeps the narrowed bounds and the page's data");
    require(refused(munmap(renewed, 2 * PAGE)),
            "munmap of the middle and the last page through the reference renewed over the middle one");
    require(refused(munmap(three, 3 * PAGE)), "munmap of the whole block through the reference from before");
    require(!munmap(renewed, PAGE), "munmap of the middle page through the renewed reference");
    require(registered(base) && !registered(base + PAGE) && registered(base + 2 * PAGE),
            "the block lives on with the outer pages mapped");
    require(!munmap(renewed, PAGE), "the renewed reference is still live: munmap of its hole again");
    for (unsigned i = 0; i < 16; ++i) {
        char *q = map(PAGE);
        require(cursor(q) + PAGE <= base || cursor(q) >= base + 3 * PAGE,
                "a new mapping lies outside the block");
    }
    pass("renewed_narrow_keeps_block");

    char *two = map(2 * PAGE);
    unsigned long base2 = cursor(two);
    memset(two, 0x41, 2 * PAGE);
    require(!munmap(two, PAGE), "munmap of the first page through the block's reference");
    char *last = capstone_cap_renew(__builtin_capstone_cap_shrink(two, base2 + PAGE, base2 + 2 * PAGE));
    char *object = capstone_cap_derive(last, 64, 64);
    object[0] = 0x42;
    require(last[64] == 0x42, "the object lies in the renewed block");
    require(refused(munmap(two + PAGE, PAGE)), "munmap of the last page through the reference from before");
    require(!munmap(last, PAGE), "munmap of the last mapped page through the renewed reference");
    require(!registered(base2), "the block has ended");
    pass("renewed_narrow_ends_block");
    stale = object;
    stale_read_bytes(base2, 2 * PAGE);
    return 0;
}

/* mremap-move (ends with a fault 25): a move maps a new block, copies, and unmaps the old range through
 * the reference the program passed. A move of part of a block removes only those pages, and the block
 * lives on. A move of a whole block ends the old block with CREVOKE on that reference, so no reference
 * into it, a copy of the block's reference or an object derived in it, reaches a later mapping at the
 * old address; the last line reads through the object after the program mapped the old address again,
 * and must fault with 25. */
static char *volatile moved_copy;

static int mode_mremap_move(void)
{
    (void)grant();
    char *three = map(3 * PAGE);
    unsigned long base3 = cursor(three);
    memset(three, 0x51, 3 * PAGE);
    char *part = mremap(three + PAGE, PAGE, 2 * PAGE, MREMAP_MAYMOVE);
    require(part != MAP_FAILED && (cursor(part) + 2 * PAGE <= base3 || cursor(part) >= base3 + 3 * PAGE),
            "mremap moves the middle page out of the block");
    require(part[0] == 0x51 && part[PAGE] == 0 && three[0] == 0x51 && three[2 * PAGE] == 0x51,
            "the moved page keeps its data, the outer pages keep theirs");
    require(registered(base3) && !registered(base3 + PAGE) && registered(base3 + 2 * PAGE),
            "the block lives on without the moved page");
    pass("mremap_move_part_removes_pages");

    char *two = map(2 * PAGE);
    unsigned long base2 = cursor(two);
    memset(two, 0x61, 2 * PAGE);
    char *object = capstone_cap_derive(two, 64, 64);
    moved_copy = two;
    char *moved = mremap(two, 2 * PAGE, 3 * PAGE, MREMAP_MAYMOVE);
    require(moved != MAP_FAILED && cursor(moved) != base2 && moved[0] == 0x61 &&
            moved[2 * PAGE - 1] == 0x61 && moved[2 * PAGE] == 0, "mremap moves the whole block");
    require(!registered(base2), "the old block has ended");
    require(refused(munmap(moved_copy, 2 * PAGE)), "munmap through a copy of the old block's reference");
    pass("mremap_move_whole_ends_block");
    stale = object;
    stale_read_bytes(base2, 2 * PAGE);
    return 0;
}

/* kernel-owned (exit 0): the kernel's own blocks refuse munmap, mremap and mprotect, of the whole block
 * or of a page, through the references the program holds to them: the image (gp) and the main stack
 * (sp). The launcher refuses each request itself (its check of the program's mappings). With root grants
 * and the hooks launcher's CAPSTONE_VM_TEST_OWNED_BYPASS, which skips that check, each request reaches
 * the module instead, and the module's kernel-owned flag must refuse it: owned_refused rises at every
 * one (owned_flag_refuses_each). Without the bypass that check skips. */
static int owned_counting;
static long owned_count;

/* A request of the kernel-owned mode: it failed with EINVAL (refused), and where the run counts, the
 * module's kernel-owned flag refused it, so owned_refused rose since the last request. */
static int owned_refusal(int refused_einval, const char *request)
{
    long now;

    if (!refused_einval)
        return 0;
    if (!owned_counting)
        return 1;
    now = module_number("owned_refused");
    printf("ROOT_KERNEL_OWNED %s owned_refused=%ld\n", request, now);
    fflush(stdout);
    if (now <= owned_count)
        return 0;
    owned_count = now;
    return 1;
}

static int mode_kernel_owned(void)
{
    (void)grant();
    void *image, *sp = stack_pointer();
    __asm__ volatile("movc %0, gp" : "=r"(image));
    unsigned long image_base = base_of(image), image_bytes = end_of(image) - image_base;
    unsigned long stack_base = base_of(sp), stack_bytes = end_of(sp) - stack_base;
    char *text = at_address(image, image_base), *data = at_address(image, image_base + image_bytes - PAGE);
    char *low = at_address(sp, stack_base);
    char data_rights[4], low_rights[4];
    const char *bypass = getenv("CAPSTONE_VM_TEST_OWNED_BYPASS");
    protection(image_base + image_bytes - PAGE, data_rights);
    protection(stack_base, low_rights);
    owned_counting = root_grants && bypass && atoi(bypass) > 0;
    owned_count = owned_counting ? module_number("owned_refused") : 0;
    printf("ROOT_KERNEL_OWNED image=0x%lx+0x%lx stack=0x%lx+0x%lx counting=%d owned_refused=%ld\n",
           image_base, image_bytes, stack_base, stack_bytes, owned_counting, owned_count);
    fflush(stdout);
    require(type_of(text) == 1 && type_of(data) == 1 && type_of(low) == 1 &&
            !(image_bytes & (PAGE - 1)) && !(stack_bytes & (PAGE - 1)) && image_bytes > PAGE &&
            stack_bytes > PAGE && (!owned_counting || owned_count >= 0),
            "the references to the image and the main stack");

    printf("ROOT_KERNEL_OWNED try=image-unmap\n");
    fflush(stdout);
    require(owned_refusal(refused(munmap(text, image_bytes)), "image-unmap") &&
            owned_refusal(refused(munmap(data, PAGE)), "image-unmap-page"),
            "munmap of the whole image and of its last page");
    pass("owned_image_unmap_refused");
    errno = 0;
    require(owned_refusal(mremap(text, image_bytes, image_bytes - PAGE, 0) == MAP_FAILED && errno == EINVAL,
                          "image-shrink"), "mremap shrinking the image");
    errno = 0;
    require(owned_refusal(mremap(text, image_bytes, image_bytes + PAGE, MREMAP_MAYMOVE) == MAP_FAILED &&
                          errno == EINVAL, "image-grow"), "mremap growing or moving the image");
    errno = 0;
    require(owned_refusal(mremap(text, image_bytes, image_bytes, MREMAP_MAYMOVE) == MAP_FAILED &&
                          errno == EINVAL, "image-same"), "mremap of the image to its own size");
    pass("owned_image_mremap_refused");
    errno = 0;
    require(owned_refusal(mprotect(data, PAGE, PROT_READ) == -1 && errno == EINVAL, "image-protect") &&
            protected_as(image_base + image_bytes - PAGE, data_rights), "mprotect of the image's last page");
    pass("owned_image_mprotect_refused");

    require(owned_refusal(refused(munmap(low, stack_bytes)), "stack-unmap") &&
            owned_refusal(refused(munmap(low, PAGE)), "stack-unmap-page"),
            "munmap of the whole main stack and of its lowest page");
    pass("owned_stack_unmap_refused");
    errno = 0;
    require(owned_refusal(mremap(low, stack_bytes, stack_bytes - PAGE, 0) == MAP_FAILED && errno == EINVAL,
                          "stack-shrink"), "mremap shrinking the main stack");
    errno = 0;
    require(owned_refusal(mremap(low, stack_bytes, stack_bytes + PAGE, MREMAP_MAYMOVE) == MAP_FAILED &&
                          errno == EINVAL, "stack-grow"), "mremap growing or moving the main stack");
    errno = 0;
    require(owned_refusal(mremap(low, stack_bytes, stack_bytes, MREMAP_MAYMOVE) == MAP_FAILED &&
                          errno == EINVAL, "stack-same"), "mremap of the main stack to its own size");
    pass("owned_stack_mremap_refused");
    errno = 0;
    require(owned_refusal(mprotect(low, PAGE, PROT_READ) == -1 && errno == EINVAL, "stack-protect") &&
            protected_as(stack_base, low_rights), "mprotect of the main stack's lowest page");
    pass("owned_stack_mprotect_refused");

    if (owned_counting)
        pass("owned_flag_refuses_each");
    else
        skip("owned_flag_refuses_each", root_grants ? "the launcher refuses these blocks first "
             "(no CAPSTONE_VM_TEST_OWNED_BYPASS)" : "root_grants=0 has no kernel-owned flag");
    return 0;
}

int main(int argc, char **argv)
{
    const char *mode = argc > 1 ? argv[1] : "unmap";

    if (!strcmp(mode, "unmap")) return mode_unmap();
    if (!strcmp(mode, "refuse")) return mode_refuse();
    if (!strcmp(mode, "narrow")) return mode_narrow();
    if (!strcmp(mode, "renew")) return mode_renew(1);
    if (!strcmp(mode, "renew-ref")) return mode_renew(0);
    if (!strcmp(mode, "mremap")) return mode_mremap();
    if (!strcmp(mode, "exit-victim")) return mode_exit_victim();
    if (!strcmp(mode, "exit-renew")) return mode_exit_renew();
    if (!strcmp(mode, "create")) return mode_create();
    if (!strcmp(mode, "create-fail")) return mode_create_fail();
    if (!strcmp(mode, "detached")) return mode_detached();
    if (!strcmp(mode, "exit-length")) return mode_exit_length(STACK - PAGE, "wrong");
    if (!strcmp(mode, "exit-huge")) return mode_exit_length(~0UL, "huge");
    if (!strcmp(mode, "exit-frame")) return mode_exit_frame();
    if (!strcmp(mode, "protect-held")) return mode_protect_held();
    if (!strcmp(mode, "protect")) return mode_protect();
    if (!strcmp(mode, "create-gate")) return mode_create_gate();
    if (!strcmp(mode, "join-two")) return mode_join_two();
    if (!strcmp(mode, "join-after")) return mode_join_after();
    if (!strcmp(mode, "exit-many")) return mode_exit_many();
    if (!strcmp(mode, "exit-word")) return mode_exit_word();
    if (!strcmp(mode, "attach-readonly")) return mode_attach_readonly();
    if (!strcmp(mode, "exit-word-refs")) return mode_exit_word_refs();
    if (!strcmp(mode, "futex")) return mode_futex();
    if (!strcmp(mode, "advise")) return mode_advise();
    if (!strcmp(mode, "join-burned")) return mode_join_burned();
    if (!strcmp(mode, "attach-exchange")) return mode_attach_exchange();
    if (!strcmp(mode, "attach-address")) return mode_attach_address();
    if (!strcmp(mode, "futex-race")) return mode_futex_race();
    if (!strcmp(mode, "futex-remap")) return mode_futex_remap();
    if (!strcmp(mode, "exit-race")) return mode_exit_race();
    if (!strcmp(mode, "join-withdrawn")) return mode_join_withdrawn();
    if (!strcmp(mode, "rollback-fail")) return mode_rollback_fail();
    if (!strcmp(mode, "die-blocked")) return mode_die_blocked();
    if (!strcmp(mode, "word-metadata")) return mode_word_metadata();
    if (!strcmp(mode, "futex-readonly")) return mode_futex_readonly();
    if (!strcmp(mode, "futex-park")) return mode_futex_park();
    if (!strcmp(mode, "futex-end")) return mode_futex_end();
    if (!strcmp(mode, "futex-flags")) return mode_futex_flags();
    if (!strcmp(mode, "eintr-word")) return mode_eintr_word();
    if (!strcmp(mode, "eintr-exit")) return mode_eintr_exit();
    if (!strcmp(mode, "advise-refuse")) return mode_advise_refuse();
    if (!strcmp(mode, "die-stderr")) return mode_die_stderr();
    if (!strcmp(mode, "protect-exec")) return mode_protect_exec();
    if (!strcmp(mode, "partial-hole")) return mode_partial_hole();
    if (!strcmp(mode, "partial-all")) return mode_partial_all();
    if (!strcmp(mode, "partial-narrow")) return mode_partial_narrow();
    if (!strcmp(mode, "partial-fixed")) return mode_partial_fixed();
    if (!strcmp(mode, "partial-mremap")) return mode_partial_mremap();
    if (!strcmp(mode, "mremap-validate")) return mode_mremap_validate();
    if (!strcmp(mode, "pages-enomem")) return mode_pages_enomem();
    if (!strcmp(mode, "pages-tagged")) return mode_pages_tagged();
    if (!strcmp(mode, "futex-stale")) return mode_futex_stale();
    if (!strcmp(mode, "pages-derive")) return mode_pages_derive();
    if (!strcmp(mode, "pages-fault")) return mode_pages_fault();
    if (!strcmp(mode, "pages-admit")) return mode_pages_admit();
    if (!strcmp(mode, "release-fail")) return mode_release_fail();
    if (!strcmp(mode, "pages-closed")) return mode_pages_closed();
    if (!strcmp(mode, "pages-discard")) return mode_pages_discard();
    if (!strcmp(mode, "pages-unclear")) return mode_pages_unclear();
    if (!strcmp(mode, "top-page")) return mode_top_page();
    if (!strcmp(mode, "grow-maps")) return mode_grow_maps();
    if (!strcmp(mode, "grow-objects")) return mode_grow_objects();
    if (!strcmp(mode, "nogrow-map")) return mode_nogrow_map();
    if (!strcmp(mode, "nogrow-malloc")) return mode_nogrow_malloc();
    if (!strcmp(mode, "realloc-exhausted")) return mode_realloc_exhausted();
    if (!strcmp(mode, "partial-renew")) return mode_partial_renew();
    if (!strcmp(mode, "mremap-move")) return mode_mremap_move();
    if (!strcmp(mode, "kernel-owned")) return mode_kernel_owned();
    fail("unknown mode");
    return 1;
}
