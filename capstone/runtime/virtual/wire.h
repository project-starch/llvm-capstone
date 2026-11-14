#ifndef CAPSTONE_VIRTUAL_WIRE_H
#define CAPSTONE_VIRTUAL_WIRE_H
#include "vm-abi.h"
#include <linux/types.h>
#include <linux/ioctl.h>

/* The end of the user address range under Sv39 (2^38) and the highest end of a mapping block, a page
 * below it. No reference's cursor can be 2^38, so an object ending there could not hold a tagged pointer
 * one past its end: no block holds the topmost page (the module's add() refuses it, capstone_vm.c
 * CV_BLOCK_END; the launcher holds the page itself, exec.c top_guard). Only R ends at 2^38. */
#define CV_USER_END (1UL << 38)
#define CV_BLOCK_END_MAX (CV_USER_END - 4096)

/* Experimental one-hart adapter; all thread and mapping addresses belong to
 * the owning Linux mm and share one lifetime table. */
/* A grant. reserved is 0: a grant carries no capability rights of its own (block A, 2026-11-13; it was
 * the rights, 0 to 7): Sublet checks bounds and lifetime, and the page table decides read, write and
 * execute. The field keeps the layout, so CV_ADD and CV_REVIVE keep their numbers. */
struct cv_map {
    __u64 thread, address, bytes, reserved, reg, cursor, linear;
    __u64 id;
    /* The image grant only (reg 3): its first rx_bytes are the read+execute
     * segment, mapped from the image file without write permission. They are
     * neither pinned nor swept; the rest is admitted page by page at first
     * touch, as any grant. */
    __u64 rx_bytes;
    /* The length the program's mmap exposes, from the base: bytes minus the
     * PROT_NONE padding behind it. 0 means bytes. A whole number of pages, at
     * most bytes. With root_grants a program's request to end the mapping must
     * name exactly this length (capstone_vm.c, request_operand). CV_REVIVE
     * ignores it: the parked arena keeps the length of its CV_ADD. */
    __u64 visible;
};
struct cv_step {
    /* Zero is the original process thread. Other IDs are allocated by
     * CV_THREAD_CREATE and name a paused CSRUNV context. */
    __u64 thread;
    __u64 reply, result;
    __u64 kind, cause, pc, address, args[8];
    /* On a TERMINAL step (a fault), the cursors of x1 (ra) and x8 (fp) as the
     * escape left them in the frame's register slots -- the faulting
     * instruction's return address and frame pointer, so a fault inside a
     * leaf routine (musl's strlen) can be tied to its caller without a stack
     * walk. Zero when the platform did not write the slot (a QEMU before the
     * escape stored them; a paused step). Diagnostic only: nothing decides on
     * them. These two words also pad the record to 136 bytes. Linux's RISC-V
     * user copy (both pointers 8-byte aligned, as here) moves 64-byte chunks
     * by word while more than one chunk remains and the rest byte by byte: 120
     * bytes cost a 56-byte tail and 128 a 64-byte one; 136 is two chunks and 8
     * bytes. The size is part of the CV_STEP ioctl number: module and launcher
     * change together. */
    __u64 ra, fp;
};
/* A start frame is a page-aligned, writable user page containing the virtual
 * CSRUNV frame format. The trusted adapter fills satp and srevroot, then
 * consumes the tagged register slots when the thread first runs.
 *
 * creator: the thread paused at CV_SERVICE_THREAD_CREATE. With root_grants
 * the hand-over of the frame is the program's request to end the frame's
 * mapping: the creator passes the frame's reference in a0, and the module
 * checks it like a munmap operand and ends the mapping only if the thread is
 * created. A refused reference refuses the create (EINVAL) and changes
 * nothing. Without root_grants the module ignores creator. */
struct cv_thread_create {
    __u64 frame;
    __u64 thread;
    __u64 creator;
};
struct cv_thread_control {
    __u64 thread;
    __u64 frame;
};
/* CV_THREAD_EXIT: stop thread and release the arenas the launcher names with
 * it, each by its arena id or 0. frame: the thread's frame. stack: the stack
 * the thread asks to unmap at its exit (CV_SERVICE_PTHREAD_EXIT, a1 the
 * stack's reference, a2 its length). The launcher names only a registered heap
 * mapping, not the allocator's metadata, held by no worker, of the length the
 * call names; since stage 4e also one that holds the exit word (a0), whose
 * store the module then drops (CV_WORD_RELEASED), as Linux drops a
 * clear_child_tid store into a stack that is gone. The
 * module refuses a stack that is the thread's own frame, and releases that
 * arena once, as the frame. With root_grants the stack is a request of
 * the program: the module checks the reference in a1 and ends the stack's
 * block through it (CREVOKE on a1, while the paused thread's frame still holds
 * it) before the outcome is written, and after the thread has stopped it
 * releases the arena. A kernel-owned block (root_grants) is never a stack the
 * module releases. released (out): CV_EXIT_FRAME and
 * CV_EXIT_STACK for the arenas the module released; the launcher then unmaps
 * those ranges. The module writes it before the thread stops, and with
 * root_grants nothing after that can fail; the stack's CREVOKE comes before it,
 * so if that write fails, the ioctl answers EFAULT, the ended stack stays
 * registered and the launcher ends the program. A refused or impossible
 * release does not stop the exit: the thread exits and that arena stays
 * registered (a leak, counted in exit_kept and reported).
 *
 * word (in, stage 4d): 1 asks the module to store 0 into the thread's exit word, the int at a0's
 * cursor (CV_SERVICE_PTHREAD_EXIT), at the end of this same ioctl, after the thread has stopped
 * and the frame and stack releases are committed; 0 asks nothing. The launcher asks only after its
 * own conditions on the address (one registered mapping, not the allocator's metadata, pages that
 * Linux maps writable, outside every transport it writes itself). The module checks the word as
 * CV_WORD does before it clears the frame (with root_grants a0 holds a live reference that covers
 * the word; in both modes the word is aligned, inside one arena past the image's read+execute
 * prefix, and in no thread's frame), and checks it again at the store, under vm_lock, which also
 * excludes the program's execution: the reference is still live, the same arena still holds the
 * word (no release of this ioctl took it) and no thread's frame does. Then it admits the word's page
 * (CV_TOUCH's rule) and stores through the program's mapping, so the page table decides the write
 * right. Check and store are one step: no thread can end the word's lifetime, unmap or remap its
 * page in between. word_reason (out): CV_WORD_ACCEPTED if the module stored the word, else the
 * reason of the refusal; written at the end, and 0 when word is 0. A refused word does not stop the
 * exit. */
struct cv_thread_exit {
    __u64 thread, frame, stack, released, word, word_reason;
};
#define CV_EXIT_FRAME 1
#define CV_EXIT_STACK 2
struct cv_nodes {
    __u64 thread, available;
};
struct cv_stats {
    __u64 arenas, pinned_pages, peak_pages, nodes, steps, faults;
    __u64 collections, reclaimed, nodes_high_water, nodes_live, nodes_retired;
    __u64 node_capacity, node_bytes, node_growths;
};
struct cv_global {
    __u64 contexts, arenas, pinned_pages, nodes_allocated, nodes_high_water;
    __u64 nodes_live, nodes_retired, collections, reclaimed;
    __u64 node_capacity, node_bytes;
};
#define CV_ADD _IOWR('V', 16, struct cv_map)
#define CV_STEP _IOWR('V', 17, struct cv_step)
#define CV_RETIRE _IOW('V', 18, __u64)
#define CV_STATS _IOR('V', 19, struct cv_stats)
#define CV_RESOLVE _IOW('V', 20, struct cv_thread_control)
#define CV_GLOBAL _IOR('V', 21, struct cv_global)
#define CV_THREAD_CREATE _IOWR('V', 22, struct cv_thread_create)
#define CV_THREAD_EXIT _IOWR('V', 23, struct cv_thread_exit)
#define CV_NODES _IOW('V', 24, struct cv_nodes)
/* 25 and 26 were CV_REMAP_BEGIN and CV_REMAP_END, the remap transaction of the first design. */
#define CV_PROFILE _IOR('V', 27, __u64)
/* A span the launcher is about to WRITE on the domain's behalf through Linux: a worker's meta block
 * and exchange region, the process exchange (since stage 4d the module admits an exit word or a
 * futex word itself, with the same rule, inside the ioctl that accesses it). Every page of it inside a lazily admitted
 * arena is admitted first, pinned and checked exactly as the domain's own fault would admit it,
 * so the launcher's write never creates backing the domain can store a capability into without
 * the module's pin and sweep (a frame freed at exit_mm with its tags). Pages already held and
 * arenas pinned whole need nothing; a span outside every arena is refused (EINVAL), one inside
 * the image's read+execute prefix too (EACCES). Launcher and module change together. */
struct cv_touch {
    __u64 address, bytes;
};
#define CV_TOUCH _IOW('V', 29, struct cv_touch)
/* Mapping cache (exec.c, CAPSTONE_VM_MAPCACHE; EXPERIMENTAL, default off). The launcher keeps the
 * Linux mapping of a heap arena the domain unmapped and hands the same backing to a later grant:
 *   CV_PARK    the arena id. Ends the domain's access now (the arena's ancestor lifetime is retired,
 *              so every capability to it is dead) but keeps the registration and the pages it
 *              holds pinned. Pinned mode only (EOPNOTSUPP otherwise); the launcher then unmaps as
 *              before. A parked arena is retired like any other by CV_RETIRE.
 *   CV_REVIVE  a struct cv_map shaped as the reply grant CV_ADD would mint, with .id the parked
 *              arena's id and address/bytes its range. Overwrites every pinned page with zeros (data
 *              and tags), mints a new ancestor and gives the arena a new id, returned in .id. */
#define CV_PARK _IOW('V', 30, __u64)
#define CV_REVIVE _IOWR('V', 31, struct cv_map)
/* With root_grants, CV_RETIRE releases only an arena whose block has already ended: through a
 * program's request (CV_UNMAP, the frame's hand-over at CV_THREAD_CREATE) or a park. While the
 * block lives it is refused with EPERM and changes nothing. The launcher's one CV_RETIRE is the
 * eviction of a parked mapping from its pool. CV_PARK is refused with EPERM: a park is a
 * program's request and goes through CV_UNMAP. */
/* A program's own unmap (CV_SERVICE_UNMAP): thread is the caller, paused at the service, id the
 * arena the launcher resolved from the cursor of the reference in a0. park (in): 1 asks to pool
 * the mapping (CV_PARK's effect) where the module can, 0 to retire it. park (out): what the module
 * did, one of the CV_UNMAP_* values; the launcher unmaps the range unless it reads
 * CV_UNMAP_PARKED. One call decides, so a refused reference is refused and counted once. Without
 * root_grants the module ignores thread. With root_grants the reference is the authority: the
 * module reads it with its tag from the caller's saved a0 and ends the block only if it is a live
 * reference whose record lies one level below R (a block's, never an object inside it) and whose
 * bounds cover the length the program's mmap exposed, and the call names exactly that length
 * (capstone_vm.c, request_operand). The module writes park (out) and then ends the block with
 * CREVOKE on that reference; a refused CREVOKE answers EINVAL, the launcher then discards park (out),
 * and nothing has changed. Every other refusal is EINVAL and changes nothing too; so is a kernel-owned
 * block with root_grants, and the image without (as before the stage "drop M"). After the CREVOKE
 * nothing can fail. */
struct cv_unmap {
    __u64 thread, id, park;
};
#define CV_UNMAP_RETIRED 0
#define CV_UNMAP_PARKED 1
#define CV_UNMAP_NO_POOL 2   /* retired: this module does not pool (unpinned) */
#define CV_UNMAP _IOWR('V', 32, struct cv_unmap)
/* A program's mprotect (CV_SERVICE_PROTECT), asked before the launcher changes the protection,
 * or a program's question before mremap moves a page range (CV_SERVICE_MAPPING_CHECK): thread is
 * the caller, paused at the service (the module reads which one from its a7), id the arena the
 * launcher resolved from the cursor of the reference in a0. In both modes an mprotect that asks for
 * PROT_EXEC (a2) is refused with EACCES (only the image's text is executable), and a range that
 * holds a page a partial munmap removed with ENOMEM (mprotect) or EFAULT (the mapping check).
 * Without root_grants the module accepts the rest. With root_grants the reference is the authority,
 * as for munmap: the module accepts only a live reference whose record lies one level below R and
 * whose bounds cover [a0's cursor, + a1), any page range inside the visible length, whatever its
 * cursor. A kernel-owned block is refused with root_grants, the image without. It ends nothing. A
 * refusal is EINVAL, and the launcher then changes nothing. */
struct cv_protect {
    __u64 thread, id;
};
#define CV_PROTECT _IOW('V', 33, struct cv_protect)
/* The compare of a futex wait (CV_SERVICE_FUTEX, stage 4d), one step that checks and reads: thread
 * is the caller, paused at the service; the word is the int at a0's cursor and the value is a2. The
 * module checks the word as at an exit (with root_grants a0 holds a live non-linear reference whose
 * bounds cover the word; in both modes the word is aligned, lies inside the visible length of one
 * registered arena past the image's read+execute prefix, and in no page that holds the frame of a
 * thread), admits its page (CV_TOUCH's rule) and reads it through the program's mapping, so the
 * page table decides the read right, all under vm_lock, which also excludes the program's
 * execution: no thread can end the word's lifetime, unmap or remap its page between the check and
 * the read. Since stage 4e the read admits no page: a page never touched reads Linux's zero page, and
 * a read-only page compares as any other. reason (out): CV_WORD_ACCEPTED, or the reason of the
 * refusal. equal (out): 1 if the word read holds the value. arena and offset (out, when accepted):
 * the arena id of the mapping that holds the word and the word's offset in it (stage 4d's key; since
 * stage 4e the launcher keys its waiter by the checked address, as Linux does, and wakes the waiters
 * of every range that stops being mapped). lifetime (out, when accepted, stage 4f): the lifetime id
 * of the reference in a0 with root_grants, else 0; a wake counts the waiter only while it is live
 * (CV_LIVE). Linux never sees the program's address. The ioctl itself fails only for a caller that
 * is not paused at the service (EINVAL). Changes no program memory. */
struct cv_word {
    __u64 thread, reason, equal, arena, offset, lifetime;
};
#define CV_WORD_ACCEPTED 0
#define CV_WORD_NOT_REFERENCE 1
#define CV_WORD_NOT_LIVE 2
#define CV_WORD_NOT_COVERING 3
#define CV_WORD_UNALIGNED 4
#define CV_WORD_NOT_MAPPED 5
#define CV_WORD_IN_FRAME 6
#define CV_WORD_NOT_ADMITTED 7      /* the module cannot admit the word's page (CV_TOUCH's rule) */
#define CV_WORD_NO_ACCESS 8         /* the page table refuses the read or the write */
#define CV_WORD_RELEASED 9          /* an exit word in the stack this exit released: dropped, no error */
#define CV_WORD_REASONS 10
#define CV_WORD _IOWR('V', 34, struct cv_word)
/* A worker's attachment of its own transport (CV_SERVICE_THREAD_DELEGATE, stage 4c): thread is the
 * caller, paused at the service; meta and exchange are the arenas the launcher resolved from the
 * cursors of a0 (the meta block's reference) and a1 (the exchange region's); meta_bytes is the
 * meta block's length, a2 the exchange region's. With root_grants each reference must pass the
 * whole-block check of munmap (a live reference whose record lies one level below R and whose bounds
 * cover the visible length, which the call names) and neither arena may be kernel-owned (EINVAL). In
 * both modes neither arena may hold the frame of a thread (EBUSY). Ends nothing. A refused reference is
 * EINVAL, and the launcher then refuses the attachment. */
struct cv_attach {
    __u64 thread, meta, exchange, meta_bytes;
};
#define CV_ATTACH _IOW('V', 35, struct cv_attach)
/* A program's madvise(MADV_DONTNEED or MADV_FREE) of [a0's cursor, + a1) in arena id
 * (CV_SERVICE_ADVISE, stage 4c): thread is the caller, paused at the service. With root_grants
 * the reference in a0 must be a live reference whose record lies one level below R and whose bounds
 * cover the named page range, which may be part of the block, as for mprotect, and the block may not
 * be kernel-owned (EINVAL). In both modes the arena may not hold the frame of a thread (EBUSY) or be the
 * image (EINVAL).
 * zeroed (out): 1 if the module, after pinning every resident page of the range, overwrote every
 * page of it that it holds with zeros (data and tags); the launcher then changes nothing. 0 in the
 * unpinned mode, where the module holds no pages: the launcher gives the range back to Linux with
 * madvise(MADV_DONTNEED). Either way the range reads zero afterwards. A refusal is EINVAL. */
struct cv_advise {
    __u64 thread, id, zeroed;
};
#define CV_ADVISE _IOWR('V', 36, struct cv_advise)
/* A page operation inside one mapping block, which ends no lifetime (stage 4e; stage 4f makes it one
 * step for the program): a partial munmap (CV_PAGES_REMOVE, the caller paused at CV_SERVICE_UNMAP,
 * the range [a0's cursor, + a1)), MAP_FIXED inside the block (CV_PAGES_FIXED, at CV_SERVICE_REFILL,
 * the range [a0's cursor, + a1), the protection a2) and mremap's shrink and
 * growth in place (CV_PAGES_SHRINK, CV_PAGES_GROW, at CV_SERVICE_RESIZE: the old range [a0's cursor,
 * + a1) and the new length a2; a shrink removes [+ a2, + a1), a growth refills [+ a1, + a2)). thread is
 * the caller, id the arena the launcher resolved; the registers are read from the caller's saved
 * frame. Three calls with service_lock held across them, and Linux's change of the range between
 * the first two (block A: the refill service no longer takes the operation in a3):
 *   commit CV_PAGES_CHECK: the range (for a resize the old range) is a nonempty page range inside the
 *     visible length, the arena holds no thread's frame (EBUSY), MAP_FIXED asks for no PROT_EXEC (a2,
 *     EACCES), no other page operation of the context is open (EBUSY), the block is not the image
 *     (EINVAL), and with root_grants it is not kernel-owned (EINVAL) and a0 holds a block's reference:
 *     live, its record one level below
 *     R, its bounds covering the range whatever its cursor (EINVAL otherwise). A resize first validates
 *     the whole old range: every page of it logically mapped (EFAULT). A growth needs every page of the
 *     new tail inside the visible length, removed and covered by the reference (ENOMEM). MAP_FIXED and
 *     a growth follow the refill rule: no object derived below the reference (EINVAL for MAP_FIXED,
 *     ENOMEM for a growth; without root_grants EOPNOTSUPP and ENOMEM).
 *     Then the module pins every resident page of the range it does not hold yet and opens the
 *     operation: until its commit or abort no thread of the program runs and no page of the range is
 *     admitted. A removal of every logically mapped page that is left opens nothing: it ends the
 *     block in this call: ended 1 is written, then the CREVOKE on the passed reference (a refused one
 *     answers EINVAL, the launcher discards ended, and nothing has changed), then the release of the
 *     arena, as CV_UNMAP's retire; the launcher then gives the whole extent back to Linux.
 *   commit CV_PAGES_COMMIT, after the launcher changed the protection of the range with mprotect
 *     (PROT_NONE for a removal) and, for a refill, discarded its contents with MADV_DONTNEED: a refill
 *     asks the refill rule again and fails closed (ends the block through the request's reference,
 *     ended 1) if an object below it exists now; the module clears the tags of every page of the range
 *     it holds and unpins it,
 *     rebuilds the arena's frame set, updates the removed pages and the count of logically mapped
 *     pages, and closes the operation.
 *   commit CV_PAGES_ABORT, after the launcher's protection change failed and the launcher restored the
 *     range's protection: the module closes the operation and changes nothing. Only a protection change
 *     can be undone this way.
 *   commit CV_PAGES_FAIL (block A), after a refill's discard (MADV_DONTNEED) failed, possibly part way:
 *     the module fails closed, ends the block (CREVOKE on the request's reference, the sweep and the
 *     release of the arena) and answers ended 1; the launcher gives the extent back and answers
 *     EINVAL.
 *   A commit that fails closed, at CV_PAGES_COMMIT or CV_PAGES_FAIL, clears the tags of every page of the
 *     arena before Linux reuses its frame: the pages it holds through the retire, and, if the retire's
 *     preparation cannot pin a resident page, every page it does not hold in place through the page
 *     table. If a page stays as it was there (a swapped, shared or huge page, outside the profile), the
 *     module answers ended CV_PAGES_ENDED_HELD instead of CV_PAGES_ENDED: the launcher keeps the whole
 *     extent mapped PROT_NONE and never gives it back to Linux. So it does when the CREVOKE on the
 *     request's reference is refused (the reference was renewed or ended in the operation's window,
 *     which only a hooks build allows), since the block may then live on in a renewal: the module then
 *     keeps the arena registered and quarantined instead of releasing it.
 * removed (out): the arena's removed pages after the call, which the launcher compares with its own
 * count. A removed page is never admitted until a refill maps it again; no other mapping receives an
 * address inside the block while it lives. */
struct cv_pages {
    __u64 thread, id, op, commit, ended, removed;
};
#define CV_PAGES_REMOVE 1
#define CV_PAGES_FIXED 2
#define CV_PAGES_GROW 3
#define CV_PAGES_SHRINK 4
#define CV_PAGES_CHECK 0
#define CV_PAGES_COMMIT 1
#define CV_PAGES_ABORT 2
#define CV_PAGES_FAIL 3
#define CV_PAGES_ENDED 1          /* ended: the block's lifetime ended, its extent goes back to Linux */
#define CV_PAGES_ENDED_HELD 2     /* ended, and a page kept its tags: the extent stays mapped PROT_NONE */
#define CV_PAGES _IOWR('V', 37, struct cv_pages)
/* Is the lifetime id (a futex waiter's, cv_word.lifetime) still live (stage 4f)? live (out): 1 if
 * it is, or if the module cannot tell (no root_grants, id 0). Reads the table only. */
struct cv_live {
    __u64 lifetime, live;
};
#define CV_LIVE _IOWR('V', 38, struct cv_live)
/* The image file, by descriptor, before the launcher validates it: the module
 * denies writers to it (ETXTBSY, as Linux does for a running executable) until
 * the context is released, so the bytes validated are the bytes loaded. */
#define CV_IMAGE _IOW('V', 28, __s32)
#endif
