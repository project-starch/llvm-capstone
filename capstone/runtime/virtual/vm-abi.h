#ifndef CAPSTONE_VIRTUAL_VM_ABI_H
#define CAPSTONE_VIRTUAL_VM_ABI_H
/* Version 7: the image is two load segments (virtual/link.ld), text mapped
 * from the image file without write permission and data after it, with
 * mallocng compiled inside Capstone. Like version 6 it runs on the DEFAULT
 * QEMU profile: bounds are representable and allocator-enforced, so no
 * exact-bounds option is needed (option iii). The launcher accepts this
 * version only; an image of any earlier version, whose layout or allocator
 * differs, must be rebuilt, not re-run. */
#define CV_IMAGE_MAGIC 0x374d56564e4f5043
#define CV_SERVICE_DELEGATE 0
/* a0 the visible length, a1 the alignment, a2 the protection (PROT_EXEC refused with EACCES: only
 * the image's text is executable), a3 the rights, a4 the kind. */
#define CV_SERVICE_MAP 1
/* a0 the mapping's reference, its cursor the first page; a1 the length. A call that names the whole
 * block, [base, + visible length), ends it (with root_grants through a live reference to the
 * mapping that covers it: CREVOKE of M). Since stage 4e a call that names any other page range
 * inside the block is a partial munmap, with root_grants only through the block's own reference,
 * whose bounds cover the whole block, whatever its cursor: the kernel removes those pages (they stay
 * part of the block's VMAs, PROT_NONE, their contents discarded); no lifetime ends, and no other
 * mapping receives addresses inside the block. Unmapping a removed page again changes nothing. The
 * removal of the last pages that are still mapped ends the block as a whole-block call does. A range
 * beyond the block, across two blocks or by a plain address (with root_grants) is refused (EINVAL).
 * A refusal by Linux (ENOMEM: a VMA split at the map count limit) changes nothing (stage 4f). With
 * root_grants=0 a partial munmap is authorised by the address alone: a legacy profile without the
 * reference guarantee. */
#define CV_SERVICE_UNMAP 2
/* a0: the start frame's reference. With root_grants the kernel checks it as a munmap operand and
 * ends the frame's mapping when it creates the thread. -EAGAIN: the kernel took the frame and the
 * launcher released it after a later failure, so the program must not unmap it; every other error
 * leaves the frame the program's. */
#define CV_SERVICE_THREAD_CREATE 3
#define CV_SERVICE_THREAD_EXIT 4
/* a0 the thread id. 0 once that thread has exited, also when it exited before the call (thread
 * ids are never reused). -EDEADLK for the caller's own (0 for the main thread), -ESRCH for an id
 * never issued. A create that fails issues no id: its id goes to the next create that succeeds, and
 * until then a join of it answers -ESRCH. Any number of threads may join one thread. */
#define CV_SERVICE_THREAD_JOIN 5
/* a0: the reference of the mapping, its cursor the first page; a1 the length; a2 the protection.
 * With root_grants the kernel checks the reference as for munmap, except that the range may be
 * any page range inside the mapping, which the reference must cover. It ends nothing. A mapping
 * that holds a worker's transport answers EBUSY, as for munmap. PROT_EXEC is refused (EACCES): only
 * the image's text is executable. A range with a page that a partial munmap removed answers ENOMEM,
 * as Linux answers for an unmapped page. The image and the launcher's other mappings are refused
 * (EINVAL): the image's data reference covers the whole image, so only the page's write bit
 * protects its text. */
#define CV_SERVICE_PROTECT 6
#define CV_SERVICE_WAIT 7
/* a0 the word's reference, a1 the operation (FUTEX_WAIT, FUTEX_WAKE and their BITSET forms), a2
 * the value, a3 and a4 the timeout (seconds -1: none; a relative timeout that overflows the clock
 * waits without one, as Linux clamps it), a5 the bitset. A wait reads the word, so the kernel checks
 * it as an exit word, without the conditions on writing: with root_grants a live reference whose
 * bounds cover the word, and in both modes an aligned int inside one mapping, in a page that is not
 * removed, that is not the allocator's metadata and in no thread's frame (EFAULT otherwise). The
 * kernel checks the reference and compares the word with the value in one step (EAGAIN if they
 * differ), and the waiter then sleeps on the address the kernel checked, as Linux keys a private
 * futex (stage 4e). When a mapping ends or a page range of it is removed (munmap, a park, a thread's
 * stack or frame at its exit, a partial munmap) every waiter in that range is woken and answers 0:
 * the libc reads the word again through its own reference, which faults if it is stale, so the
 * wake is a spurious one. A wake neither reads nor writes the word, so an integer address is
 * enough; it must be an aligned int inside one registered mapping, in a page that is not removed
 * (EFAULT otherwise). FUTEX_CLOCK_REALTIME with any operation but FUTEX_WAIT_BITSET answers ENOSYS,
 * as Linux does, and so does an unknown operation or flag. */
#define CV_SERVICE_FUTEX 8
/* a0 the exit word's reference (or 0), a1 the reference of the mapping that holds the thread's
 * stack (or 0), a2 its length, a3 the status. With root_grants the stack is unmapped only through
 * a1, as for munmap. A refused stack does not stop the exit: the mapping stays, and the program can
 * still unmap it through a reference it kept. A stack of the wrong length is refused this way. An
 * exit word inside the stack does not keep it (stage 4e): the stack is released and the word is not
 * written, without an error, as Linux drops a clear_child_tid store into an unmapped stack. A stack
 * that is the thread's own frame is refused as a stack too,
 * but the exit releases the frame, which ends that mapping. After the thread has stopped and its
 * frame and stack are released, the kernel stores 0 into the exit word, and the launcher wakes the
 * word's futex waiters, if the word is an aligned int inside one mapping that is not the
 * allocator's metadata, in pages that the program's grant may write and that Linux maps writable,
 * outside the launcher's own transport and every worker's, and if the kernel accepts it at the
 * store: with root_grants a0 must be a live reference whose bounds cover the word, and in both
 * modes the word may not lie in the frame of any thread. A word that fails is not written (the
 * launcher logs the refusal) and the exit still completes; futex waiters on a refused word stay
 * blocked, which is the program's bug. The kernel's wait for its own lock is retried after a signal
 * (EINTR comes before any effect). */
#define CV_SERVICE_PTHREAD_EXIT 9
/* a0 the meta block's reference, a1 the exchange region's, a2 its length. The process transport
 * (the launcher's own meta block and exchange region, named by their addresses) is served as is.
 * A worker's first call attaches its own transport: each block must be a whole mapping of the
 * program held by no worker, every page of it writable, and with root_grants each reference must
 * pass the whole-mapping check of munmap. A refused attachment ends the program. */
#define CV_SERVICE_THREAD_DELEGATE 10
/* The calling context's launcher id + 1. Since the transport lock reads its identity from gp's
 * cursor (delegate.c dl_wire_identity), only a context created with no identity slot free asks. */
#define CV_SERVICE_THREAD_SELF 11
#define CV_SERVICE_NODES 12
/* 13 was the remap transaction of the first design (no caller since the libc's mremap became
 * map, copy and unmap); it answers ENOSYS. */
/* A worker gives its transport (meta block and exchange region) back: the launcher stops
 * writing into them and releases its hold on their mappings, which until then refuses their
 * unmap to everyone, the worker included. The worker's next delegate service attaches again. */
#define CV_SERVICE_TRANSPORT_DETACH 14
/* a0 the reference of a mapping, its cursor the first page of a range; a1 the range's length; a2
 * flags (stage 4f). mremap's validator of its old range: answers 0 if the range is a page range inside
 * one block (EINVAL otherwise, EBUSY while a worker's transport holds the block), every page of it is
 * logically mapped (EFAULT otherwise, as Linux answers for an unmapped part of mremap's old range),
 * and with root_grants a0 is the block's own reference, whose bounds cover the whole block (EINVAL
 * otherwise); with CV_CHECK_MOVE in a2 also every page of it read-write (ENOTSUP otherwise), the
 * condition of a move. Changes nothing. The libc's mremap asks it before it returns the reference of a
 * resize to the same page count (flags 0) and before it copies anything (CV_CHECK_MOVE). */
#define CV_SERVICE_MAPPING_CHECK 15
#define CV_CHECK_MOVE 1
/* a0 the reference of a mapping, its cursor the first page; a1 the length; a2 MADV_DONTNEED or
 * MADV_FREE (EINVAL otherwise). The range reads zero afterwards. With root_grants the kernel checks
 * the reference as for mprotect: a live reference to the mapping, not to an object inside it,
 * whose bounds cover the page range. It ends nothing. A mapping that holds a worker's transport
 * answers EBUSY, as for mprotect. */
#define CV_SERVICE_ADVISE 16
/* Stage 4e. a0 the block's reference, its cursor the first page of the range; a1 the length; a2 the
 * protection (PROT_EXEC refused with EACCES); a3 CV_PAGES_FIXED (wire.h; any other value EINVAL).
 * MAP_FIXED inside the block: with root_grants only, through the block's own reference (bounds over
 * the whole block) and only while the program derived no object below it (the refill rule; EINVAL
 * otherwise, EOPNOTSUPP without root_grants); the range then reads zero with protection a2, removed
 * pages count as mapped again, and no lifetime ends. The same VMAs keep the range (stage 4f: mprotect
 * and MADV_DONTNEED, no new mapping), and a refusal by Linux (ENOMEM: a VMA split at the map count
 * limit) changes nothing. A mapping that a worker's transport holds answers EBUSY. */
#define CV_SERVICE_REFILL 17
/* Stage 4f. mremap in place: a0 the block's reference, its cursor the first page of the old range; a1
 * the old length; a2 the new length (whole pages, not a1). The kernel validates the whole old range
 * first, as the mapping check does without CV_CHECK_MOVE (EINVAL, EBUSY, EFAULT). A shorter length
 * removes the tail [a0 + a2, a0 + a1) as a partial munmap does (the references stay valid, and the block
 * keeps at least the pages of the new length). A longer one refills the new tail [a0 + a1, a0 + a2)
 * under the refill rule of MAP_FIXED: every page of it removed and inside the block's visible length,
 * no object derived below the block's reference (ENOMEM otherwise, and always without root_grants:
 * the libc then moves with MREMAP_MAYMOVE); the tail reads zero with the protection of the old range's
 * last page. A refusal by Linux (ENOMEM) changes nothing. */
#define CV_SERVICE_RESIZE 18
/* Node instructions pause at this remaining cleanup reserve. */
#define CV_NODE_RESERVE 256
#define CV_MAP_HEAP 0
#define CV_MAP_APPLICATION 1
#define CV_MAP_METADATA 2
/* Resource policy of the QEMU application profile, shared by libc, the
 * launcher and the module: one registered region, and all regions of one
 * context. Heap regions are backed and pinned page by page as they are
 * touched, so registered VA may exceed what is resident. */
#define CV_MAX_REGION_BYTES (2UL << 30)
#define CV_MAX_BYTES (4UL << 30)
/* A registration the module pins in full at once (the image, its data and
 * its stacks) is checked page against page, so it keeps the first profile's
 * size; no such region comes near it. */
#define CV_MAX_EAGER_BYTES (256UL << 20)
#endif
