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
#define CV_SERVICE_MAP 1
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
 * that holds a worker's transport answers EBUSY, as for munmap. */
#define CV_SERVICE_PROTECT 6
#define CV_SERVICE_WAIT 7
/* a0 the word's reference, a1 the operation (FUTEX_WAIT, FUTEX_WAKE and their BITSET forms), a2
 * the value, a3 and a4 the timeout (seconds -1: none), a5 the bitset. A wait reads the word, so
 * the kernel checks it as an exit word, without the conditions on writing: with root_grants a live
 * reference whose bounds cover the word, and in both modes an aligned int inside one mapping and in
 * no thread's frame (EFAULT otherwise). A wake neither reads nor writes the word, so an integer
 * address is enough; it must lie inside one registered mapping (EFAULT otherwise). */
#define CV_SERVICE_FUTEX 8
/* a0 the exit word's reference (or 0), a1 the reference of the mapping that holds the thread's
 * stack (or 0), a2 its length, a3 the status. With root_grants the stack is unmapped only through
 * a1, as for munmap. A refused stack does not stop the exit: the mapping stays, and the program can
 * still unmap it through a reference it kept. A stack of the wrong length and one that holds the
 * exit word are refused this way. A stack that is the thread's own frame is refused as a stack too,
 * but the exit releases the frame, which ends that mapping. After the thread has stopped the
 * launcher stores 0 into the exit word and wakes its futex waiters, if the word is an aligned int
 * inside one mapping, in pages that the program's grant may write and that Linux maps writable,
 * outside the launcher's own transport and every worker's, and if the kernel accepts it: with
 * root_grants a0 must be a live reference whose bounds cover the word, and in both modes the word
 * may not lie in the frame of any thread. A word that fails is not written (the launcher logs the
 * refusal) and the exit still completes; futex waiters on a refused word stay blocked, which is
 * the program's bug. */
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
/* a0 the reference of a mapping, a1 its length. Answers 0 if a request to end that mapping would
 * be accepted (as for munmap: with root_grants a live reference to the whole mapping, and a1 its
 * whole visible length; EINVAL otherwise, EBUSY while a worker's transport holds it) and every
 * page of it is read-write (ENOTSUP otherwise). Changes nothing. The libc's mremap asks it before
 * it copies anything. */
#define CV_SERVICE_MAPPING_CHECK 15
/* a0 the reference of a mapping, its cursor the first page; a1 the length; a2 MADV_DONTNEED or
 * MADV_FREE (EINVAL otherwise). The range reads zero afterwards. With root_grants the kernel checks
 * the reference as for mprotect: a live reference to the mapping, not to an object inside it,
 * whose bounds cover the page range. It ends nothing. A mapping that holds a worker's transport
 * answers EBUSY, as for mprotect. */
#define CV_SERVICE_ADVISE 16
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
