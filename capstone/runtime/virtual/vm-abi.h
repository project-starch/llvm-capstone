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
#define CV_SERVICE_THREAD_JOIN 5
/* a0: the reference of the mapping, its cursor the first page; a1 the length; a2 the protection.
 * With root_grants the kernel checks the reference as for munmap, except that the range may be
 * any page range inside the mapping, which the reference must cover. It ends nothing. */
#define CV_SERVICE_PROTECT 6
#define CV_SERVICE_WAIT 7
#define CV_SERVICE_FUTEX 8
/* a0 the exit word, a1 the reference of the mapping that holds the thread's stack (or 0), a2 its
 * length, a3 the status. With root_grants the stack is unmapped only through a1, as for munmap.
 * A refused stack does not stop the exit: the mapping stays, and the program can still unmap it
 * through a reference it kept. */
#define CV_SERVICE_PTHREAD_EXIT 9
#define CV_SERVICE_THREAD_DELEGATE 10
/* The calling context's launcher id + 1. Since the transport lock reads its identity from gp's
 * cursor (delegate.c dl_wire_identity), only a context created with no identity slot free asks. */
#define CV_SERVICE_THREAD_SELF 11
#define CV_SERVICE_NODES 12
#define CV_SERVICE_REMAP 13
/* A worker gives its transport (meta block and exchange region) back: the launcher stops
 * writing into them and releases its hold on their mappings, which until then refuses their
 * unmap to everyone, the worker included. The worker's next delegate service attaches again. */
#define CV_SERVICE_TRANSPORT_DETACH 14
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
