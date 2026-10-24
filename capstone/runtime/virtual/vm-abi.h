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
#define CV_SERVICE_THREAD_CREATE 3
#define CV_SERVICE_THREAD_EXIT 4
#define CV_SERVICE_THREAD_JOIN 5
#define CV_SERVICE_PROTECT 6
#define CV_SERVICE_WAIT 7
#define CV_SERVICE_FUTEX 8
#define CV_SERVICE_PTHREAD_EXIT 9
#define CV_SERVICE_THREAD_DELEGATE 10
#define CV_SERVICE_THREAD_SELF 11
#define CV_SERVICE_NODES 12
#define CV_SERVICE_REMAP 13
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
