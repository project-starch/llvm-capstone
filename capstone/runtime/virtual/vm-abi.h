#ifndef CAPSTONE_VIRTUAL_VM_ABI_H
#define CAPSTONE_VIRTUAL_VM_ABI_H
/* Shared by assembly, libc and the trusted launcher. Version 4 adds the
 * native mallocng service. A v4 launcher accepts v2/v3 images; older
 * launchers reject v4 images. Existing mapping/thread services are unchanged. */
#define CV_IMAGE_MAGIC_V2 0x324d56564e4f5043
#define CV_IMAGE_MAGIC_V3 0x334d56564e4f5043
#define CV_IMAGE_MAGIC 0x344d56564e4f5043
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
#define CV_SERVICE_HEAP 13
#define CV_HEAP_MALLOC 0
#define CV_HEAP_CALLOC 1
#define CV_HEAP_REALLOC 2
#define CV_HEAP_FREE 3
#define CV_HEAP_ALIGNED 4
#define CV_HEAP_LINEAR 5
#define CV_HEAP_FREE_LINEAR 6
#define CV_HEAP_STATISTICS 7
/* Node instructions pause at this remaining cleanup reserve. */
#define CV_NODE_RESERVE 256
#define CV_MAP_HEAP 0
#define CV_MAP_APPLICATION 1
#define CV_MAP_METADATA 2
#endif
