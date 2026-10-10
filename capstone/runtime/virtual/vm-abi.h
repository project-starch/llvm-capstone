#ifndef CAPSTONE_VIRTUAL_VM_ABI_H
#define CAPSTONE_VIRTUAL_VM_ABI_H
/* Version 5 compiles mallocng inside Capstone and adds only a VM-remap
 * service. It requires the explicit exact-bounds QEMU profile. Versions 2/3
 * remain accepted by the launcher; the withdrawn native-heap v4 is rejected. */
#define CV_IMAGE_MAGIC_V2 0x324d56564e4f5043
#define CV_IMAGE_MAGIC_V3 0x334d56564e4f5043
#define CV_IMAGE_MAGIC 0x354d56564e4f5043
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
#endif
