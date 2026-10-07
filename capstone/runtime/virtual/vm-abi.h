#ifndef CAPSTONE_VIRTUAL_VM_ABI_H
#define CAPSTONE_VIRTUAL_VM_ABI_H
/* Shared by the application assembly, libc and trusted launcher. Version 2
 * changes MAP's arguments; old virtual images must fail before execution. */
#define CV_IMAGE_MAGIC 0x324d56564e4f5043
#define CV_SERVICE_DELEGATE 0
#define CV_SERVICE_MAP 1
#define CV_SERVICE_UNMAP 2
#define CV_SERVICE_THREAD_CREATE 3
#define CV_SERVICE_THREAD_EXIT 4
#define CV_SERVICE_THREAD_JOIN 5
#define CV_SERVICE_PROTECT 6
#define CV_SERVICE_WAIT 7
#define CV_MAP_HEAP 0
#define CV_MAP_APPLICATION 1
#define CV_MAP_METADATA 2
#endif
