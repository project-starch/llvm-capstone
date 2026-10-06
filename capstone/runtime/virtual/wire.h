#ifndef CAPSTONE_VIRTUAL_WIRE_H
#define CAPSTONE_VIRTUAL_WIRE_H
#include <linux/types.h>
#include <linux/ioctl.h>

/* Experimental one-hart adapter; addresses belong to the owning Linux mm. */
struct cv_map {
    __u64 address, bytes, permissions, reg, cursor, linear;
    __u64 id;
};
struct cv_step {
    __u64 reply, result;
    __u64 kind, cause, pc, address, args[8];
};
struct cv_stats {
    __u64 arenas, pinned_pages, peak_pages, nodes, steps, faults;
    __u64 collections, reclaimed, nodes_high_water, nodes_live, nodes_retired;
};
struct cv_global {
    __u64 contexts, arenas, pinned_pages, nodes_allocated, nodes_high_water;
    __u64 nodes_live, nodes_retired, collections, reclaimed;
};
#define CV_ADD _IOWR('V', 16, struct cv_map)
#define CV_STEP _IOWR('V', 17, struct cv_step)
#define CV_RETIRE _IOW('V', 18, __u64)
#define CV_STATS _IOR('V', 19, struct cv_stats)
#define CV_RESOLVE _IO('V', 20)
#define CV_GLOBAL _IOR('V', 21, struct cv_global)
#define CV_MAX_ARENAS 512
#define CV_MAX_REGION_BYTES (256UL << 20)
#define CV_MAX_BYTES (1024UL << 20)
#define CV_NODE_ORDER 8
#define CV_NODE_BYTES (4096UL << CV_NODE_ORDER)
/* Runtime ECALL requests: 0 delegate, 1 mapping, 2 retirement. */
#define CV_SERVICE_DELEGATE 0
#define CV_SERVICE_MAP 1
#define CV_SERVICE_UNMAP 2
#endif
