#ifndef CAPSTONE_VIRTUAL_WIRE_H
#define CAPSTONE_VIRTUAL_WIRE_H
#include "vm-abi.h"
#include <linux/types.h>
#include <linux/ioctl.h>

/* Experimental one-hart adapter; all thread and mapping addresses belong to
 * the owning Linux mm and share one lifetime table. */
struct cv_map {
    __u64 thread, address, bytes, permissions, reg, cursor, linear;
    __u64 id;
};
struct cv_step {
    /* Zero is the original process thread. Other IDs are allocated by
     * CV_THREAD_CREATE and name a paused CSRUNV context. */
    __u64 thread;
    __u64 reply, result;
    __u64 kind, cause, pc, address, args[8];
};
/* A start frame is a page-aligned, writable user page containing the virtual
 * CSRUNV frame format. The trusted adapter fills satp and srevroot, then
 * consumes the tagged register slots when the thread first runs. */
struct cv_thread_create {
    __u64 frame;
    __u64 thread;
};
struct cv_thread_control {
    __u64 thread;
    __u64 frame;
};
struct cv_remap {
    __u64 thread, id, bytes, address; /* address zero cancels after Linux failure */
};
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
#define CV_THREAD_EXIT _IOW('V', 23, struct cv_thread_control)
#define CV_NODES _IOW('V', 24, struct cv_nodes)
#define CV_REMAP_BEGIN _IOW('V', 25, struct cv_remap)
#define CV_REMAP_END _IOW('V', 26, struct cv_remap)
#define CV_PROFILE _IOR('V', 27, __u64)
#define CV_MAX_ARENAS 32768
#define CV_MAX_REGION_BYTES (1024UL << 20)
#define CV_MAX_BYTES (4096UL << 20)
#endif
