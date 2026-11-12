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
/* CV_THREAD_EXIT. thread: the exiting thread; frame (out): its frame's user
 * address, as before. stack: the arena id of the stack the thread asks to
 * unmap at its exit (CV_SERVICE_PTHREAD_EXIT with a stack), or 0; stack_reg:
 * the register that holds the stack's reference, 11 (a1). With root_grants
 * the module checks that reference before it clears the frame and ends the
 * stack's mapping after the thread has stopped; stack_ended (out) is then 1.
 * A refused reference does not stop the exit: the thread exits, the stack's
 * mapping stays, and a later CV_RETIRE of it is refused. Without root_grants
 * the module ignores stack and stack_reg. */
struct cv_thread_exit {
    __u64 thread, frame, stack, stack_reg, stack_ended;
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
#define CV_THREAD_EXIT _IOWR('V', 23, struct cv_thread_exit)
#define CV_NODES _IOW('V', 24, struct cv_nodes)
#define CV_REMAP_BEGIN _IOW('V', 25, struct cv_remap)
#define CV_REMAP_END _IOW('V', 26, struct cv_remap)
#define CV_PROFILE _IOR('V', 27, __u64)
/* A span the launcher is about to WRITE on the domain's behalf through Linux: a thread's exit
 * word, a worker's meta block and exchange region. Every page of it inside a lazily admitted
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
/* With root_grants, CV_RETIRE releases only an arena whose mapping has already ended: through a
 * program's request (CV_UNMAP, the frame's hand-over at CV_THREAD_CREATE, the stack at
 * CV_THREAD_EXIT) or a park. While the mapping lives it is refused with EPERM and changes nothing.
 * CV_PARK is refused with EPERM: a park is a program's request and goes through CV_UNMAP. */
/* A program's own unmap (CV_SERVICE_UNMAP): thread is the caller, paused at the service, id the
 * arena the launcher resolved from the cursor of the reference in a0. park (in): 1 asks to pool
 * the mapping (CV_PARK's effect) where the module can, 0 to retire it. park (out): what the module
 * did, one of the CV_UNMAP_* values; the launcher unmaps the range unless it reads
 * CV_UNMAP_PARKED. One call decides, so a refused reference is refused and counted once. Without
 * root_grants the module ignores thread. With root_grants the reference is the authority: the
 * module reads it with its tag from the caller's saved a0 and ends the mapping only if it is a
 * live reference whose record is a child of the mapping's reservation record and whose bounds
 * cover the length the program's mmap exposed, and the call names exactly that length
 * (capstone_vm.c, request_operand). A refusal is EINVAL and changes nothing. */
struct cv_unmap {
    __u64 thread, id, park;
};
#define CV_UNMAP_RETIRED 0
#define CV_UNMAP_PARKED 1
#define CV_UNMAP_NO_POOL 2   /* retired: this module does not pool (unpinned) */
#define CV_UNMAP _IOWR('V', 32, struct cv_unmap)
/* The image file, by descriptor, before the launcher validates it: the module
 * denies writers to it (ETXTBSY, as Linux does for a running executable) until
 * the context is released, so the bytes validated are the bytes loaded. */
#define CV_IMAGE _IOW('V', 28, __s32)
#endif
