/* Lane AB experiment (experiment/in-module-io): a DIRECT delegated read/write, served by the
 * module inside CV_STEP straight into or out of the domain's own buffer.
 *
 * The request travels as an ordinary delegate entry (capstone/delegate.h layout, version 2) in a
 * block of the caller's own, named by the THREAD_DELEGATE service's a0; `flags` is
 * CV_IO_DIRECT alone and args[1] is the buffer's ADDRESS (its cursor), not an exchange offset.
 * The module re-validates everything (see capstone_vm.c, inline_io): the block and the buffer
 * must each lie in one registered arena the domain may be written through, and it takes a hold on
 * both arenas for the whole call. The module never falls back to the launcher for a DIRECT
 * request: it either serves it or answers it (EFAULT, EBADF, ENOSYS, ...), so this works only on
 * a module built from that branch with in_module_io set; elsewhere the launcher refuses the
 * unknown block and the run ends.
 *
 * What the module cannot check is the CAPABILITY: trusted S has no capability access on this ISA
 * (isa.md, CSMINT, "LDC/STC with an integer base read 24 in every mode"; no CSCHECKR/CSCHECKW), so
 * its unit of authority is the arena, not the object. io_direct_checked() therefore checks the
 * capability here, at submission, as the libc's dl_buffer_ok does; io_direct() does not, which is
 * what the adversarial test needs to show what the module alone guarantees. */
#ifndef CAPSTONE_VIRTUAL_INLINE_IO_H
#define CAPSTONE_VIRTUAL_INLINE_IO_H
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include "vm-abi.h"

#define CV_IO_META_BYTES 16384UL

struct io_entry {
    uint32_t version, count;
    uint64_t nr;
    uint64_t args[6];
    uint64_t flags;
    int64_t result;
    uint64_t pending, status;
};

struct io_transport {
    volatile struct io_entry *entry;
    char *meta, *exchange;
};

extern void __capstone_vm_delegate_transport(void *, void *, size_t);

static inline int io_transport_init(struct io_transport *t)
{
    t->meta = mmap(0, CV_IO_META_BYTES, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    t->exchange = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (t->meta == MAP_FAILED || t->exchange == MAP_FAILED) return -1;
    t->entry = (volatile struct io_entry *)t->meta;
    /* the block's signal handover (offset 4096) is never read for a DIRECT round, but the
     * module clears its count there as the launcher's publication does: touch it here */
    t->meta[4096] = 0;
    return 0;
}

/* The request names an ADDRESS: the module sees no capability. */
static inline long io_direct_at(struct io_transport *t, long nr, long fd, unsigned long address,
                                size_t bytes, long offset)
{
    volatile struct io_entry *e = t->entry;
    e->version = 2; e->count = 0; e->nr = (uint64_t)nr;
    e->args[0] = (uint64_t)fd;
    e->args[1] = address;
    e->args[2] = bytes; e->args[3] = (uint64_t)offset; e->args[4] = e->args[5] = 0;
    e->flags = CV_IO_DIRECT; e->result = -ENOSYS; e->pending = 0; e->status = ~0ULL;
    __capstone_vm_delegate_transport(t->meta, t->exchange, 4096);
    return (long)e->result;
}
static inline long io_direct(struct io_transport *t, long nr, long fd, void *buffer,
                             size_t bytes, long offset)
{
    return io_direct_at(t, nr, fd, __builtin_capstone_cap_get_cursor(buffer), bytes, offset);
}

/* The capability check the module cannot make, at submission (the libc's dl_buffer_ok): a
 * LINEAR or NONLIN data capability with the rights and the span. It cannot see a capability
 * revoked AFTER it, during a blocking call: that is the module's hold, at arena granularity. */
static inline int io_buffer_ok(const void *buffer, size_t bytes, unsigned rights)
{
    unsigned long type, base, cursor, end, perms;
    if (!bytes) return 1;
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(type) : "r"(buffer));
    if (type != 0 && type != 1) return 0;
    base = __builtin_capstone_cap_get_base((void *)buffer);
    end = __builtin_capstone_cap_get_end((void *)buffer);
    cursor = __builtin_capstone_cap_get_cursor((void *)buffer);
    perms = __builtin_capstone_cap_get_perm((void *)buffer);
    return (perms & rights) == rights && base <= cursor && cursor <= end && bytes <= end - cursor;
}

static inline long io_direct_checked(struct io_transport *t, long nr, long fd, void *buffer,
                                     size_t bytes, long offset, unsigned rights)
{
    if (!io_buffer_ok(buffer, bytes, rights)) return -EFAULT;
    return io_direct(t, nr, fd, buffer, bytes, offset);
}
#endif
