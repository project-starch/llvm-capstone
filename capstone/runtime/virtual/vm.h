#ifndef CAPSTONE_RUNTIME_VIRTUAL_VM_H
#define CAPSTONE_RUNTIME_VIRTUAL_VM_H
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include "../../sublet/sublet.h"
#include "vm-abi.h"

/* Internal libc interface. A grant must immediately be moved into a slot: it
 * is linear, unless the adapter runs with root_grants, where every grant is a
 * copyable child of the address-space reference. cap_vm_copyable accepts both.
 * Public mmap makes its grant copyable; heap allocators retain ownership and
 * derive separate object lifetimes. All grants belong to the owning virtual
 * address space, including its other threads. */
#define CAP_VM_HEAP CV_MAP_HEAP
#define CAP_VM_MAPPING CV_MAP_APPLICATION
#define CAP_VM_METADATA CV_MAP_METADATA
extern void *__capstone_vm_acquire(unsigned long, unsigned long, int, unsigned, unsigned);
extern void *__capstone_vm_remap(unsigned long, unsigned long, unsigned long);
/* The unmap service takes the mapping's reference itself in a0, with its tag, and the length in
 * a1. With root_grants the kernel ends the mapping only through that reference: a live reference
 * to the mapping (not to an object inside it, not a narrowed one) whose bounds cover the whole
 * mapping as mmap exposed it, and a1 must be that whole length. Without it the launcher reads the
 * address from a0, which is the reference's cursor. The thread-create service takes the frame's
 * reference in a0 and the exit service the stack's reference in a1, under the same rule. The
 * protect service takes the reference in a0 as well: with root_grants it must be a live reference
 * to the mapping, not to an object inside it, whose bounds cover the page range that a0's cursor
 * and a1 name; the range may be part of the mapping. */
extern long __capstone_vm_unmap(void *, unsigned long);
extern long __capstone_vm_protect(void *, unsigned long, int);
extern long __capstone_vm_wait(void);
extern long __capstone_vm_nodes(unsigned long);

/* Internal context primitive for musl's __clone and directed runtime tests.
 * It does not initialize libc thread state and is not an application API. */
__attribute__((visibility("hidden")))
long __capstone_virtual_context_create(void *(*entry)(void *), void *argument,
                                       void *stack_top, void *thread_pointer);

static inline int cap_vm_acquire(sublet_cap *slot, size_t bytes,
                                 size_t alignment, int prot,
                                 unsigned rights, unsigned kind)
{
    unsigned long raw;
    sublet_store(slot, __capstone_vm_acquire(bytes, alignment, prot, rights, kind));
    __asm__ volatile("ld %0, 0(%1)" : "=r"(raw) : "r"(slot) : "memory");
    if ((long)raw <= 0) {
        errno = raw ? -(long)raw : ENOMEM;
        sublet_clear(slot);
        return -1;
    }
    return 0;
}
/* The grant in [slot] as a copyable reference. A linear grant is delinearised.
 * A grant that is already non-linear (LCC type 1) is returned as loaded: DELIN
 * refuses everything but a linear capability with cause 26. Any other content
 * of the slot reaches DELIN and faults there, as before. */
static inline void *cap_vm_copyable(sublet_cap *slot)
{
    void *p;
    __asm__ volatile("ldc %0, 0(%1)\n"
                     ".insn r 0x5b, 0x1, 0x04, t0, %0, x1\n"
                     "addi t0, t0, -1\n"
                     "beqz t0, 1f\n"
                     "delin %0\n"
                     "1:\n"
                     : "=&r"(p) : "r"(slot) : "t0", "memory");
    return p;
}
#endif
