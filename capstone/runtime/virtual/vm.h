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
extern long __capstone_vm_unmap(unsigned long, unsigned long);
extern long __capstone_vm_protect(unsigned long, unsigned long, int);
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
