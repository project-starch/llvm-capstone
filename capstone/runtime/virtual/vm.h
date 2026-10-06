#ifndef CAPSTONE_RUNTIME_VIRTUAL_VM_H
#define CAPSTONE_RUNTIME_VIRTUAL_VM_H
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#include "../../sublet/sublet.h"
#include "vm-abi.h"

/* Internal libc interface. Grants are linear and must immediately be moved
 * into a slot. Public mmap explicitly delinearises its grant; heap allocators
 * retain ownership and derive separate object lifetimes. All grants belong to
 * the owning virtual address space, including its other threads. */
#define CAP_VM_HEAP CV_MAP_HEAP
#define CAP_VM_MAPPING CV_MAP_APPLICATION
#define CAP_VM_METADATA CV_MAP_METADATA
#define CAP_VM_MAX_BYTES (256UL << 20)
extern void *__capstone_vm_acquire(unsigned long, unsigned long, int, unsigned, unsigned);
extern long __capstone_vm_unmap(unsigned long, unsigned long);
extern long __capstone_vm_protect(unsigned long, unsigned long, int);
extern long __capstone_vm_wait(void);

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
static inline void *cap_vm_copyable(sublet_cap *slot)
{
    void *p;
    __asm__ volatile("ldc %0, 0(%1)\ndelin %0"
                     : "=&r"(p) : "r"(slot) : "memory");
    return p;
}
#endif
