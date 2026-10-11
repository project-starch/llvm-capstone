#include <capstone/capability.h>
#include <capstone/virtual.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "vm.h"

extern long __capstone_vm_thread_create_frame(void *frame);
extern void __capstone_vm_thread_exit(void);
extern long __capstone_vm_thread_join(long thread);

static void *thread_exit_entry(void *unused)
{
    (void)unused;
    capstone_virtual_thread_exit(NULL);
}

long capstone_virtual_thread_create(void *(*entry)(void *), void *argument,
                                    void *stack_top, void *thread_pointer)
{
    unsigned char *frame;
    long id;

    if (!entry || !stack_top) {
        errno = EINVAL;
        return -1;
    }
    /* mmap in the virtual profile returns page-aligned, private anonymous
     * memory. One page is enough for the fixed CSRUNV start frame and gives
     * the kernel an unambiguous pinning unit. */
    /* vm_map returns a linear grant.  Keep that grant in a slot and
     * delinearise a loaded copy before C code stores or clears the frame;
     * passing the linear return through a C variable loses its tag. */
    {
        sublet_cap slot;
        if (cap_vm_acquire(&slot, 4096, 4096, PROT_READ | PROT_WRITE,
                           6, CAP_VM_HEAP)) frame = NULL;
        else frame = cap_vm_copyable(&slot);
    }
    if (!frame) {
        errno = ENOMEM;
        return -1;
    }
    memset(frame, 0, 4096);
    ((uint64_t *)frame)[7] = 3; /* service and page-fault events */

    /* Initial x1 is a return trampoline; x2 is the caller-selected stack,
     * x3 is the process image capability, x4 is the optional per-thread TLS
     * pointer, and x10 is the argument.  gp is process authority rather than
     * thread state, so every same-mm child inherits it. */
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16),
                       (void *)thread_exit_entry);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack_top);
    __asm__ volatile("stc gp, 0(%0)" : : "r"(frame + 64 + 3 * 16) : "memory");
    if (thread_pointer)
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 4 * 16), thread_pointer);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 10 * 16), argument);
    capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)entry);

    id = __capstone_vm_thread_create_frame(frame);
    if (id < 0) {
        int error = (int)-id;
        __capstone_vm_unmap((uintptr_t)frame, 4096);
        errno = error ? error : EAGAIN;
    }
    return id;
}

_Noreturn void capstone_virtual_thread_exit(void *result)
{
    (void)result;
    __capstone_vm_thread_exit();
    for (;;) __asm__ volatile("ebreak");
}

int capstone_virtual_thread_join(long thread)
{
    long rc;
    if (thread <= 0) {
        errno = EINVAL;
        return -1;
    }
    rc = __capstone_vm_thread_join(thread);
    if (rc < 0) {
        errno = (int)-rc;
        return -1;
    }
    return 0;
}
