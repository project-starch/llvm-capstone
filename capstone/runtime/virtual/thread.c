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
/* delegate.c: the transport lock's context identity (single-context or service mode) */
extern uintptr_t __capstone_context_identity_take(void);
extern void __capstone_context_identity_give(uintptr_t identity);
extern void __capstone_context_identity_release(void);
extern int __capstone_delegate_has_private_transport(void *target_tp);

static void *thread_exit_entry(void *unused)
{
    (void)unused;
    capstone_virtual_thread_exit(NULL);
}

long capstone_virtual_thread_create(void *(*entry)(void *), void *argument,
                                    void *stack_top, void *thread_pointer)
{
    (void)argument;
    (void)thread_pointer;
    /* Only musl's pthread path initializes TLS, allocator locks and lifecycle
     * bookkeeping. A raw context cannot be made libc-safe by sharing TLS or
     * supplying a new TLS pointer. Reject before allocating any resources. */
    errno = (!entry || !stack_top) ? EINVAL : ENOTSUP;
    return -1;
}

long __capstone_virtual_context_create(void *(*entry)(void *), void *argument,
                                       void *stack_top, void *thread_pointer)
{
    unsigned char *frame;
    long id;

    if (!entry || !stack_top) {
        errno = EINVAL;
        return -1;
    }
    /* A pthread's TLS contains its private transport and musl lifecycle
     * state. A second raw context cannot attach those held mappings.
     * Refuse this unsupported sharing before creating a worker, rather than
     * letting its first delegated call terminate the whole application. */
    if (__capstone_delegate_has_private_transport(thread_pointer)) {
        errno = ENOTSUP;
        return -1;
    }
    /* mmap in the virtual profile returns page-aligned, private anonymous
     * memory. One page is enough for the fixed CSRUNV start frame and gives
     * the kernel an unambiguous pinning unit. */
    /* Grants are copyable; retain a local slot while checking acquisition. */
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
     * thread state, so every same-mm child inherits it -- the same bounds,
     * permissions and cursor, __global_pointer$, which gp-relaxed global
     * accesses read (delegate.c, "THE OWNER'S IDENTITY"). Taking it switches
     * the transport lock to service identities; it fails with EDEADLK when
     * this context holds the wire (a create from a handler mid-round). */
    uintptr_t identity = __capstone_context_identity_take();
    if (!identity) {
        int error = errno == EDEADLK ? EDEADLK : EAGAIN;
        __capstone_vm_unmap((uintptr_t)frame, 4096);
        errno = error;
        return -1;
    }
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 1 * 16),
                       (void *)thread_exit_entry);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 2 * 16), stack_top);
    __asm__ volatile("scc t0, gp, %1\n\tstc t0, 0(%0)"
                     : : "r"(frame + 64 + 3 * 16), "r"(identity) : "t0", "memory");
    if (thread_pointer)
        capstone_cap_store((capstone_cap_slot *)(frame + 64 + 4 * 16), thread_pointer);
    capstone_cap_store((capstone_cap_slot *)(frame + 64 + 10 * 16), argument);
    capstone_cap_store((capstone_cap_slot *)(frame + 64), (void *)entry);

    id = __capstone_vm_thread_create_frame(frame);
    if (id < 0) {
        int error = (int)-id;
        __capstone_context_identity_give(identity);
        __capstone_vm_unmap((uintptr_t)frame, 4096);
        errno = error ? error : EAGAIN;
    }
    return id;
}

_Noreturn void capstone_virtual_thread_exit(void *result)
{
    (void)result;
    /* The last step before the exit service: no delegated call follows. */
    __capstone_context_identity_release();
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
