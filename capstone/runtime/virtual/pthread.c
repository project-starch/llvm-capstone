#ifdef CAPSTONE_RUNTIME_TRAP
#define _GNU_SOURCE
#include <sched.h>
#include <errno.h>
#include <stddef.h>
#include <stdarg.h>
#include <time.h>
#include <syscall.h>

extern long __capstone_linux_clone(void *, void *, unsigned long, void *, int *, void *, int *);

int __clone(int (*entry)(void *), void *stack, int flags, void *arg, ...)
{
    const unsigned required = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
        CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID;
    const unsigned allowed = required | CLONE_SYSVSEM | CLONE_DETACHED;
    va_list ap;
    va_start(ap, arg);
    int *parent = va_arg(ap, int *);
    void *tls = va_arg(ap, void *);
    int *clear = va_arg(ap, int *);
    va_end(ap);
    if ((flags & required) != required || (flags & ~allowed)) return -ENOSYS;
    if (!entry || !stack || !parent || !tls || !clear) return -EINVAL;
    return __capstone_linux_clone((void *)entry, stack, flags, arg, parent, tls, clear);
}

/* __unmapself is assembly: after munmap neither stack nor TLS may be used. */
_Noreturn void __capstone_virtual_pthread_exit(long status)
{
    __syscall(SYS_exit, status);
    for (;;) __asm__ volatile("ebreak");
}

long __capstone_virtual_futex(void *word, int op, int value,
                             const struct timespec *timeout, int bitset)
{
    int command = op & 127;
    if (command != 0 && command != 1 && command != 9 && command != 10) return -ENOSYS;
    return __syscall(SYS_futex, word, op | 128, value, timeout, 0, bitset);
}
#else
/* musl owns pthread objects, TLS, locks and join semantics. Linux workers
 * execute the existing virtual contexts. Only clone and final exit differ. */
#define _GNU_SOURCE
#include <locale.h>
#include "pthread_impl.h"
#include <capstone/virtual.h>
#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>
#include <fenv.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include "capstone/delegate.h"
#include "vm.h"

extern _Noreturn void __capstone_vm_thread_exit_clear(uintptr_t, uintptr_t, size_t, long);
extern long __capstone_vm_futex(void *, int, int, long, long, int);
/* delegate.c: this context gives back the identity its creator took (thread.c), right before
 * its exit service, after its last delegated call (release_clone's frees and unmaps). */
extern void __capstone_context_identity_release(void);
struct virtual_clone {
    int (*entry)(void *);
    void *arg;
    int *clear;
    void *transport;
    void *meta;
    void *exchange;
    void *signals;
    fenv_t fenv;
};
static __thread struct virtual_clone *current_clone;
extern size_t __capstone_delegate_thread_state_size(void);
extern void __capstone_delegate_thread_attach(void *, void *, void *, void *);
extern size_t __capstone_signals_thread_state_size(void);
extern uint64_t __capstone_sigmask_current(void);
extern void __capstone_signals_thread_detach(void);

/* A worker's meta block and exchange region are whole mappings of their own, not heap blocks:
 * the launcher writes them while the worker is attached and holds their mappings against
 * retirement until the worker detaches (exec.c, struct mapping.holder); it refuses an
 * attachment whose block shares a mapping, so each block is mapped on its own. The worker
 * detaches (__capstone_vm_transport_detach) and then gives the blocks back with one munmap
 * each; before the detach that munmap is refused, to the worker too. */
#define VIRTUAL_EXCHANGE_BYTES (1UL << 20)
extern long __capstone_vm_transport_detach(void);
extern void __capstone_delegate_thread_detach(void);
static void release_transport(struct virtual_clone *r)
{
    if (!r->transport) return;
    __capstone_signals_thread_detach();
    __capstone_delegate_thread_detach();
    free(r->transport);
    r->transport = NULL;
    __capstone_vm_transport_detach();
    munmap(r->meta, CAPSTONE_DELEGATE_META_BYTES);
    munmap(r->exchange, VIRTUAL_EXCHANGE_BYTES);
    r->meta = r->exchange = NULL;
}
static void release_clone(struct virtual_clone *r)
{
    release_transport(r);
    free(r);
}
static void *transport_block(size_t bytes)
{
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
/* The calling worker's exchange region (NULL on the main thread): what a program needs to name
 * the mapping, as the transport-lifetime contract does. */
void *__capstone_virtual_exchange(void)
{
    return current_clone ? current_clone->exchange : NULL;
}
/* The calling worker gives its transport back before it exits, as its exit would: detach, then
 * unmap both blocks. Its later calls (musl's own exit path makes some) go through the process
 * transport. The contract's identity case needs a worker alive without a transport. */
void __capstone_virtual_transport_release(void)
{
    if (current_clone) release_transport(current_clone);
}

static void *clone_entry(void *p)
{
    struct virtual_clone *r = p;
    current_clone = r;
    /* A thread starts with its creator's floating-point environment (C11
     * 7.6; Linux copies it at clone). Each context has its own fcsr and
     * begins with the default one. */
    fesetenv(&r->fenv);
    __capstone_delegate_thread_attach(r->transport, r->meta, r->exchange, r->signals);
    int result = r->entry(r->arg);
    int *clear = r->clear;
    release_clone(r);
    __capstone_context_identity_release();
    __capstone_vm_thread_exit_clear((uintptr_t)clear, 0, 0, result);
}

int __clone(int (*entry)(void *), void *stack, int flags, void *arg, ...)
{
    const int required = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
                         CLONE_THREAD | CLONE_SETTLS | CLONE_PARENT_SETTID | CLONE_CHILD_CLEARTID;
    const int allowed = required | CLONE_SYSVSEM | CLONE_DETACHED;
    va_list ap;
    va_start(ap, arg);
    int *parent = va_arg(ap, int *);
    void *tls = va_arg(ap, void *);
    int *clear = va_arg(ap, int *);
    va_end(ap);
    if ((flags & required) != required || (flags & ~allowed)) return -ENOSYS;
    if (!parent || !tls || !clear) return -EINVAL;
    struct virtual_clone *r = malloc(sizeof *r);
    if (!r) return -ENOMEM;
    *r = (struct virtual_clone){.entry = entry, .arg = arg,
        .clear = flags & CLONE_CHILD_CLEARTID ? clear : NULL};
    fegetenv(&r->fenv);
    r->transport = calloc(1, __capstone_delegate_thread_state_size());
    r->meta = transport_block(CAPSTONE_DELEGATE_META_BYTES);
    r->exchange = transport_block(VIRTUAL_EXCHANGE_BYTES);
    r->signals = calloc(1, __capstone_signals_thread_state_size());
    if (!r->transport || !r->meta || !r->exchange || !r->signals) {
        free(r->signals); free(r->transport);
        if (r->meta) munmap(r->meta, CAPSTONE_DELEGATE_META_BYTES);
        if (r->exchange) munmap(r->exchange, VIRTUAL_EXCHANGE_BYTES);
        free(r);
        return -ENOMEM;
    }
    ((struct capstone_signal_block *)((char *)r->meta + CAPSTONE_SIGNAL_OFFSET))->initial_mask =
        __capstone_sigmask_current();
    long id = __capstone_virtual_context_create(clone_entry, r, stack, tls);
    if (id < 0) {
        int error = errno;
        free(r->signals); free(r->transport);
        munmap(r->meta, CAPSTONE_DELEGATE_META_BYTES); munmap(r->exchange, VIRTUAL_EXCHANGE_BYTES);
        free(r);
        return -error;
    }
    /* pthread_create holds the thread-list lock until __clone returns;
     * musl's child cannot finish before the parent publishes this identity. */
    /* From 2^22 up, as context.c mints them: above Linux's pid range, and far below bit 30,
     * which musl's lock words keep for MAYBE_WAITERS. A tid with that bit set never matched
     * itself in __lockfile, so a nested stdio lock (puts -> fwrite) in a pthread waited on its
     * own lock forever. Launcher ids are never reused, so neither is a tid. */
    int tid = 0x400000 + (int)id;
    if (flags & CLONE_PARENT_SETTID) *parent = tid;
    return tid;
}

_Noreturn void __capstone_virtual_pthread_exit(long status)
{
    struct virtual_clone *r = current_clone;
    int *clear = r ? r->clear : NULL;
    /* Free the record while the child still owns its stack and TLS. The
     * trusted launcher releases the clear word only after stopping it. */
    if (r) release_clone(r);
    __capstone_context_identity_release();
    __capstone_vm_thread_exit_clear((uintptr_t)clear, 0, 0, status);
}

_Noreturn void __unmapself(void *base, size_t size)
{
    struct virtual_clone *r = current_clone;
    int *clear = r ? r->clear : NULL;
    if (r) release_clone(r);
    __capstone_context_identity_release();
    __capstone_vm_thread_exit_clear((uintptr_t)clear, (uintptr_t)base, size, 0);
}

long __capstone_virtual_futex(void *word, int op, int value, const struct timespec *timeout, int bitset)
{
    int command = op & 127;
    /* musl has a wake fallback when requeue is unsupported. Its fourth
     * argument is an integer, not a timeout pointer, for this operation. */
    if (command != 0 && command != 1 && command != 9 && command != 10) return -ENOSYS;
    /* The original capability performs the liveness/read check. Linux gets
     * the registered address, never a bounced copy of the futex word. */
    (void)__atomic_load_n((int *)word, __ATOMIC_ACQUIRE);
    long seconds = timeout ? timeout->tv_sec : -1;
    long nanos = timeout ? timeout->tv_nsec : 0;
    return __capstone_vm_futex(word, op, value, seconds, nanos, bitset);
}

#endif
