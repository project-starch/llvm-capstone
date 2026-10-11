/* musl owns pthread objects, TLS, locks and join semantics. Linux workers
 * execute the existing virtual contexts. Only clone and final exit differ. */
#define _GNU_SOURCE
#include <locale.h>
#include "pthread_impl.h"
#include <capstone/virtual.h>
#include <stdarg.h>
#include <stdlib.h>
#include <errno.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#include "capstone/delegate.h"

extern _Noreturn void __capstone_vm_thread_exit_clear(uintptr_t, uintptr_t, size_t, long);
extern long __capstone_vm_futex(void *, int, int, long, long, int);
struct virtual_clone {
    int (*entry)(void *);
    void *arg;
    int *clear;
    void *transport;
    void *meta;
    void *exchange;
    void *signals;
};
static __thread struct virtual_clone *current_clone;
extern size_t __capstone_delegate_thread_state_size(void);
extern void __capstone_delegate_thread_attach(void *, void *, void *, void *);
extern size_t __capstone_signals_thread_state_size(void);
extern uint64_t __capstone_sigmask_current(void);
extern void __capstone_signals_thread_detach(void);

static void release_clone(struct virtual_clone *r)
{
    __capstone_signals_thread_detach();
    free(r->transport);
    free(r->meta);
    free(r->exchange);
    free(r);
}

static void *clone_entry(void *p)
{
    struct virtual_clone *r = p;
    current_clone = r;
    __capstone_delegate_thread_attach(r->transport, r->meta, r->exchange, r->signals);
    int result = r->entry(r->arg);
    int *clear = r->clear;
    release_clone(r);
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
    r->transport = calloc(1, __capstone_delegate_thread_state_size());
    r->meta = calloc(1, CAPSTONE_DELEGATE_META_BYTES);
    r->exchange = malloc(1UL << 20);
    r->signals = calloc(1, __capstone_signals_thread_state_size());
    if (!r->transport || !r->meta || !r->exchange || !r->signals) {
        free(r->signals); free(r->transport); free(r->meta); free(r->exchange); free(r);
        return -ENOMEM;
    }
    ((struct capstone_signal_block *)((char *)r->meta + CAPSTONE_SIGNAL_OFFSET))->initial_mask =
        __capstone_sigmask_current();
    long id = capstone_virtual_thread_create(clone_entry, r, stack, tls);
    if (id < 0) {
        int error = errno;
        free(r->signals); free(r->transport); free(r->meta); free(r->exchange); free(r);
        return -error;
    }
    /* pthread_create holds the thread-list lock until __clone returns;
     * musl's child cannot finish before the parent publishes this identity. */
    int tid = 0x40000000 + (int)id;
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
    __capstone_vm_thread_exit_clear((uintptr_t)clear, 0, 0, status);
}

_Noreturn void __unmapself(void *base, size_t size)
{
    struct virtual_clone *r = current_clone;
    int *clear = r ? r->clear : NULL;
    if (r) release_clone(r);
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
