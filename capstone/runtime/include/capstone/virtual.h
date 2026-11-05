#ifndef CAPSTONE_VIRTUAL_H
#define CAPSTONE_VIRTUAL_H

/* Compatibility entry point: raw creation is unsupported and returns -1
 * with errno=ENOTSUP (EINVAL for a missing entry or stack). Use pthread_create
 * for application threads so musl owns TLS, locks and thread lifecycle.
 * Sharing TLS, supplying fresh TLS, and omitting TLS are all rejected. */
long capstone_virtual_thread_create(void *(*entry)(void *), void *argument,
                                    void *stack_top, void *thread_pointer);

/* Low-level compatibility primitive for runtime contexts. Application
 * pthreads must use pthread_exit or return, preserving musl cleanup. */
_Noreturn void capstone_virtual_thread_exit(void *result);

/* Low-level context join retained for runtime compatibility. Applications
 * use pthread_join, which also completes musl thread cleanup. */
int capstone_virtual_thread_join(long thread);

#endif
