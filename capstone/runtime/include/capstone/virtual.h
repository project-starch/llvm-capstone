#ifndef CAPSTONE_VIRTUAL_H
#define CAPSTONE_VIRTUAL_H

/* OS-backed thread entry for the trusted-kernel virtual adapter. The
 * caller supplies a code capability, an argument capability, and a stack
 * capability whose cursor is the initial sp. The adapter supplies satp and
 * the process lifetime root; no user code can select another address space.
 * thread_pointer must be a distinct, valid musl TLS pointer when the entry
 * uses errno, locale, cancellation or other thread-local state; NULL is
 * suitable only for entry code that avoids those facilities. */
long capstone_virtual_thread_create(void *(*entry)(void *), void *argument,
                                    void *stack_top, void *thread_pointer);

/* The entry function must call this or return through the installed exit
 * trampoline. It terminates only the current virtual thread. */
_Noreturn void capstone_virtual_thread_exit(void *result);

/* Wait until a child virtual thread has completed its exit protocol.  The
 * join is what keeps a caller-owned stack and TLS mapping live until the
 * child has stopped using them. */
int capstone_virtual_thread_join(long thread);

#endif
