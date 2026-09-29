/* Runtime-internal locks; see capstone/lock.h. */
#include "libc.h"            /* hidden, for musl's lock.h */
#include "lock.h"            /* musl's __lock and __unlock */
#include <capstone/lock.h>

void __capstone_signals_deliver(void);   /* signals.c */
extern int __capstone_tls_ready;   /* tls.c */

/* Per context: its TLS block. Every lock and unlock is counted, whether or not
   musl really took the lock (it does not while the application has one
   context), except before the first context has a TLS block at all: its
   calloc takes the heap lock, and releases it, before tp exists. */
static __thread int depth;

int __capstone_lock_depth(void) { return __capstone_tls_ready ? depth : 0; }

void capstone_lock(volatile int *word)
{
	__lock(word);
	if (__capstone_tls_ready)
		++depth;
}

void capstone_unlock(volatile int *word)
{
	__unlock(word);
	if (__capstone_tls_ready && !--depth)
		__capstone_signals_deliver();
}

/* Times a spin lock was found taken: evidence that its holder was preempted
   inside and another context waited (thread-probe b9 reads it). */
static unsigned long contended;
unsigned long __capstone_spin_contended(void) { return contended; }

/* A spin lock is a leaf and makes no round while held, so no handler can run
   under it: it does not count towards the depth. */
void capstone_spin_lock(volatile int *word)
{
	if (!__atomic_exchange_n(word, 1, __ATOMIC_ACQUIRE))
		return;
	__atomic_fetch_add(&contended, 1, __ATOMIC_RELAXED);
	do
		while (__atomic_load_n(word, __ATOMIC_RELAXED))
			;
	while (__atomic_exchange_n(word, 1, __ATOMIC_ACQUIRE));
}

void capstone_spin_unlock(volatile int *word)
{
	__atomic_store_n(word, 0, __ATOMIC_RELEASE);
}
