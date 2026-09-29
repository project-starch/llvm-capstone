#ifndef CAPSTONE_LOCK_H
#define CAPSTONE_LOCK_H

/* Runtime-internal locks (docs/plans/delegation-threads.md, Q6).
 *
 * capstone_lock is musl's __lock, a futex lock that costs nothing while the
 * application has one context (libc.need_locks), taken by the heap, the
 * context arena and the spawn request. capstone_spin_lock is a leaf: the
 * capability-width atomics take it and nothing else is taken under it; a
 * waiter spins, and on one hart a preempted holder runs again at the next
 * quantum. The order is heap, then atomics; never the reverse.
 *
 * While a context holds a capstone_lock, no domain signal handler runs in it:
 * a handler that allocated or spawned would wait for a lock its own context
 * holds. Events accepted meanwhile run when the last one is released. A spin
 * lock makes no round while held, so no handler can run under it. */
void capstone_lock(volatile int *word);
void capstone_unlock(volatile int *word);
void capstone_spin_lock(volatile int *word);
void capstone_spin_unlock(volatile int *word);

/* The runtime-internal locks the calling context holds. */
int __capstone_lock_depth(void);

#endif
