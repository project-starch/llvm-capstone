/* __lock/__unlock for a single-threaded domain.
 *
 * musl's stdio takes a lock around the open-file list (src/stdio/ofl.c) and
 * around each FILE, so the first program to call printf pulls __ofl_lock and
 * fails to link: src/thread/__lock.c does not compile for this target -- it
 * reads `libc.need_locks`, and materialising that global's address is the
 * "cannot materialize arbitrary >64-bit constants" wall.
 *
 * A domain is single-threaded: there is no clone, no futex and no scheduler
 * inside it, and the survey's whole src/thread bucket is unimplemented. A lock
 * with no second thread to exclude has nothing to do, which is also exactly what
 * musl's own fast path does when libc.need_locks is 0.
 *
 * THIS IS A REAL ASSUMPTION, not a stub of convenience: if a domain ever gains
 * threads, these become silently wrong rather than loudly missing, and every
 * FILE and the open-file list lose their mutual exclusion. The condition to
 * re-examine is the arrival of any clone/futex support in the runtime.
 */

void __lock(volatile int *l) { (void)l; }

void __unlock(volatile int *l) { (void)l; }
