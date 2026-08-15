/* Symbols a program REFERENCES but this domain cannot provide, answered with a
 * clean errno instead of a link error.
 *
 * WHY STUBS AND NOT NOTHING. A domain has no kernel: no descriptor multiplexing,
 * no thread scheduler. A program that merely CONTAINS a call to one of these --
 * mruby-io compiles `IO.select` whether or not any Ruby ever calls it -- must
 * still link, or an entire gem becomes unusable because of a method nobody
 * invokes. Corpus row 12 is exactly that case: its trigger is File.new,
 * initialize_copy and close, and `select` is dead weight the linker still wants.
 *
 * WHY -1/ENOSYS AND NOT 0. A stub that reports SUCCESS is the worst possible
 * shape here: the caller proceeds on a lie and fails somewhere else, and a run
 * that should have been a loud "this is not supported" becomes a quiet wrong
 * answer. Every one of these fails the way an unsupported syscall fails, which
 * callers already handle.
 *
 * NOT A PLACE TO PUT THINGS THAT COULD WORK. `select` on a single-threaded
 * domain whose only descriptors are host-serviced could be implemented over
 * HostCall v0 the day something needs it; it is stubbed because nothing does.
 * If a workload starts depending on one of these, implement it -- do not widen
 * the stub.
 */
#include <errno.h>

struct timeval;
struct fd_set_placeholder;

int select(int nfds, void *readfds, void *writefds, void *exceptfds,
           struct timeval *timeout) {
  (void)nfds;
  (void)readfds;
  (void)writefds;
  (void)exceptfds;
  (void)timeout;
  errno = ENOSYS;
  return -1;
}

/* Referenced from inside musl's own locking, hidden visibility, pulled in by
 * whatever stdio path the program takes. The domain is single-threaded (see
 * locks.c, same assumption and the same condition to re-examine), so a lock that
 * is never contended never needs to time out. */
int __pthread_rwlock_timedrdlock(void *rw, const void *at) {
  (void)rw;
  (void)at;
  return 0;
}
