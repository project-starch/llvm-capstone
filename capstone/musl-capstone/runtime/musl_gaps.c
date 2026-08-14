/* Wrappers musl cannot supply, because ITS source does not compile for this
 * target -- not because a domain lacks the service.
 *
 *   src/fcntl/open.c    APInt assertion in the backend (the C-20 family)
 *   src/unistd/fsync.c  absent from libc-capstone.a for the same reason
 *
 * Both are thin wrappers over a syscall we DO implement, so writing them here
 * costs three lines each and keeps the program using ordinary POSIX. This file
 * is the place to look when a symbol is missing for that reason; if a wrapper
 * ever needs real logic it belongs in musl, not here.
 */
#include <fcntl.h>
#include <stdarg.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "syscall_arch.h"

extern long __capstone_hostcall(long n, syscall_arg_t a, syscall_arg_t b,
                                syscall_arg_t c, syscall_arg_t d,
                                syscall_arg_t e, syscall_arg_t f);

/* Returns -1 with errno set, like the libc function it replaces, so callers do
   not have to know they are talking to a hostcall. */
static long gap_ret(long r) {
  extern int *__errno_location(void);
  if (r < 0 && r > -4096) {
    *(int *)__errno_location() = (int)-r;
    return -1;
  }
  return r;
}

int open(const char *path, int flags, ...) {
  mode_t mode = 0;
  if (flags & O_CREAT) {
    va_list ap;
    va_start(ap, flags);
    mode = va_arg(ap, mode_t);
    va_end(ap);
  }
  /* AT_FDCWD is -100, and casting a NEGATIVE integer CONSTANT to a capability
     crashes the backend (ISSUES.md C-21). Routing it through a volatile makes
     it a runtime value, which lowers fine. This is also why musl's own open.c
     and fopen.c do not compile: `__scc(AT_FDCWD)` is exactly this cast, and it
     appears in every *at() wrapper. */
  volatile long at_fdcwd = AT_FDCWD;
  return (int)gap_ret(__capstone_hostcall(SYS_openat, (syscall_arg_t)at_fdcwd,
                                          (syscall_arg_t)path,
                                          (syscall_arg_t)flags,
                                          (syscall_arg_t)mode, 0, 0));
}

int fsync(int fd) {
  return (int)gap_ret(__capstone_hostcall(SYS_fsync, (syscall_arg_t)fd, 0, 0, 0, 0, 0));
}
