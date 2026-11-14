#include <syscall.h>
/* Compile-only probes expose each Linux number at its call site. */
long probe_read(int fd, void *p, unsigned long n)
{ return __syscall(SYS_read, fd, p, n); }
long probe_write(int fd, const void *p, unsigned long n)
{ return __syscall(SYS_write, fd, p, n); }
long probe_openat(int fd, const char *p, int flags, int mode)
{ return __syscall(SYS_openat, fd, p, flags, mode); }
long probe_close(int fd) { return __syscall(SYS_close, fd); }
void *probe_mmap(void *p, unsigned long n, int prot, int flags, int fd, long off)
{ return __capstone_linux_syscall_ref(SYS_mmap, p, __scc(n), __scc(prot), __scc(flags), __scc(fd), __scc(off)); }
long probe_munmap(void *p, unsigned long n) { return __syscall(SYS_munmap, p, n); }
long probe_mprotect(void *p, unsigned long n, int prot) { return __syscall(SYS_mprotect, p, n, prot); }
long probe_exit(int n) { return __syscall(SYS_exit, n); }
long probe_exit_group(int n) { return __syscall(SYS_exit_group, n); }
long probe_getpid(void) { return __syscall(SYS_getpid); }
long probe_clock_gettime(int clock, void *p) { return __syscall(SYS_clock_gettime, clock, p); }
long probe_rt_sigaction(int sig, const void *act, void *old)
{ return __syscall(SYS_rt_sigaction, sig, act, old, 8); }
long probe_rt_sigprocmask(int how, const void *set, void *old)
{ return __syscall(SYS_rt_sigprocmask, how, set, old, 8); }
long probe_sigaltstack(const void *ss, void *old)
{ return __syscall(SYS_sigaltstack, ss, old); }
