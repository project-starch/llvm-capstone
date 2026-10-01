#include "capstone/delegate.h"
#include "capstone/spawn.h"
#include <errno.h>
#include <string.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Delegated syscall ABI v2 requires little-endian scalar encoding"
#endif
_Static_assert(sizeof(struct capstone_delegate_entry) == 96, "entry ABI");
_Static_assert(sizeof(struct capstone_signal_event) == 160, "signal event ABI");
_Static_assert(offsetof(struct capstone_signal_block, events) == 24, "signal handover ABI");
_Static_assert(CAPSTONE_SIGNAL_OFFSET + sizeof(struct capstone_signal_block) <= CAPSTONE_DELEGATE_META_BYTES, "signal block fits the META region");

#define I {CAPSTONE_ARG_INT, CAPSTONE_LEN_NONE, 0, 0, 0}
#define S {CAPSTONE_ARG_STR, CAPSTONE_LEN_NONE, 0, 0, 0}
#define IN_ARG(a) {CAPSTONE_ARG_IN, CAPSTONE_LEN_ARG, a, 0, 0}
#define OUT_ARG(a) {CAPSTONE_ARG_OUT, CAPSTONE_LEN_ARG, a, 0, 0}
#define IN_FIX(n) {CAPSTONE_ARG_IN, CAPSTONE_LEN_FIXED, n, 0, 0}
#define OUT_FIX(n) {CAPSTONE_ARG_OUT, CAPSTONE_LEN_FIXED, n, 0, 0}
#define OPT_IN_FIX(n) {CAPSTONE_ARG_OPT_IN, CAPSTONE_LEN_FIXED, n, 0, 0}
#define OPT_OUT_FIX(n) {CAPSTONE_ARG_OPT_OUT, CAPSTONE_LEN_FIXED, n, 0, 0}
#define OPT_INOUT_FIX(n) {CAPSTONE_ARG_OPT_INOUT, CAPSTONE_LEN_FIXED, n, 0, 0}
#define INOUT_SCALED(a, s) {CAPSTONE_ARG_INOUT, CAPSTONE_LEN_ARG_SCALED, a, s, 0}
#define IN_SCALED(a, s) {CAPSTONE_ARG_IN, CAPSTONE_LEN_ARG_SCALED, a, s, 0}
#define OUT_SCALED(a, s) {CAPSTONE_ARG_OUT, CAPSTONE_LEN_ARG_SCALED, a, s, 0}
#define OPT_IN_ARG(a) {CAPSTONE_ARG_OPT_IN, CAPSTONE_LEN_ARG, a, 0, 0}
#define INOUT_FIX(n) {CAPSTONE_ARG_INOUT, CAPSTONE_LEN_FIXED, n, 0, 0}
/* a buffer whose length is the 32-bit word at the offset in argument a */
#define INOUT_WORD(a) {CAPSTONE_ARG_INOUT, CAPSTONE_LEN_WORD, a, 0, 0}
#define OPT_INOUT_WORD(a) {CAPSTONE_ARG_OPT_INOUT, CAPSTONE_LEN_WORD, a, 0, 0}

/* Sizes are the kernel's RV64 layouts, which is what lands in the exchange
 * region after the libc's marshalling: iovec is two 64-bit words, timespec
 * two, stat 128 bytes, utsname 390, sysinfo 112, rlimit 16, tms 32, pollfd 8. */
#define IOVEC 16
#define TIMESPEC 16
#define STAT 128
/* statfs 120, statx 256; rusage 144 is what the kernel writes, musl's struct
 * is longer and the rest stays as the caller had it */
#define STATFS 120
#define STATX 256
#define RUSAGE 144
/* epoll_event is 16 bytes on riscv64 (12, packed, on x86_64: the launcher
 * converts on such a host); the msghdr block is capstone/msghdr.h */
#define EPOLL_EVENT 16
/* itimerspec is two timespecs */
#define ITIMERSPEC 32
#define MSGHDR 64

static const struct capstone_delegate_shape shapes[] = {
  /* files and directories */
  {CAPSTONE_SYS_getcwd, CAPSTONE_GROUP_DELEGATED, 2, "getcwd", {OUT_ARG(1), I}},
  {CAPSTONE_SYS_dup, CAPSTONE_GROUP_DELEGATED, 1, "dup", {I}},
  {CAPSTONE_SYS_dup3, CAPSTONE_GROUP_DELEGATED, 3, "dup3", {I, I, I}},
  /* fcntl's third argument is a struct flock for the lock commands and an
   * integer otherwise; ioctl's is a request-specific buffer. The libc decides
   * per command whether it passes an offset or 0. 64 bytes covers termios. */
  {CAPSTONE_SYS_fcntl, CAPSTONE_GROUP_DELEGATED, 3, "fcntl", {I, I, I}},
  {CAPSTONE_SYS_ioctl, CAPSTONE_GROUP_DELEGATED, 3, "ioctl", {I, I, I}},
  {CAPSTONE_SYS_flock, CAPSTONE_GROUP_DELEGATED, 2, "flock", {I, I}},
  {CAPSTONE_SYS_fchmodat, CAPSTONE_GROUP_DELEGATED, 3, "fchmodat", {I, S, I}},
  {CAPSTONE_SYS_mkdirat, CAPSTONE_GROUP_DELEGATED, 3, "mkdirat", {I, S, I}},
  {CAPSTONE_SYS_unlinkat, CAPSTONE_GROUP_DELEGATED, 3, "unlinkat", {I, S, I}},
  {CAPSTONE_SYS_symlinkat, CAPSTONE_GROUP_DELEGATED, 3, "symlinkat", {S, I, S}},
  {CAPSTONE_SYS_ftruncate, CAPSTONE_GROUP_DELEGATED, 2, "ftruncate", {I, I}},
  {CAPSTONE_SYS_faccessat, CAPSTONE_GROUP_DELEGATED, 4, "faccessat", {I, S, I, I}},
  {CAPSTONE_SYS_chdir, CAPSTONE_GROUP_DELEGATED, 1, "chdir", {S}},
  {CAPSTONE_SYS_openat, CAPSTONE_GROUP_DELEGATED, 4, "openat", {I, S, I, I}},
  {CAPSTONE_SYS_close, CAPSTONE_GROUP_DELEGATED, 1, "close", {I}},
  {CAPSTONE_SYS_pipe2, CAPSTONE_GROUP_DELEGATED, 2, "pipe2", {OUT_FIX(8), I}},
  {CAPSTONE_SYS_getdents64, CAPSTONE_GROUP_DELEGATED, 3, "getdents64", {I, OUT_ARG(2), I}},
  {CAPSTONE_SYS_lseek, CAPSTONE_GROUP_DELEGATED, 3, "lseek", {I, I, I}},
  {CAPSTONE_SYS_read, CAPSTONE_GROUP_DELEGATED, 3, "read", {I, OUT_ARG(2), I}},
  {CAPSTONE_SYS_write, CAPSTONE_GROUP_DELEGATED, 3, "write", {I, IN_ARG(2), I}},
  {CAPSTONE_SYS_readv, CAPSTONE_GROUP_DELEGATED, 3, "readv", {I, IN_SCALED(2, IOVEC), I}},
  {CAPSTONE_SYS_writev, CAPSTONE_GROUP_DELEGATED, 3, "writev", {I, IN_SCALED(2, IOVEC), I}},
  {CAPSTONE_SYS_preadv, CAPSTONE_GROUP_DELEGATED, 5, "preadv", {I, IN_SCALED(2, IOVEC), I, I, I}},
  {CAPSTONE_SYS_pwritev, CAPSTONE_GROUP_DELEGATED, 5, "pwritev", {I, IN_SCALED(2, IOVEC), I, I, I}},
  {CAPSTONE_SYS_pread64, CAPSTONE_GROUP_DELEGATED, 4, "pread64", {I, OUT_ARG(2), I, I}},
  {CAPSTONE_SYS_pwrite64, CAPSTONE_GROUP_DELEGATED, 4, "pwrite64", {I, IN_ARG(2), I, I}},
  {CAPSTONE_SYS_ppoll, CAPSTONE_GROUP_DELEGATED, 5, "ppoll",
   {INOUT_SCALED(1, 8), I, OPT_IN_FIX(TIMESPEC), OPT_IN_FIX(8), I}},
  /* three fd_sets of FD_SETSIZE bits, the timeout the kernel may update, and
     the mask itself where the kernel takes a {sigset_t *, size} pair: the
     libc flattens the pair on the way out, the launcher rebuilds it */
  {CAPSTONE_SYS_pselect6, CAPSTONE_GROUP_DELEGATED, 6, "pselect6",
   {I, OPT_INOUT_FIX(128), OPT_INOUT_FIX(128), OPT_INOUT_FIX(128), OPT_INOUT_FIX(TIMESPEC), OPT_IN_FIX(8)}},
  {CAPSTONE_SYS_readlinkat, CAPSTONE_GROUP_DELEGATED, 4, "readlinkat", {I, S, OUT_ARG(3), I}},
  {CAPSTONE_SYS_newfstatat, CAPSTONE_GROUP_DELEGATED, 4, "newfstatat", {I, S, OUT_FIX(STAT), I}},
  {CAPSTONE_SYS_fstat, CAPSTONE_GROUP_DELEGATED, 2, "fstat", {I, OUT_FIX(STAT)}},
  {CAPSTONE_SYS_fsync, CAPSTONE_GROUP_DELEGATED, 1, "fsync", {I}},
  {CAPSTONE_SYS_fdatasync, CAPSTONE_GROUP_DELEGATED, 1, "fdatasync", {I}},
  {CAPSTONE_SYS_sync_file_range, CAPSTONE_GROUP_DELEGATED, 4, "sync_file_range", {I, I, I, I}},
  {CAPSTONE_SYS_utimensat, CAPSTONE_GROUP_DELEGATED, 4, "utimensat",
   {I, {CAPSTONE_ARG_OPT_STR, CAPSTONE_LEN_NONE, 0, 0, 0}, OPT_IN_FIX(2 * TIMESPEC), I}},
  {CAPSTONE_SYS_renameat2, CAPSTONE_GROUP_DELEGATED, 5, "renameat2", {I, S, I, S, I}},
  {CAPSTONE_SYS_linkat, CAPSTONE_GROUP_DELEGATED, 5, "linkat", {I, S, I, S, I}},
  {CAPSTONE_SYS_mknodat, CAPSTONE_GROUP_DELEGATED, 4, "mknodat", {I, S, I, I}},
  {CAPSTONE_SYS_statfs, CAPSTONE_GROUP_DELEGATED, 2, "statfs", {S, OUT_FIX(STATFS)}},
  {CAPSTONE_SYS_fstatfs, CAPSTONE_GROUP_DELEGATED, 2, "fstatfs", {I, OUT_FIX(STATFS)}},
  {CAPSTONE_SYS_statx, CAPSTONE_GROUP_DELEGATED, 5, "statx", {I, S, I, I, OUT_FIX(STATX)}},
  {CAPSTONE_SYS_truncate, CAPSTONE_GROUP_DELEGATED, 2, "truncate", {S, I}},
  {CAPSTONE_SYS_fallocate, CAPSTONE_GROUP_DELEGATED, 4, "fallocate", {I, I, I, I}},
  {CAPSTONE_SYS_fchdir, CAPSTONE_GROUP_DELEGATED, 1, "fchdir", {I}},
  {CAPSTONE_SYS_fchmod, CAPSTONE_GROUP_DELEGATED, 2, "fchmod", {I, I}},
  {CAPSTONE_SYS_fchown, CAPSTONE_GROUP_DELEGATED, 3, "fchown", {I, I, I}},
  {CAPSTONE_SYS_fchownat, CAPSTONE_GROUP_DELEGATED, 5, "fchownat", {I, S, I, I, I}},
  {CAPSTONE_SYS_faccessat2, CAPSTONE_GROUP_DELEGATED, 4, "faccessat2", {I, S, I, I}},
  /* the offsets the kernel advances are 64-bit words in the buffers */
  {CAPSTONE_SYS_sendfile, CAPSTONE_GROUP_DELEGATED, 4, "sendfile", {I, I, OPT_INOUT_FIX(8), I}},
  {CAPSTONE_SYS_copy_file_range, CAPSTONE_GROUP_DELEGATED, 6, "copy_file_range",
   {I, OPT_INOUT_FIX(8), I, OPT_INOUT_FIX(8), I, I}},
  {CAPSTONE_SYS_readahead, CAPSTONE_GROUP_DELEGATED, 3, "readahead", {I, I, I}},
  {CAPSTONE_SYS_fadvise64, CAPSTONE_GROUP_DELEGATED, 4, "fadvise64", {I, I, I, I}},
  {CAPSTONE_SYS_sync, CAPSTONE_GROUP_DELEGATED, 0, "sync", {I}},
  {CAPSTONE_SYS_syncfs, CAPSTONE_GROUP_DELEGATED, 1, "syncfs", {I}},
  {CAPSTONE_SYS_memfd_create, CAPSTONE_GROUP_DELEGATED, 2, "memfd_create", {S, I}},
  /* event and timer descriptors: made here, then read, written and polled
     through the file rows like any descriptor */
  {CAPSTONE_SYS_eventfd2, CAPSTONE_GROUP_DELEGATED, 2, "eventfd2", {I, I}},
  {CAPSTONE_SYS_timerfd_create, CAPSTONE_GROUP_DELEGATED, 2, "timerfd_create", {I, I}},
  {CAPSTONE_SYS_timerfd_settime, CAPSTONE_GROUP_DELEGATED, 4, "timerfd_settime",
   {I, I, IN_FIX(ITIMERSPEC), OPT_OUT_FIX(ITIMERSPEC)}},
  {CAPSTONE_SYS_timerfd_gettime, CAPSTONE_GROUP_DELEGATED, 2, "timerfd_gettime", {I, OUT_FIX(ITIMERSPEC)}},
  /* time */
  {CAPSTONE_SYS_nanosleep, CAPSTONE_GROUP_DELEGATED, 2, "nanosleep",
   {IN_FIX(TIMESPEC), OPT_OUT_FIX(TIMESPEC)}},
  {CAPSTONE_SYS_clock_gettime, CAPSTONE_GROUP_DELEGATED, 2, "clock_gettime", {I, OUT_FIX(TIMESPEC)}},
  {CAPSTONE_SYS_clock_nanosleep, CAPSTONE_GROUP_DELEGATED, 4, "clock_nanosleep",
   {I, I, IN_FIX(TIMESPEC), OPT_OUT_FIX(TIMESPEC)}},
  {CAPSTONE_SYS_gettimeofday, CAPSTONE_GROUP_DELEGATED, 2, "gettimeofday",
   {OPT_OUT_FIX(16), OPT_OUT_FIX(8)}},
  {CAPSTONE_SYS_times, CAPSTONE_GROUP_DELEGATED, 1, "times", {OPT_OUT_FIX(32)}},
  {CAPSTONE_SYS_clock_getres, CAPSTONE_GROUP_DELEGATED, 2, "clock_getres", {I, OPT_OUT_FIX(TIMESPEC)}},
  /* identity and limits */
  {CAPSTONE_SYS_getpid, CAPSTONE_GROUP_DELEGATED, 0, "getpid", {I}},
  {CAPSTONE_SYS_getpgid, CAPSTONE_GROUP_DELEGATED, 1, "getpgid", {I}},
  {CAPSTONE_SYS_getsid, CAPSTONE_GROUP_DELEGATED, 1, "getsid", {I}},
  {CAPSTONE_SYS_getppid, CAPSTONE_GROUP_DELEGATED, 0, "getppid", {I}},
  {CAPSTONE_SYS_getuid, CAPSTONE_GROUP_DELEGATED, 0, "getuid", {I}},
  {CAPSTONE_SYS_geteuid, CAPSTONE_GROUP_DELEGATED, 0, "geteuid", {I}},
  {CAPSTONE_SYS_getgid, CAPSTONE_GROUP_DELEGATED, 0, "getgid", {I}},
  {CAPSTONE_SYS_getegid, CAPSTONE_GROUP_DELEGATED, 0, "getegid", {I}},
  {CAPSTONE_SYS_gettid, CAPSTONE_GROUP_DELEGATED, 0, "gettid", {I}},
  {CAPSTONE_SYS_umask, CAPSTONE_GROUP_DELEGATED, 1, "umask", {I}},
  {CAPSTONE_SYS_uname, CAPSTONE_GROUP_DELEGATED, 1, "uname", {OUT_FIX(390)}},
  {CAPSTONE_SYS_sysinfo, CAPSTONE_GROUP_DELEGATED, 1, "sysinfo", {OUT_FIX(112)}},
  {CAPSTONE_SYS_prlimit64, CAPSTONE_GROUP_DELEGATED, 4, "prlimit64",
   {I, I, OPT_IN_FIX(16), OPT_OUT_FIX(16)}},
  {CAPSTONE_SYS_getrandom, CAPSTONE_GROUP_DELEGATED, 3, "getrandom", {OUT_ARG(1), I, I}},
  {CAPSTONE_SYS_getresuid, CAPSTONE_GROUP_DELEGATED, 3, "getresuid", {OUT_FIX(4), OUT_FIX(4), OUT_FIX(4)}},
  {CAPSTONE_SYS_getresgid, CAPSTONE_GROUP_DELEGATED, 3, "getresgid", {OUT_FIX(4), OUT_FIX(4), OUT_FIX(4)}},
  {CAPSTONE_SYS_getgroups, CAPSTONE_GROUP_DELEGATED, 2, "getgroups", {I, OUT_SCALED(0, 4)}},
  {CAPSTONE_SYS_getrusage, CAPSTONE_GROUP_DELEGATED, 2, "getrusage", {I, OUT_FIX(RUSAGE)}},
  {CAPSTONE_SYS_getpriority, CAPSTONE_GROUP_DELEGATED, 2, "getpriority", {I, I}},
  {CAPSTONE_SYS_setpriority, CAPSTONE_GROUP_DELEGATED, 3, "setpriority", {I, I, I}},
  {CAPSTONE_SYS_getcpu, CAPSTONE_GROUP_DELEGATED, 3, "getcpu", {OPT_OUT_FIX(4), OPT_OUT_FIX(4), I}},
  /* scheduling: the task's own thread, the launcher checks the pid */
  {CAPSTONE_SYS_sched_getaffinity, CAPSTONE_GROUP_DELEGATED, 3, "sched_getaffinity", {I, I, OUT_ARG(1)}},
  {CAPSTONE_SYS_sched_setaffinity, CAPSTONE_GROUP_DELEGATED, 3, "sched_setaffinity", {I, I, IN_ARG(1)}},
  {CAPSTONE_SYS_sched_get_priority_max, CAPSTONE_GROUP_DELEGATED, 1, "sched_get_priority_max", {I}},
  {CAPSTONE_SYS_sched_get_priority_min, CAPSTONE_GROUP_DELEGATED, 1, "sched_get_priority_min", {I}},
  {CAPSTONE_SYS_sched_rr_get_interval, CAPSTONE_GROUP_DELEGATED, 2, "sched_rr_get_interval",
   {I, OUT_FIX(TIMESPEC)}},
  {CAPSTONE_SYS_sched_yield, CAPSTONE_GROUP_DELEGATED, 0, "sched_yield", {I}},
  {CAPSTONE_SYS_set_tid_address, CAPSTONE_GROUP_DELEGATED, 1, "set_tid_address", {I}},
  {CAPSTONE_SYS_set_robust_list, CAPSTONE_GROUP_DELEGATED, 2, "set_robust_list", {I, I}},
  {CAPSTONE_SYS_futex, CAPSTONE_GROUP_DELEGATED, 6, "futex", {I, I, I, I, I, I}},
  /* process: delegated members of the task model */
  {CAPSTONE_SYS_exit, CAPSTONE_GROUP_DELEGATED, 1, "exit", {I}},
  {CAPSTONE_SYS_exit_group, CAPSTONE_GROUP_DELEGATED, 1, "exit_group", {I}},
  {CAPSTONE_SYS_kill, CAPSTONE_GROUP_DELEGATED, 2, "kill", {I, I}},
  {CAPSTONE_SYS_tkill, CAPSTONE_GROUP_DELEGATED, 2, "tkill", {I, I}},
  {CAPSTONE_SYS_setpgid, CAPSTONE_GROUP_DELEGATED, 2, "setpgid", {I, I}},
  {CAPSTONE_SYS_setsid, CAPSTONE_GROUP_DELEGATED, 0, "setsid", {I}},
  /* signals: Linux keeps mask and pending set; the launcher applies its physical mask */
  {CAPSTONE_SYS_rt_sigprocmask, CAPSTONE_GROUP_DELEGATED, 4, "rt_sigprocmask",
   {I, OPT_IN_FIX(8), OPT_OUT_FIX(8), I}},
  {CAPSTONE_SYS_rt_sigsuspend, CAPSTONE_GROUP_DELEGATED, 2, "rt_sigsuspend", {IN_FIX(8), I}},
  {CAPSTONE_SYS_rt_sigpending, CAPSTONE_GROUP_DELEGATED, 2, "rt_sigpending", {OUT_FIX(8), I}},
  {CAPSTONE_SYS_rt_sigtimedwait, CAPSTONE_GROUP_DELEGATED, 4, "rt_sigtimedwait",
   {IN_FIX(8), OPT_OUT_FIX(128), OPT_IN_FIX(TIMESPEC), I}},
  {CAPSTONE_SYS_getitimer, CAPSTONE_GROUP_DELEGATED, 2, "getitimer", {I, OUT_FIX(32)}},
  {CAPSTONE_SYS_setitimer, CAPSTONE_GROUP_DELEGATED, 3, "setitimer", {I, OPT_IN_FIX(32), OPT_OUT_FIX(32)}},
  /* a descriptor that reads the pending signals in its mask; what is pending
     is Linux's, as for rt_sigtimedwait: the signals the domain blocks, which
     the launcher's physical mask blocks too. The mask's size must be 8. */
  {CAPSTONE_SYS_signalfd4, CAPSTONE_GROUP_DELEGATED, 4, "signalfd4", {I, IN_FIX(8), I, I}},
  {CAPSTONE_SYS_wait4, CAPSTONE_GROUP_DELEGATED, 4, "wait4",
   {I, OPT_OUT_FIX(4), I, OPT_OUT_FIX(144)}},
  /* sockets: descriptors like files. Linux decides family, protocol, port
     and option; the launcher checks offsets, lengths and its private
     descriptors, SCM_RIGHTS included. Address and option buffers whose
     length the caller passes behind a pointer take the word rule, and are
     INOUT so the bytes the kernel does not write stay the caller's. */
  {CAPSTONE_SYS_socket, CAPSTONE_GROUP_DELEGATED, 3, "socket", {I, I, I}},
  {CAPSTONE_SYS_socketpair, CAPSTONE_GROUP_DELEGATED, 4, "socketpair", {I, I, I, OUT_FIX(8)}},
  {CAPSTONE_SYS_bind, CAPSTONE_GROUP_DELEGATED, 3, "bind", {I, IN_ARG(2), I}},
  {CAPSTONE_SYS_listen, CAPSTONE_GROUP_DELEGATED, 2, "listen", {I, I}},
  {CAPSTONE_SYS_accept, CAPSTONE_GROUP_DELEGATED, 3, "accept", {I, OPT_INOUT_WORD(2), OPT_INOUT_FIX(4)}},
  {CAPSTONE_SYS_accept4, CAPSTONE_GROUP_DELEGATED, 4, "accept4", {I, OPT_INOUT_WORD(2), OPT_INOUT_FIX(4), I}},
  {CAPSTONE_SYS_connect, CAPSTONE_GROUP_DELEGATED, 3, "connect", {I, IN_ARG(2), I}},
  {CAPSTONE_SYS_getsockname, CAPSTONE_GROUP_DELEGATED, 3, "getsockname", {I, INOUT_WORD(2), INOUT_FIX(4)}},
  {CAPSTONE_SYS_getpeername, CAPSTONE_GROUP_DELEGATED, 3, "getpeername", {I, INOUT_WORD(2), INOUT_FIX(4)}},
  {CAPSTONE_SYS_sendto, CAPSTONE_GROUP_DELEGATED, 6, "sendto", {I, IN_ARG(2), I, I, OPT_IN_ARG(5), I}},
  {CAPSTONE_SYS_recvfrom, CAPSTONE_GROUP_DELEGATED, 6, "recvfrom",
   {I, OUT_ARG(2), I, I, OPT_INOUT_WORD(5), OPT_INOUT_FIX(4)}},
  {CAPSTONE_SYS_setsockopt, CAPSTONE_GROUP_DELEGATED, 5, "setsockopt", {I, I, I, IN_ARG(4), I}},
  {CAPSTONE_SYS_getsockopt, CAPSTONE_GROUP_DELEGATED, 5, "getsockopt",
   {I, I, I, OPT_INOUT_WORD(4), OPT_INOUT_FIX(4)}},
  {CAPSTONE_SYS_shutdown, CAPSTONE_GROUP_DELEGATED, 2, "shutdown", {I, I}},
  /* msghdr holds pointers: the libc flattens it into the block, the launcher
     rebuilds it; recvmsg writes the lengths and flags back into the block */
  {CAPSTONE_SYS_sendmsg, CAPSTONE_GROUP_DELEGATED, 3, "sendmsg", {I, IN_FIX(MSGHDR), I}},
  {CAPSTONE_SYS_recvmsg, CAPSTONE_GROUP_DELEGATED, 3, "recvmsg", {I, INOUT_FIX(MSGHDR), I}},
  {CAPSTONE_SYS_epoll_create1, CAPSTONE_GROUP_DELEGATED, 1, "epoll_create1", {I}},
  {CAPSTONE_SYS_epoll_ctl, CAPSTONE_GROUP_DELEGATED, 4, "epoll_ctl", {I, I, I, OPT_IN_FIX(EPOLL_EVENT)}},
  /* the events the kernel counted come back; with a mask it is a wait under
     a temporary mask like ppoll, and the set size must be 8 */
  {CAPSTONE_SYS_epoll_pwait, CAPSTONE_GROUP_DELEGATED, 6, "epoll_pwait",
   {I, OUT_SCALED(2, EPOLL_EVENT), I, I, OPT_IN_FIX(8), I}},
  /* the exception groups */
  {CAPSTONE_SYS_brk, CAPSTONE_GROUP_MEMORY, 1, "brk", {I}},
  {CAPSTONE_SYS_munmap, CAPSTONE_GROUP_MEMORY, 2, "munmap", {I, I}},
  {CAPSTONE_SYS_mremap, CAPSTONE_GROUP_MEMORY, 5, "mremap", {I, I, I, I, I}},
  {CAPSTONE_SYS_mmap, CAPSTONE_GROUP_MEMORY, 6, "mmap", {I, I, I, I, I, I}},
  {CAPSTONE_SYS_mprotect, CAPSTONE_GROUP_MEMORY, 3, "mprotect", {I, I, I}},
  {CAPSTONE_SYS_madvise, CAPSTONE_GROUP_MEMORY, 3, "madvise", {I, I, I}},
  {CAPSTONE_SYS_clone, CAPSTONE_GROUP_PROCESS, 5, "clone", {I, I, I, I, I}},
  {CAPSTONE_SYS_execve, CAPSTONE_GROUP_PROCESS, 3, "execve", {S, I, I}},
  {CAPSTONE_SYS_vfork, CAPSTONE_GROUP_PROCESS, 0, "vfork", {I}},
  {CAPSTONE_SYS_fork, CAPSTONE_GROUP_PROCESS, 0, "fork", {I}},
  {CAPSTONE_SYS_rt_sigaction, CAPSTONE_GROUP_SIGNAL, 4, "rt_sigaction", {I, I, I, I}},
  {CAPSTONE_SYS_rt_sigreturn, CAPSTONE_GROUP_SIGNAL, 0, "rt_sigreturn", {I}},
  /* runtime-internal, numbered by the low 16 bits of their CAPSTONE_NR_*: the
     pointer forms of fcntl and ioctl, spawn with its block in the exchange
     region, the signal requests, hello and the context requests */
  {1, CAPSTONE_GROUP_RUNTIME, 3, "hello", {I, I, I}},
  {2, CAPSTONE_GROUP_RUNTIME, 2, "spawn", {IN_ARG(1), I}},
  {3, CAPSTONE_GROUP_RUNTIME, 3, "fcntl-lock", {I, I, {CAPSTONE_ARG_INOUT, CAPSTONE_LEN_FIXED, 32, 0, 0}}},
  {4, CAPSTONE_GROUP_RUNTIME, 3, "ioctl-buffer", {I, I, {CAPSTONE_ARG_INOUT, CAPSTONE_LEN_FIXED, 64, 0, 0}}},
  {5, CAPSTONE_GROUP_RUNTIME, 3, "sigaction", {I, I, I}},
  {6, CAPSTONE_GROUP_RUNTIME, 1, "sigdone", {I}},
  {7, CAPSTONE_GROUP_RUNTIME, 0, "sigpoll", {I}},
  {8, CAPSTONE_GROUP_RUNTIME, 3, "context-create", {I, I, I}},
  {9, CAPSTONE_GROUP_RUNTIME, 3, "context-step", {I, I, {CAPSTONE_ARG_OPT_OUT, CAPSTONE_LEN_FIXED, 48, 0, 0}}},
  {10, CAPSTONE_GROUP_RUNTIME, 1, "context-forget", {I}},
  {11, CAPSTONE_GROUP_RUNTIME, 0, "context-reserve", {I}},
};

const struct capstone_delegate_shape *capstone_delegate_shape(uint64_t nr) {
  /* Runtime requests live above every Linux number, at 0xC0DE0000 + n. */
  int runtime = (nr >> 16) == 0xC0DE;
  uint64_t key = runtime ? (nr & 0xffff) : nr;
  for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; ++i)
    if (shapes[i].nr == key && (shapes[i].group == CAPSTONE_GROUP_RUNTIME) == runtime)
      return &shapes[i];
  return NULL;
}

enum capstone_delegate_group capstone_delegate_group_of(uint64_t nr) {
  const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
  return s ? (enum capstone_delegate_group)s->group : CAPSTONE_GROUP_UNKNOWN;
}

static int is_offset(const struct capstone_delegate_arg *a) {
  return a->kind != CAPSTONE_ARG_INT;
}

static int is_optional(const struct capstone_delegate_arg *a) {
  return a->kind == CAPSTONE_ARG_OPT_IN || a->kind == CAPSTONE_ARG_OPT_OUT ||
         a->kind == CAPSTONE_ARG_OPT_INOUT || a->kind == CAPSTONE_ARG_OPT_STR;
}

size_t capstone_delegate_arg_bytes(const struct capstone_delegate_shape *shape,
                                   const struct capstone_delegate_entry *entry,
                                   const void *exchange, size_t exchange_bytes,
                                   unsigned index) {
  const struct capstone_delegate_arg *a;
  if (!shape || index >= shape->argc)
    return 0;
  a = &shape->args[index];
  switch (a->length) {
  case CAPSTONE_LEN_WORD: {
    uint32_t word;
    uint64_t at;
    if (a->size >= CAPSTONE_DELEGATE_ARGS || !exchange)
      return 0;
    at = entry->args[a->size];
    if (at == 0 || at > exchange_bytes || exchange_bytes - at < sizeof word)
      return 0;
    memcpy(&word, (const char *)exchange + at, sizeof word);
    return word;
  }
  case CAPSTONE_LEN_FIXED:
    return a->size;
  case CAPSTONE_LEN_ARG:
    return a->size < CAPSTONE_DELEGATE_ARGS ? (size_t)entry->args[a->size] : 0;
  case CAPSTONE_LEN_ARG_SCALED:
    if (a->size >= CAPSTONE_DELEGATE_ARGS)
      return 0;
    if (entry->args[a->size] > SIZE_MAX / (a->scale ? a->scale : 1))
      return SIZE_MAX;
    return (size_t)entry->args[a->size] * a->scale;
  default:
    return 0;
  }
}

int capstone_delegate_pack(struct capstone_delegate_entry *entry, uint64_t nr,
                           const uint64_t args[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
  if (!entry || !args || !s ||
      (s->group != CAPSTONE_GROUP_DELEGATED && s->group != CAPSTONE_GROUP_RUNTIME))
    return EINVAL;
  memset(entry, 0, sizeof *entry);
  entry->version = CAPSTONE_DELEGATE_VERSION;
  entry->count = 1;
  entry->nr = nr;
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i) {
    entry->args[i] = args[i];
    if (i < s->argc && is_offset(&s->args[i]) &&
        !(is_optional(&s->args[i]) && args[i] == 0))
      entry->flags |= UINT64_C(1) << i;
  }
  return 0;
}

int capstone_delegate_validate(const struct capstone_delegate_entry *entry,
                               const void *exchange, size_t exchange_bytes,
                               size_t lengths[CAPSTONE_DELEGATE_ARGS]) {
  const struct capstone_delegate_shape *s;
  if (!entry || entry->version != CAPSTONE_DELEGATE_VERSION || entry->count != 1 ||
      (entry->flags >> CAPSTONE_DELEGATE_ARGS))
    return EINVAL;
  s = capstone_delegate_shape(entry->nr);
  if (!s || (s->group != CAPSTONE_GROUP_DELEGATED && s->group != CAPSTONE_GROUP_RUNTIME))
    return ENOSYS;
  /* Two passes: a word length is read only after the word's own four bytes
     have been bounded in the first. */
  for (unsigned pass = 0; pass < 2; ++pass)
    for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i) {
      const struct capstone_delegate_arg *a = i < s->argc ? &s->args[i] : NULL;
      int flagged = (entry->flags >> i) & 1;
      size_t bytes;
      if (lengths && pass == 0)
        lengths[i] = 0;
      if (!a || !is_offset(a)) {
        if (flagged)
          return EINVAL;
        continue;
      }
      if ((a->length == CAPSTONE_LEN_WORD) != (pass == 1))
        continue;
      if (!flagged) {
        if (is_optional(a) && entry->args[i] == 0)
          continue;
        return EINVAL;
      }
      if (entry->args[i] >= exchange_bytes)
        return EFAULT;
      if (a->length == CAPSTONE_LEN_WORD && a->size < CAPSTONE_DELEGATE_ARGS &&
          entry->args[a->size] && !((entry->flags >> a->size) & 1))
        return EINVAL;   /* a word that is not a checked buffer */
      bytes = capstone_delegate_arg_bytes(s, entry, exchange, exchange_bytes, i);
      if (bytes > exchange_bytes - entry->args[i])
        return EFAULT;
      if (lengths)
        lengths[i] = bytes;
    }
  return 0;
}

int capstone_delegate_string_ok(const char *exchange, size_t exchange_bytes,
                                uint64_t offset) {
  if (!exchange || offset >= exchange_bytes)
    return 0;
  return memchr(exchange + offset, 0, exchange_bytes - offset) != NULL;
}

size_t capstone_delegate_result_bytes(uint64_t nr, unsigned index, size_t bytes,
                                      int64_t result) {
  if (result < 0) {
    if (result == -EINTR && ((nr == CAPSTONE_SYS_nanosleep && index == 1) ||
                             (nr == CAPSTONE_SYS_clock_nanosleep && index == 3)))
      return bytes;
    return 0;
  }
  if (nr == CAPSTONE_SYS_wait4 && result == 0) return 0;
  /* sigtimedwait's siginfo is written only for the signal it returns */
  if (nr == CAPSTONE_SYS_rt_sigtimedwait && index == 1 && result <= 0) return 0;
  if (nr == CAPSTONE_SYS_read || nr == CAPSTONE_SYS_pread64 ||
      nr == CAPSTONE_SYS_getdents64 || nr == CAPSTONE_SYS_getrandom ||
      nr == CAPSTONE_SYS_getcwd || nr == CAPSTONE_SYS_readlinkat ||
      nr == CAPSTONE_SYS_sched_getaffinity || (nr == CAPSTONE_SYS_recvfrom && index == 1))
    return (uint64_t)result < bytes ? (size_t)result : bytes;
  /* getgroups writes the entries it counts, four bytes each; epoll_pwait the
     events it counts, sixteen each */
  if (nr == CAPSTONE_SYS_getgroups)
    return (uint64_t)result * 4 < bytes ? (size_t)result * 4 : bytes;
  if (nr == CAPSTONE_SYS_epoll_pwait && index == 1)
    return (uint64_t)result * 16 < bytes ? (size_t)result * 16 : bytes;
  /* Sleep's remaining-time output is defined only on interruption. */
  if ((nr == CAPSTONE_SYS_nanosleep && index == 1) ||
      (nr == CAPSTONE_SYS_clock_nanosleep && index == 3)) return 0;
  return bytes;
}
