/* Shared launcher/native-bridge descriptor positions and command policy.
 * Include after the platform fcntl and ioctl constants. */
#ifndef CAPSTONE_DELEGATE_POLICY_H
#define CAPSTONE_DELEGATE_POLICY_H

static unsigned fd_arguments(uint64_t nr) {
  switch (nr) {
  case CAPSTONE_SYS_dup3: case CAPSTONE_SYS_sendfile: return 3;
  case CAPSTONE_SYS_symlinkat: return 2;
  case CAPSTONE_SYS_renameat2: case CAPSTONE_SYS_linkat: case CAPSTONE_SYS_copy_file_range:
  case CAPSTONE_SYS_epoll_ctl: return 5;
  case CAPSTONE_SYS_bind: case CAPSTONE_SYS_listen: case CAPSTONE_SYS_accept:
  case CAPSTONE_SYS_accept4: case CAPSTONE_SYS_connect: case CAPSTONE_SYS_getsockname:
  case CAPSTONE_SYS_getpeername: case CAPSTONE_SYS_sendto: case CAPSTONE_SYS_recvfrom:
  case CAPSTONE_SYS_setsockopt: case CAPSTONE_SYS_getsockopt: case CAPSTONE_SYS_shutdown:
  case CAPSTONE_SYS_sendmsg: case CAPSTONE_SYS_recvmsg: case CAPSTONE_SYS_epoll_pwait:
  case CAPSTONE_SYS_dup: case CAPSTONE_SYS_fcntl: case CAPSTONE_SYS_ioctl:
  case CAPSTONE_NR_FCNTL_LOCK: case CAPSTONE_NR_IOCTL_BUF:
  case CAPSTONE_SYS_mkdirat: case CAPSTONE_SYS_unlinkat: case CAPSTONE_SYS_ftruncate:
  case CAPSTONE_SYS_faccessat: case CAPSTONE_SYS_openat: case CAPSTONE_SYS_close:
  case CAPSTONE_SYS_getdents64: case CAPSTONE_SYS_lseek:
  case CAPSTONE_SYS_read: case CAPSTONE_SYS_write: case CAPSTONE_SYS_readv:
  case CAPSTONE_SYS_writev: case CAPSTONE_SYS_preadv: case CAPSTONE_SYS_pwritev:
  case CAPSTONE_SYS_pread64: case CAPSTONE_SYS_pwrite64: case CAPSTONE_SYS_readlinkat:
  case CAPSTONE_SYS_newfstatat: case CAPSTONE_SYS_fstat: case CAPSTONE_SYS_fsync:
  case CAPSTONE_SYS_fdatasync: case CAPSTONE_SYS_sync_file_range:
  case CAPSTONE_SYS_flock: case CAPSTONE_SYS_fchmodat:
  case CAPSTONE_SYS_utimensat: case CAPSTONE_SYS_mknodat: case CAPSTONE_SYS_fstatfs:
  case CAPSTONE_SYS_statx: case CAPSTONE_SYS_fallocate: case CAPSTONE_SYS_fchdir:
  case CAPSTONE_SYS_fchmod: case CAPSTONE_SYS_fchown: case CAPSTONE_SYS_fchownat:
  case CAPSTONE_SYS_faccessat2: case CAPSTONE_SYS_readahead: case CAPSTONE_SYS_fadvise64:
  case CAPSTONE_SYS_syncfs: case CAPSTONE_SYS_timerfd_settime: case CAPSTONE_SYS_timerfd_gettime:
  case CAPSTONE_SYS_signalfd4: return 1;
  default: return 0;
  }
}

/* Only integer commands may use the integer entry. A buffer entry is not
 * permission for an arbitrary ioctl: many commands embed more pointers. */
static int command_ok(uint64_t nr, uint64_t cmd) {
  if (nr == CAPSTONE_NR_IOCTL_BUF || nr == CAPSTONE_SYS_ioctl)
    cmd &= 0xffffffffu;   /* an ioctl request is an unsigned int to the kernel */
  if (nr == CAPSTONE_NR_FCNTL_LOCK)
    return cmd == F_GETLK || cmd == F_SETLK || cmd == F_SETLKW;
  if (nr == CAPSTONE_NR_IOCTL_BUF)
    return cmd == TIOCGWINSZ || cmd == TIOCSWINSZ || cmd == TCGETS ||
           cmd == TCSETS || cmd == TCSETSW || cmd == TCSETSF ||
           cmd == FIONREAD || cmd == FIONBIO ||
           cmd == TIOCSPTLCK || cmd == TIOCGPTN || cmd == TIOCGPGRP || cmd == TIOCSPGRP;
  if (nr == CAPSTONE_SYS_ioctl)
    return cmd == FIOCLEX || cmd == FIONCLEX;
  if (nr == CAPSTONE_SYS_fcntl)
    switch (cmd) {
    case F_DUPFD: case F_DUPFD_CLOEXEC: case F_GETFD: case F_SETFD:
    case F_GETFL: case F_SETFL: case F_GETOWN: case F_SETOWN:
    case F_GETPIPE_SZ: case F_SETPIPE_SZ: case F_GET_SEALS: case F_ADD_SEALS:
    /* directory notification: the signal comes to this task like any other */
    case F_NOTIFY: case F_SETSIG: case F_GETSIG:
      return 1;
    default: return 0;
    }
  return 1;
}

#endif
