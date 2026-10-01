#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "delegate-service.h"
#include "capstone/spawn.h"
#include <errno.h>
#include "capstone/msghdr.h"
#include <sys/wait.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/epoll.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>

/* A flagged argument becomes the launcher's address of that offset. The wire
 * validator has already bounded every offset and length. */
static void *at(const struct capstone_delegate_host *host, uint64_t offset) {
  return host->exchange + offset;
}

/* The wire carries RV64 numbers. On the guest they are the host's numbers;
 * the native tests run this same code on another architecture, so the table
 * below maps each delegated number to the build host's, where it exists. */
#define MAP(name) {CAPSTONE_SYS_##name, SYS_##name},
static const struct { uint16_t wire; long host; } numbers[] = {
#ifdef SYS_getcwd
  MAP(getcwd)
#endif
#ifdef SYS_dup
  MAP(dup)
#endif
#ifdef SYS_dup3
  MAP(dup3)
#endif
  MAP(fcntl) MAP(ioctl) MAP(mkdirat) MAP(unlinkat) MAP(ftruncate) MAP(faccessat)
  MAP(symlinkat) MAP(sync_file_range) MAP(flock) MAP(fchmodat)
  MAP(chdir) MAP(openat) MAP(close) MAP(pipe2) MAP(getdents64) MAP(lseek)
  MAP(read) MAP(write) MAP(readv) MAP(writev) MAP(preadv) MAP(pwritev) MAP(pread64) MAP(pwrite64)
  MAP(ppoll) MAP(readlinkat) MAP(newfstatat) MAP(fstat) MAP(fsync) MAP(fdatasync)
  MAP(utimensat) MAP(renameat2) MAP(nanosleep) MAP(clock_gettime)
  MAP(clock_nanosleep) MAP(gettimeofday) MAP(times) MAP(getpid) MAP(getppid)
  MAP(getuid) MAP(geteuid) MAP(getgid) MAP(getegid) MAP(gettid) MAP(umask)
  MAP(uname) MAP(sysinfo) MAP(prlimit64) MAP(getrandom) MAP(sched_yield)
  MAP(set_tid_address) MAP(set_robust_list) MAP(futex) MAP(exit) MAP(exit_group)
  MAP(kill) MAP(wait4) MAP(tkill) MAP(rt_sigsuspend) MAP(rt_sigpending)
  MAP(pselect6) MAP(getpgid) MAP(getsid)
  MAP(rt_sigtimedwait) MAP(getitimer) MAP(setitimer)
  MAP(linkat) MAP(mknodat) MAP(statfs) MAP(fstatfs) MAP(statx) MAP(truncate)
  MAP(fallocate) MAP(fchdir) MAP(fchmod) MAP(fchown) MAP(fchownat) MAP(faccessat2)
  MAP(sendfile) MAP(copy_file_range) MAP(readahead) MAP(fadvise64) MAP(sync) MAP(syncfs)
  MAP(memfd_create) MAP(clock_getres) MAP(getgroups) MAP(getrusage) MAP(getpriority)
  MAP(setpriority) MAP(getcpu) MAP(sched_getaffinity) MAP(sched_setaffinity)
  MAP(sched_get_priority_max) MAP(sched_get_priority_min) MAP(sched_rr_get_interval)
  MAP(setpgid) MAP(setsid)
  MAP(socket) MAP(socketpair) MAP(bind) MAP(listen) MAP(accept) MAP(accept4) MAP(connect)
  MAP(getsockname) MAP(getpeername) MAP(sendto) MAP(recvfrom) MAP(setsockopt) MAP(getsockopt)
  MAP(shutdown) MAP(sendmsg) MAP(recvmsg) MAP(epoll_create1) MAP(epoll_ctl) MAP(epoll_pwait)
  MAP(eventfd2) MAP(timerfd_create) MAP(timerfd_settime) MAP(timerfd_gettime) MAP(signalfd4)
  MAP(getresuid) MAP(getresgid)
};
#undef MAP

static long host_number(uint64_t nr) {
  if (nr == CAPSTONE_NR_FCNTL_LOCK)
    nr = CAPSTONE_SYS_fcntl;
  else if (nr == CAPSTONE_NR_IOCTL_BUF)
    nr = CAPSTONE_SYS_ioctl;
#if defined(__riscv) && __riscv_xlen == 64
  (void)numbers;
  return (long)nr;
#else
  for (size_t i = 0; i < sizeof numbers / sizeof numbers[0]; ++i)
    if (numbers[i].wire == nr)
      return numbers[i].host;
  return -1;
#endif
}

static int self_pid(pid_t pid) { return pid == 0 || pid == getpid(); }
static int own_group(pid_t pgid) { return self_pid(pgid) || pgid == getpgrp(); }

static int child_of(const struct capstone_delegate_host *host, pid_t pid) {
  for (unsigned i = 0; i < host->child_count; ++i)
    if (host->children[i] == pid)
      return 1;
  return 0;
}

static void forget_child(struct capstone_delegate_host *host, pid_t pid) {
  for (unsigned i = 0; i < host->child_count; ++i)
    if (host->children[i] == pid) {
      host->children[i] = host->children[--host->child_count];
      return;
    }
}

static int private_fd(const struct capstone_delegate_host *host, int fd) {
  if (host->spawner && fd == host->spawner->socket) return 1;
  for (unsigned i = 0; i < host->private_count; ++i)
    if (host->private_fds[i] == fd) return 1;
  return 0;
}

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

/* posix_spawn and execve, as one block in the exchange region. A spawn goes
 * to the unfiltered spawner with the launcher's inheritable descriptors; an
 * exec of a Capstone image is answered by the caller replacing itself. */
static long spawn(struct capstone_delegate_host *host, const struct capstone_delegate_entry *entry) {
  const char *block = host->exchange + entry->args[0];
  size_t bytes = (size_t)entry->args[1];
  static char *argv[CAPSTONE_SPAWN_STRINGS + 1], *envp[CAPSTONE_SPAWN_STRINGS + 1];
  static const char *paths[CAPSTONE_SPAWN_ACTIONS];
  struct capstone_spawn_view view;
  int fds[CAPSTONE_SPAWNER_FDS], numbers[CAPSTONE_SPAWNER_FDS];
  uint64_t cloexec;
  int count;
  long pid;
  if (capstone_spawn_unpack(block, bytes, argv, CAPSTONE_SPAWN_STRINGS + 1, envp,
                            CAPSTONE_SPAWN_STRINGS + 1, paths, CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  if (view.flags & CAPSTONE_SPAWN_EXEC) {
    if (view.flags != CAPSTONE_SPAWN_EXEC || view.actions || !view.argc)
      return -EINVAL;
    int target = open(view.path, O_RDONLY | O_CLOEXEC);
    if (target < 0) return -errno;
    close(target);
    if (!capstone_spawner_is_image(view.path))
      return -ENOSYS; /* a native program cannot take over a filtered task */
    if (bytes > sizeof host->exec_block)
      return -E2BIG;
    memcpy(host->exec_block, block, bytes);
    host->exec_bytes = bytes;
    host->exec_requested = 1;
    return 0;
  }
  if (!host->spawner)
    return -ENOSYS;
  for (unsigned i = 0; i < view.actions; ++i) {
    struct capstone_spawn_action action;
    memcpy(&action, &view.action[i], sizeof action);
    /* action.fd names a descriptor in the child's table, where the launcher's
       own never arrive; the descriptors an action reads from the launcher's
       table, a DUP2 source or an FCHDIR directory, are the ones checked */
    if ((action.cmd == CAPSTONE_SPAWN_DUP2 && private_fd(host, action.srcfd)) ||
        (action.cmd == CAPSTONE_SPAWN_FCHDIR && private_fd(host, action.fd)))
      return -EBADF;
  }
  if (host->child_count >= CAPSTONE_DELEGATE_CHILDREN)
    return -EAGAIN;
  count = capstone_spawner_descriptors(fds, numbers, &cloexec, CAPSTONE_SPAWNER_FDS,
                                       host->spawner->socket);
  if (count < 0) return count;
  unsigned kept = 0;
  uint64_t kept_cloexec = 0;
  for (int i = 0; i < count; ++i) {
    if (private_fd(host, numbers[i])) continue;
    fds[kept] = fds[i]; numbers[kept] = numbers[i];
    if ((cloexec >> i) & 1) kept_cloexec |= UINT64_C(1) << kept;
    ++kept;
  }
  pid = capstone_spawner_spawn(host->spawner, block, bytes, fds, numbers, kept_cloexec, kept,
                               capstone_signals_ignored(&host->signals), host->signals.logical);
  if (pid > 0)
    host->children[host->child_count++] = (pid_t)pid;
  return pid;
}

static int reads(unsigned kind) {
  return kind == CAPSTONE_ARG_IN || kind == CAPSTONE_ARG_OPT_IN ||
         kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
}

static int writes(unsigned kind) {
  return kind == CAPSTONE_ARG_OUT || kind == CAPSTONE_ARG_OPT_OUT ||
         kind == CAPSTONE_ARG_INOUT || kind == CAPSTONE_ARG_OPT_INOUT;
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

static long vector_call(struct capstone_delegate_host *host,
                        const struct capstone_delegate_entry *entry) {
  struct iovec iov[1024];
  uint64_t offsets[1024];
  size_t total = 0, count = entry->args[2];
  int writing = entry->nr == CAPSTONE_SYS_writev || entry->nr == CAPSTONE_SYS_pwritev;
  if (count > 1024)
    return -EINVAL;
  for (size_t i = 0; i < count; ++i) {
    uint64_t wire[2];
    memcpy(wire, host->exchange + entry->args[1] + 16 * i, sizeof wire);
    if (wire[0] > host->exchange_bytes || wire[1] > host->exchange_bytes - wire[0])
      return -EFAULT;
    if (wire[1] > (size_t)SSIZE_MAX - total)
      return -EINVAL;
    offsets[i] = wire[0];
    iov[i].iov_base = host->bounce + wire[0];
    iov[i].iov_len = wire[1];
    memcpy(iov[i].iov_base, host->exchange + wire[0], wire[1]);
    total += wire[1];
  }
  long argv[6] = {(long)(int)entry->args[0], (long)(intptr_t)iov, (long)count,
                  (long)entry->args[3], (long)entry->args[4], 0};
  long r = capstone_signals_call(&host->signals, host_number(entry->nr), argv, 0, 0);
  if (r < 0)
    return r;
  if (writing)
    host->bytes_in += (size_t)r;
  else {
    size_t left = (size_t)r;
    for (size_t i = 0; i < count && left; ++i) {
      size_t n = iov[i].iov_len < left ? iov[i].iov_len : left;
      memcpy(host->exchange + offsets[i], iov[i].iov_base, n);
      left -= n;
    }
    host->bytes_out += (size_t)r;
  }
  return r;
}

/* sendmsg and recvmsg: the msghdr block from the region, rebuilt over the
 * bounce buffer with launcher addresses, as vector_call rebuilds an iovec
 * array. Read once into the view; the region is never consulted again. A
 * descriptor of the launcher's own inside an SCM_RIGHTS message on the way
 * out is refused, as it is in every descriptor position. */
static long msg_call(struct capstone_delegate_host *host,
                     const struct capstone_delegate_entry *entry) {
  static struct capstone_msghdr_view view;
  static struct iovec iov[CAPSTONE_MSGHDR_IOVS];
  struct msghdr m;
  int sending = entry->nr == CAPSTONE_SYS_sendmsg;
  int error = capstone_msghdr_unpack(host->exchange, host->exchange_bytes, entry->args[1], &view);
  if (error)
    return -error;
  memset(&m, 0, sizeof m);
  for (uint64_t i = 0; i < view.block.iovlen; ++i) {
    iov[i].iov_base = host->bounce + view.offsets[i];
    iov[i].iov_len = (size_t)view.lengths[i];
    if (sending)
      memcpy(iov[i].iov_base, host->exchange + view.offsets[i], iov[i].iov_len);
  }
  m.msg_iov = iov;
  m.msg_iovlen = (size_t)view.block.iovlen;
  if (view.block.name) {
    m.msg_name = host->bounce + view.block.name;
    m.msg_namelen = (socklen_t)view.block.namelen;
    memcpy(m.msg_name, host->exchange + view.block.name, view.block.namelen);
  }
  if (view.block.control) {
    m.msg_control = host->bounce + view.block.control;
    m.msg_controllen = (size_t)view.block.controllen;
    memcpy(m.msg_control, host->exchange + view.block.control, view.block.controllen);
    if (sending)
      for (struct cmsghdr *c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
        if (c->cmsg_level != SOL_SOCKET || c->cmsg_type != SCM_RIGHTS)
          continue;
        size_t avail = (size_t)((char *)m.msg_control + m.msg_controllen - (char *)CMSG_DATA(c));
        size_t bytes = c->cmsg_len >= CMSG_LEN(0) ? c->cmsg_len - CMSG_LEN(0) : 0;
        if (bytes > avail) bytes = avail;
        for (size_t j = 0; j + sizeof(int) <= bytes; j += sizeof(int)) {
          int fd;
          memcpy(&fd, CMSG_DATA(c) + j, sizeof fd);
          if (private_fd(host, fd)) return -EBADF;
        }
      }
  }
  long argv[6] = {(long)(int)entry->args[0], (long)(intptr_t)&m, (long)entry->args[2], 0, 0, 0};
  long r = capstone_signals_call(&host->signals, host_number(entry->nr), argv, 0, 0);
  if (r < 0 || r == CAPSTONE_STUB_RETRY)
    return r;
  if (sending) {
    host->bytes_in += (size_t)r;
    return r;
  }
  {
    size_t left = (size_t)r;
    for (uint64_t i = 0; i < view.block.iovlen && left; ++i) {
      size_t n = iov[i].iov_len < left ? iov[i].iov_len : left;
      memcpy(host->exchange + view.offsets[i], iov[i].iov_base, n);
      left -= n;
    }
    host->bytes_out += (size_t)r;
  }
  if (view.block.name) {
    size_t n = m.msg_namelen < view.block.namelen ? m.msg_namelen : (size_t)view.block.namelen;
    memcpy(host->exchange + view.block.name, m.msg_name, n);
  }
  if (view.block.control) {
    size_t n = m.msg_controllen < view.block.controllen ? m.msg_controllen : (size_t)view.block.controllen;
    memcpy(host->exchange + view.block.control, m.msg_control, n);
  }
  /* the lengths the kernel reports and its flags go back in the block */
  {
    struct capstone_msghdr_block out = view.block;
    out.namelen = m.msg_namelen;
    out.controllen = m.msg_controllen;
    out.flags = (uint32_t)m.msg_flags;
    memcpy(host->exchange + entry->args[1], &out, sizeof out);
  }
  return r;
}

/* epoll_event is 16 bytes on riscv64 and 12, packed, on x86_64: a native test
 * host converts, as stat_call does for stat. */
static long epoll_call(struct capstone_delegate_host *host,
                       const struct capstone_delegate_entry *entry, const long a[6],
                       int has_wait, uint64_t wait_mask) {
#if defined(__riscv) && __riscv_xlen == 64
  return capstone_signals_call(&host->signals, host_number(entry->nr), a, has_wait, wait_mask);
#else
  if (entry->nr == CAPSTONE_SYS_epoll_ctl) {
    struct epoll_event ev;
    long argv[6] = {a[0], a[1], a[2], 0, 0, 0};
    if (a[3]) {
      uint64_t wire[2];
      memcpy(wire, (const void *)a[3], sizeof wire);
      ev.events = (uint32_t)wire[0];
      ev.data.u64 = wire[1];
      argv[3] = (long)(intptr_t)&ev;
    }
    return capstone_signals_call(&host->signals, host_number(entry->nr), argv, 0, 0);
  }
  {
    size_t max = (size_t)entry->args[2];
    struct epoll_event *events = NULL;
    long argv[6] = {a[0], a[1], a[2], a[3], a[4], a[5]};
    long r;
    if (max && max <= 65536) {
      events = calloc(max, sizeof *events);
      if (!events) return -ENOMEM;
      argv[1] = (long)(intptr_t)events;
    }
    r = capstone_signals_call(&host->signals, host_number(entry->nr), argv, has_wait, wait_mask);
    for (long i = 0; events && i < r && (size_t)i < max; ++i) {
      uint64_t wire[2] = {events[i].events, events[i].data.u64};
      memcpy((char *)a[1] + 16 * i, wire, sizeof wire);
    }
    free(events);
    return r;
  }
#endif
}

/* Keep the helper and unrelated native children out of waitpid(-1). Polling
 * recorded children also avoids consuming anyone else's status. A blocking
 * wait is interruptible; synchronous signal delivery is a separate milestone. */
static long wait_child(struct capstone_delegate_host *host, pid_t wanted,
                       int *out, int options, void *usage) {
  int status;
  if (options & ~(WNOHANG | WUNTRACED | WCONTINUED))
    return -EINVAL;
  for (;;) {
    unsigned matching = 0;
    for (unsigned i = 0; i < host->child_count; ++i) {
      pid_t pid = host->children[i];
      if (wanted > 0 && wanted != pid)
        continue;
      if (wanted == 0 || wanted < -1) {
        pid_t group = wanted == 0 ? getpgrp() : (pid_t)-(int64_t)wanted;
        if (getpgid(pid) != group)
          continue;
      }
      ++matching;
      long r = syscall(SYS_wait4, pid, &status,
                       options | WNOHANG, usage);
      if (r < 0)
        return -errno;
      if (r > 0) {
        if (out) memcpy(out, &status, sizeof status);
        if (WIFEXITED(status) || WIFSIGNALED(status))
          forget_child(host, pid);
        return r;
      }
    }
    if (!matching)
      return -ECHILD;
    if (options & WNOHANG)
      return 0;
    /* A signal accepted while waiting: Linux would restart wait4 under
       SA_RESTART and fail it with EINTR otherwise. The handler runs in the
       domain first either way, so RETRY is the restart. */
    if (capstone_signals_waiting(&host->signals))
      return capstone_signals_restartable(&host->signals) ? CAPSTONE_STUB_RETRY : -EINTR;
    struct timespec delay = {0, 1000000};
    if (nanosleep(&delay, NULL) && errno != EINTR)
      return -errno;
  }
}

/* Native tests run on hosts whose struct stat is not the RV64 wire layout. */
static long stat_call(struct capstone_delegate_host *host, uint64_t nr, const long a[6]) {
  (void)host;
#if defined(__riscv) && __riscv_xlen == 64
  /* The guest already has the wire layout. Keep its raw syscall path; libc
     may redirect stat through a different syscall and ABI. */
  return capstone_signals_call(&host->signals, host_number(nr), a, 0, 0);
#else
  struct stat st;
  struct rv_stat {
    uint64_t dev, ino;
    uint32_t mode, nlink, uid, gid;
    uint64_t rdev, pad1;
    int64_t size;
    int32_t blksize, pad2;
    int64_t blocks, atime_sec;
    uint64_t atime_nsec;
    int64_t mtime_sec;
    uint64_t mtime_nsec;
    int64_t ctime_sec;
    uint64_t ctime_nsec;
    uint32_t unused[2];
  } wire = {0};
  _Static_assert(sizeof wire == 128, "RV64 stat");
  int r = nr == CAPSTONE_SYS_fstat ? fstat(a[0], &st)
      : fstatat(a[0], (const char *)a[1], &st, a[3]);
  if (r < 0) return -errno;
  wire.dev = st.st_dev; wire.ino = st.st_ino; wire.mode = st.st_mode;
  wire.nlink = st.st_nlink; wire.uid = st.st_uid; wire.gid = st.st_gid;
  wire.rdev = st.st_rdev; wire.size = st.st_size; wire.blksize = st.st_blksize;
  wire.blocks = st.st_blocks;
  wire.atime_sec = st.st_atim.tv_sec; wire.atime_nsec = st.st_atim.tv_nsec;
  wire.mtime_sec = st.st_mtim.tv_sec; wire.mtime_nsec = st.st_mtim.tv_nsec;
  wire.ctime_sec = st.st_ctim.tv_sec; wire.ctime_nsec = st.st_ctim.tv_nsec;
  memcpy((void *)a[nr == CAPSTONE_SYS_fstat ? 1 : 2], &wire, sizeof wire);
  return 0;
#endif
}

static long run(struct capstone_delegate_host *host, const struct capstone_delegate_shape *s,
                struct capstone_delegate_entry *entry,
                const size_t lengths[CAPSTONE_DELEGATE_ARGS]) {
  long a[CAPSTONE_DELEGATE_ARGS];
  size_t bytes[CAPSTONE_DELEGATE_ARGS] = {0};
  long r;
  unsigned fd_mask = fd_arguments(entry->nr);
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
    if ((fd_mask & (1u << i)) && private_fd(host, (int)entry->args[i]))
      return -EBADF;
  if (entry->nr == CAPSTONE_SYS_ppoll)
    for (uint64_t i = 0; i < entry->args[1]; ++i) {
      int32_t fd;
      memcpy(&fd, host->exchange + entry->args[0] + 8 * i, sizeof fd);
      if (private_fd(host, fd)) return -EBADF;
    }
  if (!command_ok(entry->nr, entry->args[1]))
    return -ENOSYS;
  /* These calls retain pointers past the round or need shared thread memory.
     Domain addresses must never become launcher addresses. */
  if (entry->nr == CAPSTONE_SYS_set_tid_address ||
      entry->nr == CAPSTONE_SYS_set_robust_list || entry->nr == CAPSTONE_SYS_futex)
    return -ENOSYS;
  if ((entry->nr == CAPSTONE_SYS_ppoll && entry->args[3] && entry->args[4] != 8) ||
      (entry->nr == CAPSTONE_SYS_epoll_pwait && entry->args[4] && entry->args[5] != 8) ||
      (entry->nr == CAPSTONE_SYS_signalfd4 && entry->args[2] != 8))
    return -EINVAL;
  if (entry->nr == CAPSTONE_SYS_prlimit64 && !self_pid((pid_t)entry->args[0]))
    return -EPERM;
  /* scheduling and priority: this task only; a process group is formed by
     the task or a child, or a child joins the task's group or a child's */
  if ((entry->nr == CAPSTONE_SYS_sched_getaffinity || entry->nr == CAPSTONE_SYS_sched_setaffinity ||
       entry->nr == CAPSTONE_SYS_sched_rr_get_interval) && !self_pid((pid_t)entry->args[0]))
    return -EPERM;
  if ((entry->nr == CAPSTONE_SYS_getpriority || entry->nr == CAPSTONE_SYS_setpriority) &&
      (entry->args[0] != PRIO_PROCESS || !self_pid((pid_t)entry->args[1])))
    return -EPERM;
  if (entry->nr == CAPSTONE_SYS_setpgid &&
      (!(self_pid((pid_t)entry->args[0]) || child_of(host, (pid_t)entry->args[0])) ||
       !(own_group((pid_t)entry->args[1]) || child_of(host, (pid_t)entry->args[1]) ||
         entry->args[1] == entry->args[0])))
    return -EPERM;
  if (!host->bounce)
    host->bounce = malloc(host->exchange_bytes);
  if (!host->bounce)
    return -ENOMEM;
  if (entry->nr == CAPSTONE_SYS_readv || entry->nr == CAPSTONE_SYS_writev ||
      entry->nr == CAPSTONE_SYS_preadv || entry->nr == CAPSTONE_SYS_pwritev)
    return vector_call(host, entry);
  if (entry->nr == CAPSTONE_SYS_sendmsg || entry->nr == CAPSTONE_SYS_recvmsg)
    return msg_call(host, entry);
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i) {
    int flagged = (entry->flags >> i) & 1;
    if (!flagged) {
      a[i] = (long)entry->args[i];
      continue;
    }
    if (s->args[i].kind == CAPSTONE_ARG_STR || s->args[i].kind == CAPSTONE_ARG_OPT_STR) {
      if (!capstone_delegate_string_ok(host->exchange, host->exchange_bytes, entry->args[i]))
        return -EFAULT;
      a[i] = (long)(intptr_t)at(host, entry->args[i]);
      continue;
    }
    bytes[i] = lengths[i];
    if (host->bounce) {
      /* Ordinary pages keep wire offsets. Initialize output padding so
         kernel structs cannot expose malloc contents. */
      if (reads(s->args[i].kind))
        memcpy(host->bounce + entry->args[i], host->exchange + entry->args[i], bytes[i]);
      else
        memset(host->bounce + entry->args[i], 0, bytes[i]);
      a[i] = (long)(intptr_t)(host->bounce + entry->args[i]);
    } else {
      a[i] = (long)(intptr_t)at(host, entry->args[i]);
    }
    if (reads(s->args[i].kind))
      host->bytes_in += bytes[i];
    if (writes(s->args[i].kind))
      host->bytes_out += bytes[i];
  }
  /* a length given as a word: the kernel sees the length the validator
     checked, whatever the domain wrote into the word since */
  for (unsigned i = 0; i < s->argc; ++i)
    if (s->args[i].length == CAPSTONE_LEN_WORD && ((entry->flags >> i) & 1) &&
        s->args[i].size < CAPSTONE_DELEGATE_ARGS && entry->args[s->args[i].size]) {
      uint32_t word = (uint32_t)lengths[i];
      memcpy(host->bounce + entry->args[s->args[i].size], &word, sizeof word);
    }
  /* kill and wait4 are delegated but confined to this task, its children and
     the task that spawned it: a child domain may signal its parent. Anything
     else is EPERM, or ESRCH when there is no such process, as Linux answers a
     kill of a pid that has been reaped. Signal 0 to the task's own group
     signals nothing and asks only what Linux always answers, since the task
     is in its group: that goes through. */
  if (entry->nr == CAPSTONE_SYS_kill && !child_of(host, (pid_t)a[0]) &&
      (pid_t)a[0] != getpid() && (pid_t)a[0] != getppid() && (a[0] != 0 || a[1] != 0))
    return kill((pid_t)a[0], 0) < 0 && errno == ESRCH ? -ESRCH : -EPERM;
  if (entry->nr == CAPSTONE_SYS_wait4 && a[0] > 0 && !child_of(host, (pid_t)a[0]))
    return -ECHILD;
  /* raise() is tkill on the task's own thread; nothing else is a domain's to signal */
  if (entry->nr == CAPSTONE_SYS_tkill && (pid_t)a[0] != getpid())
    return -EPERM;
  if (entry->nr == CAPSTONE_SYS_wait4)
    r = wait_child(host, (pid_t)a[0], (int *)a[1], (int)a[2], (void *)a[3]);
  else if (entry->nr == CAPSTONE_SYS_fstat || entry->nr == CAPSTONE_SYS_newfstatat)
    r = stat_call(host, entry->nr, a);
  else {
    long number = host_number(entry->nr);
    if (number < 0)
      return -ENOSYS;
    /* A wait with a temporary mask classifies the events it accepts. */
    int has_wait = 0;
    uint64_t wait_mask = 0;
    if (entry->nr == CAPSTONE_SYS_rt_sigsuspend ||
        (entry->nr == CAPSTONE_SYS_ppoll && entry->args[3]) ||
        (entry->nr == CAPSTONE_SYS_pselect6 && entry->args[5]) ||
        (entry->nr == CAPSTONE_SYS_epoll_pwait && entry->args[4])) {
      void *mask = (void *)a[entry->nr == CAPSTONE_SYS_ppoll ? 3
                              : entry->nr == CAPSTONE_SYS_pselect6 ? 5
                              : entry->nr == CAPSTONE_SYS_epoll_pwait ? 4 : 0];
      memcpy(&wait_mask, mask, sizeof wait_mask);
      /* A temporary kernel mask must not undo the runtime's backpressure.
         Keep the domain's requested mask in wait_mask for handler delivery. */
      uint64_t physical_wait = wait_mask | host->signals.backpressure;
      memcpy(mask, &physical_wait, sizeof physical_wait);
      has_wait = 1;
    }
    /* pselect6's sixth argument is a {sigset_t *, size} pair to the kernel;
       on the wire it is the mask itself, or 0 */
    struct { void *ss; size_t len; } sig6 = {(void *)a[5], 8};
    if (entry->nr == CAPSTONE_SYS_pselect6)
      a[5] = (uint64_t)(uintptr_t)&sig6;
    if (entry->nr == CAPSTONE_SYS_epoll_ctl || entry->nr == CAPSTONE_SYS_epoll_pwait)
      r = epoll_call(host, entry, a, has_wait, wait_mask);
    else
      r = capstone_signals_call(&host->signals, number, a, has_wait, wait_mask);
  }
  if (r == CAPSTONE_STUB_RETRY)
    return r;
  if (host->bounce)
    for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
      if (bytes[i] && ((entry->flags >> i) & 1) && writes(s->args[i].kind))
        memcpy(host->exchange + entry->args[i], host->bounce + entry->args[i],
               capstone_delegate_result_bytes(entry->nr, i, bytes[i], r));
  return r;
}

void capstone_delegate_serve(struct capstone_delegate_host *host,
                             struct capstone_delegate_entry *entry) {
  struct capstone_delegate_entry snapshot;
  const struct capstone_delegate_shape *s;
  size_t lengths[CAPSTONE_DELEGATE_ARGS];
  int error;
  /* Snapshot first: the domain's copy is shared memory. */
  memcpy(&snapshot, entry, sizeof snapshot);
  ++host->rounds;
  error = capstone_delegate_validate(&snapshot, host->exchange, host->exchange_bytes, lengths);
  if (error) {
    ++host->refused;
    entry->result = -error;
    capstone_signals_publish(&host->signals, entry, 0);
    return;
  }
  s = capstone_delegate_shape(snapshot.nr);
  host->last_nr = snapshot.nr;
  long r;
  if (snapshot.nr == CAPSTONE_NR_SPAWN) {
    ++host->syscalls;
    r = spawn(host, &snapshot);
  } else if (snapshot.nr == CAPSTONE_NR_HELLO) {
    host->entry_address = snapshot.args[0];
    host->code_base = snapshot.args[1];
    host->code_end = snapshot.args[2];
    host->hello_seen = 1;
    r = 0;
  } else if (snapshot.nr == CAPSTONE_SYS_exit_group || snapshot.nr == CAPSTONE_SYS_exit) {
    host->exiting = 1;
    host->exit_status = (int)snapshot.args[0] & 0xff;
    r = 0;
  } else if (snapshot.nr == CAPSTONE_NR_SIGACTION) {
    r = capstone_signals_action(&host->signals, (int)snapshot.args[0],
                                (unsigned)snapshot.args[1], (unsigned)snapshot.args[2]);
  } else if (snapshot.nr == CAPSTONE_NR_SIGDONE) {
    r = capstone_signals_done(&host->signals, snapshot.args[0]);
  } else if (snapshot.nr == CAPSTONE_NR_SIGPOLL) {
    r = 0;
  } else if (snapshot.nr == CAPSTONE_NR_CONTEXT_CREATE || snapshot.nr == CAPSTONE_NR_CONTEXT_STEP ||
             snapshot.nr == CAPSTONE_NR_CONTEXT_FORGET) {
    r = host->context ? host->context(host, &snapshot) : -ENOSYS;
  } else if (snapshot.nr == CAPSTONE_SYS_rt_sigprocmask) {
    /* the logical mask is the domain's; the kernel gets the physical one */
    uint64_t set = 0, old = 0;
    ++host->syscalls;
    if (snapshot.args[3] != 8)
      r = -EINVAL;
    else {
      if (snapshot.args[1]) memcpy(&set, host->exchange + snapshot.args[1], sizeof set);
      r = capstone_signals_procmask(&host->signals, (int)snapshot.args[0],
                                    snapshot.args[1] ? &set : NULL, &old);
      if (!r && snapshot.args[2]) memcpy(host->exchange + snapshot.args[2], &old, sizeof old);
    }
  } else {
    ++host->syscalls;
    host->last_nr = snapshot.nr;
    r = run(host, s, &snapshot, lengths);
  }
  entry->result = r == CAPSTONE_STUB_RETRY ? 0 : r;
  capstone_signals_publish(&host->signals, entry, r == CAPSTONE_STUB_RETRY);
}

void capstone_delegate_host_free(struct capstone_delegate_host *host) {
  free(host->bounce);
  host->bounce = NULL;
}

/* Every delegated number, plus the launcher's own: the device ioctls, its
 * mappings, its stderr, and the signal work of a fault exit. Anything else
 * answers ENOSYS rather than killing the process, so a missing entry is a
 * visible error and not a silent death. */
static const uint16_t launcher_own[] = {
  CAPSTONE_SYS_ioctl, CAPSTONE_SYS_mmap, CAPSTONE_SYS_munmap, CAPSTONE_SYS_close,
  CAPSTONE_SYS_exit_group, CAPSTONE_SYS_exit, CAPSTONE_SYS_rt_sigaction,
  CAPSTONE_SYS_rt_sigprocmask, CAPSTONE_SYS_rt_sigreturn, CAPSTONE_SYS_write,
  CAPSTONE_SYS_read, CAPSTONE_SYS_fcntl, CAPSTONE_SYS_getpid, CAPSTONE_SYS_gettid,
  CAPSTONE_SYS_kill, CAPSTONE_SYS_brk, CAPSTONE_SYS_mprotect, CAPSTONE_SYS_futex,
  CAPSTONE_SYS_madvise, CAPSTONE_SYS_mremap, CAPSTONE_SYS_openat, CAPSTONE_SYS_newfstatat,
  CAPSTONE_SYS_fstat, CAPSTONE_SYS_lseek, CAPSTONE_SYS_getrandom, CAPSTONE_SYS_clock_gettime,
  131 /* tgkill, which raise() uses */, 211 /* sendmsg */, 212 /* recvmsg */,
  206 /* sendto */, 207 /* recvfrom */, CAPSTONE_SYS_readlinkat, CAPSTONE_SYS_getdents64,
  /* exec in place restarts this program under the inherited filter: its own
     startup needs the image memfd, the spawner fork and socket, and the
     second filter installation */
  CAPSTONE_SYS_execve, 279 /* memfd_create */, 167 /* prctl */, 277 /* seccomp */,
  CAPSTONE_SYS_clone, 199 /* socketpair */, CAPSTONE_SYS_dup3, CAPSTONE_SYS_pipe2,
  CAPSTONE_SYS_getcwd, 154 /* setpgid */, 155 /* getpgid */
};

int capstone_delegate_seccomp(void) {
  /* Collect the allowlist: delegated shapes are enumerated by number range. */
  uint16_t allow[512];
  unsigned n = 0;
  for (unsigned nr = 0; nr < 1100 && n < 500; ++nr) {
    const struct capstone_delegate_shape *s = capstone_delegate_shape(nr);
    if (s && s->group == CAPSTONE_GROUP_DELEGATED)
      allow[n++] = (uint16_t)nr;
  }
  for (unsigned i = 0; i < sizeof launcher_own / sizeof launcher_own[0] && n < 510; ++i) {
    unsigned j;
    for (j = 0; j < n && allow[j] != launcher_own[i]; ++j)
      ;
    if (j == n)
      allow[n++] = launcher_own[i];
  }
  {
    struct sock_filter program[3 + 512 + 2];
    unsigned k = 0;
    struct sock_fprog fprog;
    program[k++] = (struct sock_filter)BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
        offsetof(struct seccomp_data, arch));
    program[k++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_RISCV64, 1, 0);
    program[k++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (ENOSYS & SECCOMP_RET_DATA));
    program[k++] = (struct sock_filter)BPF_STMT(BPF_LD | BPF_W | BPF_ABS,
        offsetof(struct seccomp_data, nr));
    for (unsigned i = 0; i < n; ++i)
      program[k++] = (struct sock_filter)BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, allow[i], n - i, 0);
    program[k++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | (ENOSYS & SECCOMP_RET_DATA));
    program[k++] = (struct sock_filter)BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW);
    fprog.len = (unsigned short)k;
    fprog.filter = program;
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0))
      return errno;
    if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &fprog))
      return errno;
  }
  return 0;
}

static void hex(char *out, uint64_t v) {
  static const char digits[] = "0123456789abcdef";
  char tmp[17];
  int n = 0;
  do { tmp[n++] = digits[v & 15]; v >>= 4; } while (v);
  while (n)
    *out++ = tmp[--n];
  *out = 0;
}

void capstone_delegate_fault_record(int fd, const struct capstone_delegate_host *host,
                                    const char *image, uint64_t cause, uint64_t pc,
                                    uint64_t address) {
  /* No stdio: this can run after a closed or full pipe, and must not block. */
  char line[512], number[24];
  size_t n = 0;
  int flags;
#define PUT(text) do { const char *t_ = (text); while (*t_ && n + 1 < sizeof line) line[n++] = *t_++; } while (0)
#define PUTHEX(v) do { hex(number, (v)); PUT("0x"); PUT(number); } while (0)
  PUT("capstone-exec: domain fault cause=");
  { /* decimal, as every cause in the issue registry is written */
    char dec[24]; int k = 0; uint64_t v = cause;
    do { dec[k++] = (char)('0' + v % 10); v /= 10; } while (v);
    while (k && n + 1 < sizeof line) line[n++] = dec[--k];
  }
  PUT(" pc="); PUTHEX(pc);
  PUT(" address="); PUTHEX(address);
  if (host && host->hello_seen) {
    PUT(" entry="); PUTHEX(host->entry_address);
    PUT(" code="); PUTHEX(host->code_base); PUT("-"); PUTHEX(host->code_end);
  } else {
    PUT(" entry=unknown");
  }
  if (host) {
    /* the last request served, and the one the domain was preparing */
    PUT(" last="); PUTHEX(host->last_nr);
    PUT(" preparing="); PUTHEX(host->preparing_nr);
    if (host->image_sha256[0]) { PUT(" sha256="); PUT(host->image_sha256); }
  }
  if (image) { PUT(" image="); PUT(image); }
  PUT("\n");
#undef PUT
#undef PUTHEX
  flags = fcntl(fd, F_GETFL);
  if (flags >= 0)
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  (void)!write(fd, line, n);
  if (flags >= 0)
    fcntl(fd, F_SETFL, flags);
}
