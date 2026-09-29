#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "delegate-service.h"
#include "capstone/spawn.h"
#include <errno.h>
#include <sys/wait.h>
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
  MAP(chdir) MAP(openat) MAP(close) MAP(pipe2) MAP(getdents64) MAP(lseek)
  MAP(read) MAP(write) MAP(readv) MAP(writev) MAP(pread64) MAP(pwrite64)
  MAP(ppoll) MAP(readlinkat) MAP(newfstatat) MAP(fstat) MAP(fsync) MAP(fdatasync)
  MAP(utimensat) MAP(renameat2) MAP(nanosleep) MAP(clock_gettime)
  MAP(clock_nanosleep) MAP(gettimeofday) MAP(times) MAP(getpid) MAP(getppid)
  MAP(getuid) MAP(geteuid) MAP(getgid) MAP(getegid) MAP(gettid) MAP(umask)
  MAP(uname) MAP(sysinfo) MAP(prlimit64) MAP(getrandom) MAP(sched_yield)
  MAP(set_tid_address) MAP(set_robust_list) MAP(futex) MAP(exit) MAP(exit_group)
  MAP(kill) MAP(wait4)
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
  unsigned count;
  long pid;
  if (capstone_spawn_unpack(block, bytes, argv, CAPSTONE_SPAWN_STRINGS + 1, envp,
                            CAPSTONE_SPAWN_STRINGS + 1, paths, CAPSTONE_SPAWN_ACTIONS, &view))
    return -EINVAL;
  if (view.flags & CAPSTONE_SPAWN_EXEC) {
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
  if (host->child_count >= CAPSTONE_DELEGATE_CHILDREN)
    return -EAGAIN;
  count = capstone_spawner_descriptors(fds, numbers, &cloexec, CAPSTONE_SPAWNER_FDS,
                                       host->spawner->socket);
  pid = capstone_spawner_spawn(host->spawner, block, bytes, fds, numbers, cloexec, count);
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

static long run(struct capstone_delegate_host *host, const struct capstone_delegate_shape *s,
                struct capstone_delegate_entry *entry) {
  long a[CAPSTONE_DELEGATE_ARGS];
  size_t bytes[CAPSTONE_DELEGATE_ARGS] = {0};
  long r;
  if (!host->bounce)
    host->bounce = malloc(host->exchange_bytes);
  for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i) {
    int flagged = (entry->flags >> i) & 1;
    if (!flagged) {
      a[i] = (long)entry->args[i];
      continue;
    }
    if (s->args[i].kind == CAPSTONE_ARG_STR) {
      if (!capstone_delegate_string_ok(host->exchange, host->exchange_bytes, entry->args[i]))
        return -EFAULT;
      a[i] = (long)(intptr_t)at(host, entry->args[i]);
      continue;
    }
    bytes[i] = capstone_delegate_arg_bytes(s, entry, i);
    if (host->bounce) {
      /* the mirror keeps the offsets, so the kernel sees ordinary pages */
      if (reads(s->args[i].kind))
        memcpy(host->bounce + entry->args[i], host->exchange + entry->args[i], bytes[i]);
      a[i] = (long)(intptr_t)(host->bounce + entry->args[i]);
    } else {
      a[i] = (long)(intptr_t)at(host, entry->args[i]);
    }
    if (reads(s->args[i].kind))
      host->bytes_in += bytes[i];
    if (writes(s->args[i].kind))
      host->bytes_out += bytes[i];
  }
  /* kill and wait4 are delegated but confined to this task and its children */
  if (entry->nr == CAPSTONE_SYS_kill && !child_of(host, (pid_t)a[0]) &&
      a[0] != getpid() && a[0] != 0)
    return -EPERM;
  if (entry->nr == CAPSTONE_SYS_wait4 && a[0] > 0 && !child_of(host, (pid_t)a[0]))
    return -ECHILD;
  {
    long number = host_number(entry->nr);
    if (number < 0)
      return -ENOSYS;
    r = syscall(number, a[0], a[1], a[2], a[3], a[4], a[5]);
    r = r == -1 ? -errno : r;
  }
  if (host->bounce)
    for (unsigned i = 0; i < CAPSTONE_DELEGATE_ARGS; ++i)
      if (bytes[i] && ((entry->flags >> i) & 1) && writes(s->args[i].kind))
        memcpy(host->exchange + entry->args[i], host->bounce + entry->args[i], bytes[i]);
  return r;
}

void capstone_delegate_serve(struct capstone_delegate_host *host,
                             struct capstone_delegate_entry *entry) {
  struct capstone_delegate_entry snapshot;
  const struct capstone_delegate_shape *s;
  int error;
  /* Snapshot first: the domain's copy is shared memory. */
  memcpy(&snapshot, entry, sizeof snapshot);
  ++host->rounds;
  error = capstone_delegate_validate(&snapshot, host->exchange_bytes);
  if (error) {
    ++host->refused;
    entry->result = -error;
    entry->pending = 0;
    return;
  }
  s = capstone_delegate_shape(snapshot.nr);
  if (snapshot.nr == CAPSTONE_NR_SPAWN) {
    ++host->syscalls;
    entry->result = spawn(host, &snapshot);
    entry->pending = 0;
    return;
  }
  if (snapshot.nr == CAPSTONE_NR_HELLO) {
    host->entry_address = snapshot.args[0];
    host->code_base = snapshot.args[1];
    host->code_end = snapshot.args[2];
    host->hello_seen = 1;
    entry->result = 0;
    entry->pending = 0;
    return;
  }
  if (snapshot.nr == CAPSTONE_SYS_exit_group || snapshot.nr == CAPSTONE_SYS_exit) {
    host->exiting = 1;
    host->exit_status = (int)snapshot.args[0] & 0xff;
    entry->result = 0;
    entry->pending = 0;
    return;
  }
  ++host->syscalls;
  host->last_nr = snapshot.nr;
  entry->result = run(host, s, &snapshot);
  entry->pending = 0;
  /* a reaped child leaves the confinement list */
  if (snapshot.nr == CAPSTONE_SYS_wait4 && entry->result > 0)
    forget_child(host, (pid_t)entry->result);
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
  CAPSTONE_SYS_getcwd
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
