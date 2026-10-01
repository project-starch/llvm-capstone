#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pty.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <sys/eventfd.h>
#include <sys/resource.h>
#include <sys/signalfd.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/timerfd.h>
#include <sys/wait.h>
#include <unistd.h>
#include <capstone/capability.h>

void *__capstone_region(unsigned);
static volatile unsigned char *volatile saved_allocation;

/* A single, identifiable load site for the heap fault oracle. The runner
 * checks its linked PC as well as the architectural exception cause. */
__attribute__((noinline)) unsigned char capstone_heap_fault_load(volatile const unsigned char *p) {
  return *p;
}

static int heap_fault_ready(void) {
  return write(1, "heap: ready\n", 12) == 12;
}

static int heap_fault_survived(void) {
  if (write(1, "heap: survived\n", 15) != 15) return 68;
  return 90; /* only the unprotected control may reach this sentinel */
}

extern char **environ;
static int constructed;
__attribute__((constructor)) static void initialize(void) {
  const char *value = getenv("CAPSTONE_CONTRACT");
  constructed = value && !strcmp(value, "environment with spaces");
}

int main(int argc, char **argv) {
  if (!constructed || argc != 4 || strcmp(argv[2], "") ||
      strcmp(argv[3], "argument with spaces\nand newline"))
    return 41;
  if (write(1, "stdout\n", 7) != 7 || write(2, "stderr\n", 7) != 7)
    return 42;
  /* These two controls require the same application built with HEAP=sublet. */
  if (!strcmp(argv[1], "fault-stale")) {
    volatile unsigned char *stale = malloc(64);
    if (!stale) return 46;
    *stale = 42;
    free((void *)stale);
    if (!heap_fault_ready()) return 68;
    (void)capstone_heap_fault_load(stale);
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "fault-exhaust")) {
    /* Keep adding valid tree ancestors without revoking them. Unlike malloc/
     * free churn, these structural nodes cannot be collected. The runtime must
     * report exhaustion and preserve enough nodes to destroy this process. */
    capstone_cap_slot region, handle;
    capstone_cap_store(&region, __capstone_region(0));
    for (;;) {
      capstone_cap_make_handle(&region, &handle);
      capstone_cap_clear(&handle);
    }
  }
  if (!strcmp(argv[1], "churn") || !strcmp(argv[1], "fault-reused")) {
    saved_allocation = malloc(64);
    if (!saved_allocation) return 46;
    *saved_allocation = 42;
    free((void *)saved_allocation);
    unsigned char *held = malloc(256);
    if (!held) return 46;
    memset(held, 0x5a, 256);
    /* The held block prevents whole-pool coalescing on every iteration. Each
     * iteration retires an identity while the useful working set stays small.
     * More than three times the 65,536-node capacity must fit in one process. */
    for (unsigned i = 0; i < 200000; ++i) {
      volatile unsigned char *p = malloc(64);
      if (!p) return 46;
      p[0] = i & 255;
      p[63] = (i >> 8) & 255;
      if (p[0] != (i & 255) || p[63] != ((i >> 8) & 255)) return 47;
      free((void *)p);
      if (held[i & 255] != 0x5a) return 48;
    }
    free(held);
    if (!strcmp(argv[1], "fault-reused")) {
      if (!heap_fault_ready()) return 68;
      (void)capstone_heap_fault_load(saved_allocation);
      return heap_fault_survived();
    }
  }
  /* The heap qualification (docs/plans/capstone-heap-protection.md). Every
   * fault-* case below must end in SIGSEGV on the HEAP=sublet build and run to
   * completion on the control, HEAP=level0 built with
   * CAPSTONE_LEVEL0_OBJECT_BOUNDS=0. On the default level0 build, which bounds
   * each allocation, the spatial cases (fault-bounds, fault-bounds-large,
   * fault-realloc-shrink) end in SIGSEGV too and the others complete. The heap-*
   * cases fall through to the ordinary tail on success and return a distinct
   * code on the first failed check. */
  if (!strcmp(argv[1], "fault-bounds")) {
    /* one byte past a small allocation; the last byte inside it first */
    volatile unsigned char *p = malloc(24);
    if (!p) return 46;
    p[0] = 1; p[23] = 2;
    if (p[0] != 1 || p[23] != 2) return 61;
    if (!heap_fault_ready()) return 68;
    (void)capstone_heap_fault_load(p + 24);
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "fault-bounds-large")) {
    /* 5000 is a multiple of the 8-byte grain the heap rounds to above 4096
       bytes, so the alias ends exactly at the requested size */
    volatile unsigned char *p = malloc(5000);
    if (!p) return 46;
    p[4999] = 3;
    if (p[4999] != 3) return 61;
    if (!heap_fault_ready()) return 68;
    (void)capstone_heap_fault_load(p + 5000);
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "fault-realloc-shrink")) {
    /* a realloc that shrinks returns a pointer to the new size, whether the
       heap moves the block (sublet) or shrinks it in place and releases the
       tail (level0): the last byte inside first, then one past it */
    volatile unsigned char *p = malloc(4096);
    if (!p) return 46;
    volatile unsigned char *q = realloc((void *)p, 24);
    if (!q) return 46;
    q[0] = 1; q[23] = 2;
    if (q[0] != 1 || q[23] != 2) return 61;
    if (!heap_fault_ready()) return 68;
    (void)capstone_heap_fault_load(q + 24);
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "fault-double-free")) {
    unsigned char *p = malloc(64);
    if (!p) return 46;
    free(p);
    if (!heap_fault_ready()) return 68;
    free(p);   /* the probe read in free faults on the revoked alias */
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "fault-double-free-reused")) {
    /* free, get the same address back, then free the old alias: it must
       fault at the probe, before it could revoke the new owner's handle */
    unsigned char *p = malloc(64);
    if (!p) return 46;
    unsigned long address = __builtin_capstone_cap_get_cursor(p);
    free(p);
    unsigned char *q = malloc(64);
    if (!q) return 46;
    if (__builtin_capstone_cap_get_cursor(q) != address) return 63;
    memset(q, 0x5a, 64);
    if (!heap_fault_ready()) return 68;
    free(p);
    return heap_fault_survived();
  }
  if (!strcmp(argv[1], "heap-bounds")) {
    volatile unsigned char *p = malloc(24);
    if (!p) return 46;
    p[0] = 1; p[23] = 2;
    if (p[0] != 1 || p[23] != 2) return 61;
    free((void *)p);
    volatile unsigned char *q = malloc(5000);
    if (!q) return 46;
    q[0] = 4; q[4999] = 5;
    if (q[0] != 4 || q[4999] != 5) return 61;
    free((void *)q);
  }
  if (!strcmp(argv[1], "heap-neighbour")) {
    /* freeing one block leaves its live neighbour intact, and the freed
       address comes back to the next allocation of the same size */
    unsigned char *a = malloc(64);
    unsigned char *b = malloc(64);
    if (!a || !b) return 46;
    memset(a, 0x11, 64);
    memset(b, 0x22, 64);
    unsigned long address = __builtin_capstone_cap_get_cursor(a);
    free(a);
    for (unsigned i = 0; i < 64; ++i)
      if (b[i] != 0x22) return 64;
    unsigned char *c = malloc(64);
    if (!c) return 46;
    if (__builtin_capstone_cap_get_cursor(c) != address) return 63;
    memset(c, 0x33, 64);
    for (unsigned i = 0; i < 64; ++i)
      if (b[i] != 0x22) return 64;
    free(b);
    free(c);
  }
  if (!strcmp(argv[1], "heap-companion")) {
    free(NULL);
    unsigned char *zero = malloc(0);
    if (!zero) return 46;
    zero[0] = 9;   /* the documented one-byte policy */
    free(zero);
    unsigned char *cleared = calloc(16, 16);
    if (!cleared) return 46;
    for (unsigned i = 0; i < 256; ++i)
      if (cleared[i]) return 65;
    free(cleared);
    /* realloc keeps bytes and the capability stored inside the block */
    unsigned char *inner = malloc(8);
    unsigned char **outer = malloc(48);
    if (!inner || !outer) return 46;
    inner[0] = 7;
    outer[0] = inner;
    memset((unsigned char *)outer + sizeof(void *), 0x44, 48 - sizeof(void *));
    unsigned char **grown = realloc(outer, 4096);
    if (!grown) return 46;
    if (grown[0] != inner || grown[0][0] != 7) return 66;
    for (unsigned i = sizeof(void *); i < 48; ++i)
      if (((unsigned char *)grown)[i] != 0x44) return 66;
    /* a failed realloc leaves the original usable */
    errno = 0;
    void *huge = realloc(grown, (size_t)1 << 40);
    if (huge || errno != ENOMEM) return 67;
    if (grown[0][0] != 7 || ((unsigned char *)grown)[47] != 0x44) return 67;
    free(grown);
    free(inner);
  }
  if (!strcmp(argv[1], "heap-realloc-shrink")) {
    /* shrinking keeps the bytes and a capability stored in the block, the
       shrunk block is usable to its new end, and an allocation after it,
       which on level0 may take the released tail, leaves it intact */
    unsigned char *inner = malloc(8);
    unsigned char **p = malloc(4096);
    if (!inner || !p) return 46;
    inner[0] = 7;
    p[0] = inner;
    memset((unsigned char *)p + sizeof(void *), 0x55, 4096 - sizeof(void *));
    unsigned char **q = realloc(p, 48);
    if (!q) return 46;
    if (q[0] != inner || q[0][0] != 7) return 66;
    for (unsigned i = sizeof(void *); i < 48; ++i)
      if (((unsigned char *)q)[i] != 0x55) return 66;
    memset((unsigned char *)q + sizeof(void *), 0x66, 48 - sizeof(void *));
    unsigned char *later = malloc(2048);
    if (!later) return 46;
    memset(later, 0x77, 2048);
    if (q[0] != inner || q[0][0] != 7) return 66;
    for (unsigned i = sizeof(void *); i < 48; ++i)
      if (((unsigned char *)q)[i] != 0x66) return 66;
    free(later);
    free(q);
    free(inner);
  }
  if (!strcmp(argv[1], "write-loop")) {
    char output[8192];
    memset(output, 'x', sizeof output);
    while (write(1, output, sizeof output) > 0) {}
    return 43;
  }
  if (!strcmp(argv[1], "loop")) {
    for (;;)
      __asm__ volatile ("" ::: "memory");
  }
  if (!strcmp(argv[1], "fault-mrev")) {
    __asm__ volatile (".insn r 0x5b, 1, 8, t0, x0, x0" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-privcsr")) {
    __asm__ volatile ("csrw mie, zero" ::: "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-debugcap")) {
    __asm__ volatile (".insn r 0x5b, 1, 64, t0, x0, x0" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-capenter")) {
    __asm__ volatile (".insn r 0x5b, 1, 13, x0, x0, x0" ::: "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-vector")) {
    __asm__ volatile (".insn i 0x5b, 7, x0, x0, 0\n"
                      "li sp, 0\nli gp, 0\nld t0, 0(sp)" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault-stack")) {
    __asm__ volatile("li sp, 0\nli gp, 0\nld t0, 0(sp)" ::: "t0", "memory");
    return 43;
  }
  if (!strcmp(argv[1], "fault")) {
    volatile unsigned long *invalid = (void *)1;
    *invalid = 1;
    return 43;
  }
  if (!strcmp(argv[1], "exit139"))
    return 139;
  if (!strcmp(argv[1], "spawn")) {
    /* popen and system go through posix_spawn; the child is a native program */
    char line[64];
    FILE *child = popen("echo from a child; exit 3", "r");
    if (!child) { perror("popen"); return 50; }
    if (!fgets(line, sizeof line, child) || strcmp(line, "from a child\n")) return 51;
    int status = pclose(child);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 3) {
      fprintf(stderr, "pclose: status=%d errno=%d\n", status, errno);
      return 52;
    }
    status = system("exit 7");
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 7) return 53;
    if (system("echo stdout\n >/dev/null") != 0) return 54;
    puts("spawn: ok");
    return 0;
  }
  if (!strcmp(argv[1], "spawn-image")) {
    /* a Capstone image as the child, spawned directly: the launcher starts
       it through itself, with our pipe as its stdout and our stdin */
    char line[64];
    int p[2];
    pid_t pid;
    int status;
    posix_spawn_file_actions_t fa;
    char *next[] = {argv[0], "healthy", "", "argument with spaces\nand newline", NULL};
    if (pipe(p)) return 55;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, p[1], 1);
    posix_spawn_file_actions_addclose(&fa, p[0]);
    posix_spawn_file_actions_addclose(&fa, p[1]);
    if (posix_spawn(&pid, argv[0], &fa, NULL, next, environ)) { perror("posix_spawn"); return 55; }
    close(p[1]);
    FILE *child = fdopen(p[0], "r");
    if (!child) return 55;
    if (!fgets(line, sizeof line, child) || strcmp(line, "stdout\n")) {
      fprintf(stderr, "spawn-image: first line %s errno=%d\n", line, errno);
      return 56;
    }
    if (!fgets(line, sizeof line, child) || strcmp(line, "application: ok\n")) return 57;
    fclose(child);
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0) return 58;
    puts("spawn-image: ok");
    return 0;
  }
  if (!strcmp(argv[1], "exec")) {
    /* replace this task with the same image in healthy mode: pid and streams stay */
    char *next[] = {argv[0], "healthy", "", "argument with spaces\nand newline", NULL};
    execv(argv[0], next);
    return 59;
  }
  if (!strcmp(argv[1], "pty")) {
    /* a pseudo-terminal pair through musl's openpty: /dev/ptmx, unlockpt and
       ptsname over the terminal ioctls, then the slave; bytes cross it, the
       slave is a terminal, and the healthy checks follow */
    int master, slave;
    char name[64], line[16];
    if (openpty(&master, &slave, name, NULL, NULL)) { perror("openpty"); return 60; }
    if (strncmp(name, "/dev/pts/", 9) || !isatty(slave)) return 61;
    if (write(master, "ping\n", 5) != 5) return 62;
    ssize_t got = read(slave, line, sizeof line);
    if (got != 5 || memcmp(line, "ping\n", 5)) return 63;
    if (tcgetpgrp(master) != 0) return 64;   /* answered for the slave: no foreground group */
    close(slave);
    close(master);
  }
  char input[16];
  ssize_t n = read(0, input, sizeof input);
  if (n != 6 || memcmp(input, "input\n", 6))
    return 44;
  char cwd[1024];
  if (!getcwd(cwd, sizeof cwd) || strcmp(cwd, "/tmp"))
    return 45;
  if (getsid(0) <= 0 || getpgid(0) <= 0)
    return 46;
  /* the plain rows: a file system's block size, this task's usage, the
     processor count musl reads from sched_getaffinity, a scratch file cut by
     truncate, measured by statx, given a second name by linkat */
  struct statfs fs;
  struct rusage usage;
  struct stat linked;
  struct statx sx;
  int rows = open("contract.rows", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (statfs("/", &fs) || fs.f_bsize <= 0 || getrusage(RUSAGE_SELF, &usage) ||
      sysconf(_SC_NPROCESSORS_ONLN) < 1)
    return 47;
  if (rows < 0 || write(rows, "hello", 5) != 5 || close(rows) || truncate("contract.rows", 2) ||
      statx(AT_FDCWD, "contract.rows", 0, STATX_SIZE, &sx) || sx.stx_size != 2 ||
      link("contract.rows", "contract.link") || stat("contract.link", &linked) ||
      linked.st_size != 2 || linked.st_nlink != 2 || unlink("contract.link") ||
      unlink("contract.rows"))
    return 48;
  /* the descriptor rows: an event counter, a timer that expires once, a
     signal this task blocks and then reads from a signalfd, and the ids */
  uint64_t count = 2;
  int event = eventfd(3, EFD_CLOEXEC);
  if (event < 0 || write(event, &count, 8) != 8 || read(event, &count, 8) != 8 || count != 5 ||
      close(event))
    return 65;
  struct itimerspec expiry = {{0, 0}, {0, 10 * 1000 * 1000}}, left;
  int timer = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
  if (timer < 0 || timerfd_settime(timer, 0, &expiry, NULL) || timerfd_gettime(timer, &left) ||
      read(timer, &count, 8) != 8 || count != 1 || close(timer))
    return 66;
  sigset_t usr1, before;
  struct signalfd_siginfo info;
  sigemptyset(&usr1);
  sigaddset(&usr1, SIGUSR1);
  int sigs = sigprocmask(SIG_BLOCK, &usr1, &before) ? -1 : signalfd(-1, &usr1, SFD_NONBLOCK | SFD_CLOEXEC);
  if (sigs < 0 || kill(getpid(), SIGUSR1) || read(sigs, &info, sizeof info) != sizeof info ||
      info.ssi_signo != SIGUSR1 || close(sigs) || sigprocmask(SIG_SETMASK, &before, NULL))
    return 67;
  uid_t r, e, s;
  gid_t gr, ge, gs;
  if (getresuid(&r, &e, &s) || e != geteuid() || getresgid(&gr, &ge, &gs) || ge != getegid())
    return 68;
  puts("application: ok");
  return 0;
}
