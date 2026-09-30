#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pty.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/wait.h>
#include <unistd.h>
#include <capstone/capability.h>

void *__capstone_region(unsigned);
static volatile unsigned char *volatile saved_allocation;

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
    return *stale;
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
    if (!strcmp(argv[1], "fault-reused")) return *saved_allocation;
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
  puts("application: ok");
  return 0;
}
