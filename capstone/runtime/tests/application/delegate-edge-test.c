#define _GNU_SOURCE
#include "../../linux/delegate-service.h"
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static char exchange[4096];
static struct capstone_delegate_host host = {.exchange = exchange, .exchange_bytes = sizeof exchange};

static long call6(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d, uint64_t e, uint64_t f) {
  struct capstone_delegate_entry entry;
  uint64_t args[6] = {a, b, c, d, e, f};
  assert(!capstone_delegate_pack(&entry, nr, args));
  capstone_delegate_serve(&host, &entry);
  return entry.result;
}
static long call(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  return call6(nr, a, b, c, d, 0, 0);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  if (!strcmp(argv[1], "short-read")) {
    int p[2];
    assert(!pipe(p));
    assert(write(p[1], "abc", 3) == 3);
    memset(exchange + 64, 'x', 16);
    assert(call(CAPSTONE_SYS_read, p[0], 64, 16, 0) == 3);
    assert(!memcmp(exchange + 64, "abcxxxxxxxxxxxxx", 16));
    assert(call(CAPSTONE_SYS_read, -1, 64, 16, 0) == -EBADF);
    assert(!memcmp(exchange + 64, "abcxxxxxxxxxxxxx", 16));
    close(p[0]); close(p[1]);
  } else if (!strcmp(argv[1], "raw-pointer")) {
    int p[2], available = -123;
    assert(!pipe(p));
    assert(write(p[1], "x", 1) == 1);
    assert(call(CAPSTONE_SYS_ioctl, p[0], FIONREAD, (uintptr_t)&available, 0) < 0);
    assert(available == -123);
    close(p[0]); close(p[1]);
  } else if (!strcmp(argv[1], "vector")) {
    int p[2];
    uint64_t vectors[] = {128, 2, 144, 3};
    char out[5];
    assert(!pipe(p));
    memcpy(exchange + 16, vectors, sizeof vectors);
    memcpy(exchange + 128, "ab", 2);
    memcpy(exchange + 144, "cde", 3);
    assert(call(CAPSTONE_SYS_writev, p[1], 16, 2, 0) == 5);
    assert(read(p[0], out, 5) == 5 && !memcmp(out, "abcde", 5));
    vectors[0] = (uintptr_t)out;
    memcpy(exchange + 16, vectors, sizeof vectors);
    assert(call(CAPSTONE_SYS_writev, p[1], 16, 2, 0) == -EFAULT);
    close(p[0]); close(p[1]);
  } else if (!strcmp(argv[1], "stat-boundary")) {
    host.bounce = malloc(sizeof exchange + 16);
    assert(host.bounce);
    memset(host.bounce + sizeof exchange, 'x', 16);
    int fd = open("/dev/null", O_RDONLY);
    assert(fd >= 0);
    assert(call(CAPSTONE_SYS_fstat, fd, sizeof exchange - 128, 0, 0) == 0);
    assert(!memcmp(host.bounce + sizeof exchange, "xxxxxxxxxxxxxxxx", 16));
    close(fd);
  } else if (!strcmp(argv[1], "private-fd")) {
    int fd = open("/dev/null", O_RDONLY);
    assert(fd >= 0);
    host.private_fds[host.private_count++] = fd;
    assert(call(CAPSTONE_SYS_close, fd, 0, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_dup, fd, 0, 0, 0) == -EBADF);
    assert(call(CAPSTONE_SYS_dup3, 0, fd, 0, 0) == -EBADF);
    assert(fcntl(fd, F_GETFD) >= 0);
    close(fd);
  } else if (!strcmp(argv[1], "wait-empty")) {
    struct capstone_spawner spawner;
    assert(!capstone_spawner_start(&spawner));
    host.spawner = &spawner;
    long result = call(CAPSTONE_SYS_wait4, -1, 16, WNOHANG, 0);
    capstone_spawner_stop(&spawner);
    assert(result == -ECHILD);
  } else if (!strcmp(argv[1], "spawn-target")) {
    /* a file action names descriptors in the child's table, where the
       launcher's own never arrive: dup2 onto a number the launcher uses
       privately, or closing it there, is the application's business; a
       DUP2 source or an FCHDIR directory that is the launcher's is refused */
    struct capstone_spawner spawner;
    int mine = open("/dev/null", O_RDONLY), p[2];
    char *args[] = {"sh", "-c", "echo out >&$CAPSTONE_TARGET; exit 5", NULL};
    char env[32], *envp[] = {"PATH=/usr/bin:/bin", env, NULL};
    size_t bytes;
    assert(mine >= 0 && !pipe(p));
    snprintf(env, sizeof env, "CAPSTONE_TARGET=%d", mine);
    host.private_fds[host.private_count++] = mine;
    assert(!capstone_spawner_start(&spawner));
    host.spawner = &spawner;
    struct capstone_spawn_action onto_mine[] = {{CAPSTONE_SPAWN_DUP2, (uint32_t)mine, (uint32_t)p[1], 0, 0, 0},
                                                {CAPSTONE_SPAWN_CLOSE, (uint32_t)p[1], 0, 0, 0, 0}};
    const char *no_paths[] = {NULL, NULL};
    assert(!capstone_spawn_pack(exchange + 1024, sizeof exchange - 1024, CAPSTONE_SPAWN_SEARCH_PATH, 0,
                                "sh", args, envp, onto_mine, 2, no_paths, &bytes));
    long pid = call(CAPSTONE_NR_SPAWN, 1024, bytes, 0, 0);
    assert(pid > 0);
    close(p[1]);
    char out[8] = {0};
    assert(read(p[0], out, sizeof out) == 4 && !memcmp(out, "out\n", 4));
    assert(call(CAPSTONE_SYS_wait4, (uint64_t)pid, 16, 0, 0) == pid);
    int status;
    memcpy(&status, exchange + 16, sizeof status);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 5);
    struct capstone_spawn_action from_mine[] = {{CAPSTONE_SPAWN_DUP2, 1, (uint32_t)mine, 0, 0, 0}};
    assert(!capstone_spawn_pack(exchange + 1024, sizeof exchange - 1024, CAPSTONE_SPAWN_SEARCH_PATH, 0,
                                "sh", args, envp, from_mine, 1, no_paths, &bytes));
    assert(call(CAPSTONE_NR_SPAWN, 1024, bytes, 0, 0) == -EBADF);
    struct capstone_spawn_action into_mine[] = {{CAPSTONE_SPAWN_FCHDIR, (uint32_t)mine, 0, 0, 0, 0}};
    assert(!capstone_spawn_pack(exchange + 1024, sizeof exchange - 1024, CAPSTONE_SPAWN_SEARCH_PATH, 0,
                                "sh", args, envp, into_mine, 1, no_paths, &bytes));
    assert(call(CAPSTONE_NR_SPAWN, 1024, bytes, 0, 0) == -EBADF);
    capstone_spawner_stop(&spawner);
    assert(fcntl(mine, F_GETFD) >= 0);
    close(mine); close(p[0]);
  } else if (!strcmp(argv[1], "wait-stopped")) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) { raise(SIGSTOP); _exit(0); }
    host.children[host.child_count++] = pid;
    assert(call(CAPSTONE_SYS_wait4, pid, 16, WUNTRACED, 0) == pid);
    int status;
    memcpy(&status, exchange + 16, sizeof status);
    assert(WIFSTOPPED(status));
    unsigned remaining = host.child_count;
    kill(pid, SIGCONT);
    assert(waitpid(pid, &status, 0) == pid);
    assert(remaining == 1);
  } else if (!strcmp(argv[1], "kill-group")) {
    /* the scope of kill: this task, its children and its parent; a process
       group is refused, another process is EPERM, a pid nobody has is ESRCH.
       Signal 0 to the own group signals nothing and Linux answers it with 0,
       since the task is in its group: that one goes through. */
    assert(call(CAPSTONE_SYS_kill, 0, 0, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_kill, 0, SIGTERM, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_kill, (uint64_t)-1, 0, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_kill, (uint64_t)-(int64_t)getpgrp(), 0, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_kill, (uint64_t)getpid(), 0, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_kill, (uint64_t)getppid(), 0, 0, 0) == 0);
    assert(call(CAPSTONE_SYS_kill, 1, 0, 0, 0) == -EPERM);
    assert(call(CAPSTONE_SYS_kill, 2147483000, 0, 0, 0) == -ESRCH);
  } else if (!strcmp(argv[1], "pselect")) {
    /* three fd_sets and a timeout through the buffers, the mask as the sixth
       argument: a readable pipe end is reported, an empty one times out, and a
       mask of a blocked signal is accepted */
    int p[2];
    struct timespec zero = {0, 0};
    uint64_t mask = 0;
    assert(!pipe(p));
    memset(exchange + 128, 0, 128);
    exchange[128 + p[0] / 8] |= (char)(1u << (p[0] % 8));
    memcpy(exchange + 384, &zero, sizeof zero);
    assert(write(p[1], "x", 1) == 1);
    assert(call6(CAPSTONE_SYS_pselect6, (uint64_t)p[0] + 1, 128, 0, 0, 384, 0) == 1);
    assert(exchange[128 + p[0] / 8] & (char)(1u << (p[0] % 8)));
    assert(read(p[0], (char[1]){0}, 1) == 1);
    exchange[128 + p[0] / 8] |= (char)(1u << (p[0] % 8));
    memcpy(exchange + 384, &zero, sizeof zero);
    memcpy(exchange + 512, &mask, sizeof mask);
    assert(call6(CAPSTONE_SYS_pselect6, (uint64_t)p[0] + 1, 128, 0, 0, 384, 512) == 0);
    assert(call(CAPSTONE_SYS_getpgid, 0, 0, 0, 0) == getpgrp());
    assert(call(CAPSTONE_SYS_getsid, 0, 0, 0, 0) == getsid(0));
    close(p[0]); close(p[1]);
  } else if (!strcmp(argv[1], "pty")) {
    /* the pseudo-terminal requests cross through the buffer entry: unlock
       the pair, read its number back, and the kernel's own answer for the
       foreground group; a request outside the list and the integer entry
       for a pointer request stay refused */
    int master = open("/dev/ptmx", O_RDWR | O_NOCTTY | O_CLOEXEC), number = -1, unlock = 0;
    assert(master >= 0);
    assert(!ioctl(master, TIOCGPTN, &number) && number >= 0);
    memcpy(exchange + 64, &unlock, sizeof unlock);
    assert(call(CAPSTONE_NR_IOCTL_BUF, master, TIOCSPTLCK, 64, 0) == 0);
    memset(exchange + 64, 0xff, 4);
    assert(call(CAPSTONE_NR_IOCTL_BUF, master, TIOCGPTN, 64, 0) == 0);
    assert(!memcmp(exchange + 64, &number, sizeof number));
    /* the master's TIOCGPGRP is answered for its slave: no foreground group yet */
    memset(exchange + 64, 0xff, 4);
    assert(call(CAPSTONE_NR_IOCTL_BUF, master, TIOCGPGRP, 64, 0) == 0);
    assert(!memcmp(exchange + 64, &(int){0}, sizeof(int)));
    /* the request as musl's int-taking ioctl would send it: sign-extended */
    memset(exchange + 64, 0xff, 4);
    assert(call(CAPSTONE_NR_IOCTL_BUF, master, (uint64_t)(long)(int)TIOCGPTN, 64, 0) == 0);
    assert(!memcmp(exchange + 64, &number, sizeof number));
    assert(call(CAPSTONE_NR_IOCTL_BUF, master, TIOCSTI, 64, 0) == -ENOSYS);
    assert(call(CAPSTONE_SYS_ioctl, master, TIOCGPTN, 64, 0) == -ENOSYS);
    close(master);
  } else {
    abort();
  }
  capstone_delegate_host_free(&host);
  return 0;
}
