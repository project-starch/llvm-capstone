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

static long call(uint64_t nr, uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
  struct capstone_delegate_entry entry;
  uint64_t args[6] = {a, b, c, d, 0, 0};
  assert(!capstone_delegate_pack(&entry, nr, args));
  capstone_delegate_serve(&host, &entry);
  return entry.result;
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
    assert(call(CAPSTONE_SYS_kill, 0, 0, 0, 0) == -EPERM);
  } else {
    abort();
  }
  capstone_delegate_host_free(&host);
  return 0;
}
