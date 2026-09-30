/* End-to-end checks of libc marshalling and process state, run in a domain. */
#define _GNU_SOURCE
#include <errno.h>
#include <elf.h>
#include <malloc.h>
#include <fcntl.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;
#define CHECK(test) do { if (!(test)) { \
  fprintf(stderr, "delegate-contract:%d: %s (errno %d)\n", __LINE__, #test, errno); return 1; \
} } while (0)

static int waited(pid_t pid, int code) {
  int status = 0;
  return waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == code;
}

int main(int argc, char **argv) {
  CHECK(argc >= 2);
  if (!strcmp(argv[1], "after-closed-exec")) {
    for (int fd = 0; fd < 3; ++fd) CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
    return 0;
  }
  if (!strcmp(argv[1], "child")) {
    CHECK(!strcmp(argv[0], "custom argv zero"));
    return 7;
  }
  if (!strcmp(argv[1], "after-exec")) {
    CHECK(!strcmp(argv[0], "custom argv zero"));
    CHECK(getpid() == atoi(getenv("REVIEW_PID")));
    CHECK(waited(atoi(getenv("REVIEW_CHILD")), 9));
    CHECK(system("exit 3") == 3 * 256);
    puts("delegate-contract: exec ok");
    return 0;
  }
  if (!strcmp(argv[1], "lock-child")) {
    /* argv[2]: the file; argv[3]: the pid that holds a write lock on its bytes 0 to 99, or 0
       once it has released them. Another process's view of those locks, through the
       struct flock pointer fcntl takes (musl patch 0005). */
    CHECK(argc == 4);
    pid_t owner = (pid_t)atoi(argv[3]);
    int fd = open(argv[2], O_RDWR);
    CHECK(fd >= 0);
    struct flock probe = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 10, .l_len = 1};
    CHECK(fcntl(fd, F_GETLK, &probe) == 0);
    struct flock byte = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 10, .l_len = 1};
    if (owner) {
      CHECK(probe.l_type == F_WRLCK && probe.l_pid == owner && probe.l_start == 0 &&
            probe.l_len == 100);
      CHECK(fcntl(fd, F_SETLK, &byte) == -1 && (errno == EAGAIN || errno == EACCES));
      struct flock beyond = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 100, .l_len = 1};
      CHECK(fcntl(fd, F_SETLK, &beyond) == 0);
    } else {
      CHECK(probe.l_type == F_UNLCK);
      CHECK(fcntl(fd, F_SETLK, &byte) == 0);
    }
    return 8;
  }
  if (!strcmp(argv[1], "usable-size")) {
    /* malloc_usable_size, answered by the runtime's heap: at least what was asked for, and
       every byte it reports is the block's own. Filling a whole reported block must leave
       its neighbour, allocated after it, untouched. */
    CHECK(malloc_usable_size(NULL) == 0);
    for (size_t n = 1; n <= 5000; n = n * 3 + 1) {
      unsigned char *a = malloc(n), *b = malloc(n);
      CHECK(a && b);
      memset(b, 0x5a, n);
      size_t usable = malloc_usable_size(a);
      CHECK(usable >= n);
      memset(a, 0xa5, usable);
      for (size_t i = 0; i < n; ++i)
        CHECK(b[i] == 0x5a);
      if (usable >= sizeof(void *)) {
        /* a pointer stored in the last aligned slot it reports survives */
        void **slot = (void **)(a + ((usable - sizeof(void *)) & ~(sizeof(void *) - 1)));
        *slot = b;
        CHECK(*slot == b && **(unsigned char **)slot == 0x5a);
      }
      free(a);
      free(b);
    }
    puts("delegate-contract: usable size ok");
    return 0;
  }
  if (!strcmp(argv[1], "buffer-bounds")) {
    int fds[2];
    char *p = malloc(16);
    const char contents[] = "0123456789abcdefghijklmnopqrstuv";
    CHECK(p && !pipe(fds));
    CHECK(write(fds[1], contents, sizeof contents - 1) == sizeof contents - 1);
    /* Rejection must happen before the file offset or pipe contents change. */
    errno = 0;
    CHECK(read(fds[0], p, 4096) == -1 && errno == EFAULT);
    errno = 0;
    CHECK(read(fds[0], (void *)0x1000, 1) == -1 && errno == EFAULT);
    char *readonly = __builtin_capstone_cap_tighten(p, 4);
    errno = 0;
    CHECK(read(fds[0], readonly, 1) == -1 && errno == EFAULT);
    struct iovec iov[2] = {{p, 16}, {p, 17}};
    errno = 0;
    CHECK(readv(fds[0], iov, 2) == -1 && errno == EFAULT);
    CHECK(read(fds[0], p, 16) == 16 && !memcmp(p, contents, 16));
    CHECK(read(fds[0], p, 16) == 16 && !memcmp(p, contents + 16, 16));
    errno = 0;
    CHECK(write(fds[1], p, 4096) == -1 && errno == EFAULT);
    char *writeonly = __builtin_capstone_cap_tighten(p, 2);
    errno = 0;
    CHECK(write(fds[1], writeonly, 1) == -1 && errno == EFAULT);
    CHECK(!close(fds[1]));
    CHECK(read(fds[0], p, 16) == 0);
    CHECK(!close(fds[0]));
    free(p);
    puts("delegate-contract: buffer bounds ok");
    return 0;
  }
  CHECK(argc == 3);
  char *image = argv[2];
  if (!strcmp(argv[1], "lock")) {
    /* POSIX record locks: one this process holds, as a second process sees it, and after
       its release through F_SETLKW (musl's cancellable path). A process never conflicts
       with its own locks, so F_GETLK here reports none. */
    char path[] = "/tmp/capstone-lock-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    CHECK(!ftruncate(fd, 4096));
    struct flock mine = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 100};
    CHECK(fcntl(fd, F_SETLK, &mine) == 0);
    struct flock own = {.l_type = F_WRLCK, .l_whence = SEEK_SET, .l_start = 10, .l_len = 1};
    CHECK(fcntl(fd, F_GETLK, &own) == 0 && own.l_type == F_UNLCK);
    char number[32];
    snprintf(number, sizeof number, "%d", (int)getpid());
    char *held[] = {"lock-child", "lock-child", path, number, NULL};
    pid_t pid;
    CHECK(!posix_spawn(&pid, image, NULL, NULL, held, environ));
    CHECK(waited(pid, 8));
    struct flock off = {.l_type = F_UNLCK, .l_whence = SEEK_SET, .l_start = 0, .l_len = 100};
    CHECK(fcntl(fd, F_SETLKW, &off) == 0);
    char *freed[] = {"lock-child", "lock-child", path, "0", NULL};
    CHECK(!posix_spawn(&pid, image, NULL, NULL, freed, environ));
    CHECK(waited(pid, 8));
    close(fd);
    CHECK(!unlink(path));
    puts("delegate-contract: record locks ok");
    return 0;
  }
  if (!strcmp(argv[1], "exec-error")) {
    char path[] = "/tmp/capstone-bad-exec-XXXXXX";
    int fd = mkstemp(path);
    CHECK(fd >= 0);
    Elf64_Ehdr header = {0};
    memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_machine = 259;
    CHECK(write(fd, &header, sizeof header) == sizeof header);
    close(fd);
    char *next[] = {"custom argv zero", NULL};
    CHECK(execve(path, next, environ) == -1 && errno == ENOEXEC);
    CHECK(!unlink(path));
    CHECK(execve(path, next, environ) == -1 && errno == ENOENT);
    CHECK(system("exit 4") == 4 * 256);
    puts("delegate-contract: failed exec preserved task");
    return 0;
  }
  if (!strcmp(argv[1], "exec-closed")) {
    for (int fd = 0; fd < 3; ++fd) close(fd);
    char *next[] = {"custom argv zero", "after-closed-exec", NULL};
    execve(image, next, environ);
    return 1;
  }
  if (!strcmp(argv[1], "exec")) {
    pid_t pid;
    char *args[] = {"sh", "-c", "sleep 0.1; exit 9", NULL};
    CHECK(!posix_spawn(&pid, "/bin/sh", NULL, NULL, args, environ));
    char number[32];
    snprintf(number, sizeof number, "%d", (int)pid);
    CHECK(!setenv("REVIEW_CHILD", number, 1));
    snprintf(number, sizeof number, "%d", (int)getpid());
    CHECK(!setenv("REVIEW_PID", number, 1));
    char *next[] = {"custom argv zero", "after-exec", NULL};
    execve(image, next, environ);
    CHECK(0);
  }
  int p[2];
  char buffer[16];
  CHECK(!pipe(p));
  CHECK(write(p[1], NULL, 0) == 0);
  CHECK(read(p[0], NULL, 0) == 0);
  CHECK(write(p[1], "abc", 3) == 3);
  memset(buffer, 'x', sizeof buffer);
  CHECK(read(p[0], buffer, sizeof buffer) == 3);
  CHECK(!memcmp(buffer, "abcxxxxxxxxxxxxx", sizeof buffer));
  CHECK(read(-1, buffer, sizeof buffer) == -1 && errno == EBADF);
  CHECK(!memcmp(buffer, "abcxxxxxxxxxxxxx", sizeof buffer));
  struct iovec out[] = {{"ab", 2}, {"cde", 3}};
  CHECK(writev(p[1], out, 2) == 5);
  struct iovec in[] = {{buffer, 2}, {buffer + 4, 8}};
  CHECK(readv(p[0], in, 2) == 5);
  CHECK(!memcmp(buffer, "abcxcdexxxxxxxxx", sizeof buffer));
  close(p[0]); close(p[1]);
  int status;
  CHECK(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD);
  CHECK(!chdir("/tmp"));
  umask(0037);
  CHECK(system("test \"$(pwd)\" = /tmp && test \"$(umask)\" = 0037") == 0);
  pid_t pid;
  char *args[] = {"custom argv zero", "child", NULL};
  CHECK(!posix_spawn(&pid, image, NULL, NULL, args, environ));
  CHECK(waited(pid, 7));
  char *base = strrchr(image, '/');
  CHECK(base);
  *base = 0;
  CHECK(!setenv("PATH", image, 1));
  char *child_env[] = {"PATH=/no/such/directory", NULL};
  CHECK(!posix_spawnp(&pid, base + 1, NULL, NULL, args, child_env));
  *base = '/';
  CHECK(waited(pid, 7));
  puts("delegate-contract: io and spawn ok");
  return 0;
}
