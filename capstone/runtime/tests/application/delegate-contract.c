/* End-to-end checks of libc marshalling and process state, run in a domain. */
#define _GNU_SOURCE
#include <errno.h>
#include <elf.h>
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
  CHECK(argc == 3);
  char *image = argv[2];
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
