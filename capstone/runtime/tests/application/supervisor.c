#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int check_file(const char *path, const char *expected) {
  char data[256];
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 0;
  ssize_t n = read(fd, data, sizeof data);
  close(fd);
  return n == (ssize_t)strlen(expected) && !memcmp(data, expected, (size_t)n);
}

static int run(const char *launcher, const char *image, const char *mode) {
  pid_t child = fork();
  if (child < 0) return 0;
  if (!child) {
    int in = open("contract.in", O_RDONLY);
    int out = open("contract.out", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    int err = open("contract.err", O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (in < 0 || out < 0 || err < 0 || dup2(in, 0) < 0 ||
        dup2(out, 1) < 0 || dup2(err, 2) < 0) _exit(124);
    close(in); close(out); close(err);
    setenv("CAPSTONE_CONTRACT", "environment with spaces", 1);
    execl(launcher, launcher, image, mode, "",
          "argument with spaces\nand newline", (char *)NULL);
    _exit(124);
  }
  int status;
  while (waitpid(child, &status, 0) < 0) {
    if (errno != EINTR) return 0;
  }
  int fault = !strncmp(mode, "fault", 5), exited = !strcmp(mode, "exit139");
  int passed = fault ? WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV
                     : WIFEXITED(status) && WEXITSTATUS(status) == (exited ? 139 : 0);
  passed = passed && check_file("contract.out", fault || exited ?
      "stdout\n" : "stdout\napplication: ok\n") && check_file("contract.err", "stderr\n");
  printf("application %s: %s (wait status=%d)\n", mode, passed ? "PASS" : "FAIL", status);
  fflush(stdout);
  return passed;
}

int main(int argc, char **argv) {
  if (argc != 3 || chdir("/tmp")) return 2;
  int input = open("contract.in", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (input < 0 || write(input, "input\n", 6) != 6) return 2;
  close(input);
  if (!run(argv[1], argv[2], "healthy") || !run(argv[1], argv[2], "fault") ||
      !run(argv[1], argv[2], "fault-stack") ||
      !run(argv[1], argv[2], "exit139") || !run(argv[1], argv[2], "healthy")) return 1;
  puts("application sequence: PASS");
  return 0;
}
