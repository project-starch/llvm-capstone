#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static int check_file(const char *path, const char *expected) {
  char data[256];
  int fd = open(path, O_RDONLY);
  if (fd < 0) return 0;
  ssize_t n = read(fd, data, sizeof data);
  close(fd);
  return n == (ssize_t)strlen(expected) && !memcmp(data, expected, (size_t)n);
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}

static int run(const char *launcher, const char *image, const char *mode, int stop) {
  int clear = open("contract.out", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (clear < 0) return 0;
  close(clear);
  clear = open("contract.err", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (clear < 0) return 0;
  close(clear);
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
  int status = 0;
  double deadline = now() + 20;
  if (stop) {
    while ((!check_file("contract.out", "stdout\n") ||
            !check_file("contract.err", "stderr\n")) && now() < deadline) {
      struct timespec pause = {.tv_nsec = 10000000};
      nanosleep(&pause, NULL);
    }
    /* The program must have reached main before cancellation is measured. */
    if (!check_file("contract.out", "stdout\n")) { kill(child, SIGKILL); return 0; }
    struct timespec running = {.tv_nsec = 50000000};
    nanosleep(&running, NULL);
    if (waitpid(child, &status, WNOHANG) != 0) return 0;
    kill(child, stop);
    deadline = now() + 5;
  }
  for (;;) {
    pid_t result = waitpid(child, &status, WNOHANG);
    if (result == child) break;
    if (result < 0 && errno != EINTR) return 0;
    if (now() > deadline) {
      char path[64], state[512];
      snprintf(path, sizeof path, "/proc/%ld/stat", (long)child);
      int fd = open(path, O_RDONLY);
      ssize_t n = fd < 0 ? -1 : read(fd, state, sizeof state - 1);
      if (fd >= 0) close(fd);
      if (n >= 0) state[n] = 0;
      fprintf(stderr, "application %s: timeout waiting for %ld; state=%s\n",
              mode, (long)child, n >= 0 ? state : "unavailable");
      kill(child, SIGKILL); waitpid(child, &status, 0); return 0;
    }
    struct timespec pause = {.tv_nsec = 1000000};
    nanosleep(&pause, NULL);
  }
  int fault = !strncmp(mode, "fault", 5), exited = !strcmp(mode, "exit139");
  int passed = stop ? WIFSIGNALED(status) && WTERMSIG(status) == stop : fault ? WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV
                     : WIFEXITED(status) && WEXITSTATUS(status) == (exited ? 139 : 0);
  passed = passed && check_file("contract.out", fault || exited || stop ?
      "stdout\n" : "stdout\napplication: ok\n") && check_file("contract.err", "stderr\n");
  printf("application %s: %s (wait status=%d)\n", mode, passed ? "PASS" : "FAIL", status);
  fflush(stdout);
  return passed;
}

int main(int argc, char **argv) {
  if ((argc != 3 && argc != 4 && argc != 5) || chdir("/tmp")) return 2;
  unsigned count = argc == 4 ? (unsigned)strtoul(argv[3], NULL, 10) : 1;
  if (!count || count > 10000) return 2;
  int input = open("contract.in", O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (input < 0 || write(input, "input\n", 6) != 6) return 2;
  close(input);
  if (argc == 5) {
    if (strcmp(argv[3], "--mode")) return 2;
    return run(argv[1], argv[2], argv[4], 0) ? 0 : 1;
  }
  const char *modes[] = {"healthy", "fault", "fault-stack", "fault-vector", "exit139"};
  for (unsigned i = 0; i < count; ++i)
    for (unsigned j = 0; j < sizeof modes / sizeof modes[0]; ++j)
      if (!run(argv[1], argv[2], modes[j], 0)) return 1;
  const char *invalid[] = {"fault-mrev", "fault-privcsr", "fault-debugcap", "fault-capenter"};
  for (unsigned j = 0; j < sizeof invalid / sizeof invalid[0]; ++j)
    if (!run(argv[1], argv[2], invalid[j], 0)) return 1;
  if (!run(argv[1], argv[2], "loop", SIGTERM) ||
      !run(argv[1], argv[2], "loop", SIGKILL) ||
      !run(argv[1], argv[2], "loop", SIGINT) ||
      !run(argv[1], argv[2], "healthy", 0)) return 1;
  puts("application sequence: PASS");
  return 0;
}
