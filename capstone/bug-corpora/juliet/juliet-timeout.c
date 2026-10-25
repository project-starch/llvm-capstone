/* juliet-timeout STEP LIMIT_MS -- PROGRAM [ARGS...]
 *
 * Runs PROGRAM in its own process group with stdin from /dev/null and a monotonic deadline.
 * On expiry: SIGKILL to the whole group, reap, print
 *   [runner] TIMEOUT step=STEP limit_ms=LIMIT reaped=R group_gone=G
 * and return 124. Otherwise print
 *   [runner] WAIT step=STEP exit=N        or   [runner] WAIT step=STEP signal=N core=C
 * and return N, or 128+N for a signal (the shell's convention, which virtualvm.split() reads).
 * reaped is 1 when waitpid returned the child; group_gone is 1 when no member of the group can still be signalled.
 * The helper itself stays outside the group, so the kill never reaches the batch shell.
 * Build it static with the guest kernel's cross toolchain (the adapter's CROSS_COMPILE):
 *   ${CROSS_COMPILE}gcc -O2 -static -Wall -Wextra -Werror juliet-timeout.c -o juliet-timeout */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static long long now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

int main(int argc, char **argv) {
  if (argc < 5 || strcmp(argv[3], "--") != 0) {
    fprintf(stderr, "usage: %s STEP LIMIT_MS -- PROGRAM [ARGS...]\n", argv[0]);
    return 125;
  }
  const char *step = argv[1];
  long limit = strtol(argv[2], NULL, 10);
  if (limit <= 0) { fprintf(stderr, "bad LIMIT_MS\n"); return 125; }
  fflush(NULL);
  pid_t child = fork();
  if (child < 0) { perror("fork"); return 125; }
  if (child == 0) {
    setpgid(0, 0);
    sigset_t none; sigemptyset(&none); sigprocmask(SIG_SETMASK, &none, NULL);
    for (int s = 1; s < NSIG; s++) signal(s, SIG_DFL);
    int fd = open("/dev/null", O_RDONLY);
    if (fd >= 0) { dup2(fd, 0); if (fd > 0) close(fd); }
    execvp(argv[4], argv + 4);
    perror("execvp");
    _exit(127);
  }
  setpgid(child, child); /* both sides set it: no race with an early kill */
  long long deadline = now_ms() + limit;
  int status = 0;
  for (;;) {
    pid_t r = waitpid(child, &status, WNOHANG);
    if (r == child) break;
    if (r < 0 && errno != EINTR) { perror("waitpid"); return 125; }
    if (now_ms() >= deadline) {
      kill(-child, SIGKILL);
      pid_t w;
      while ((w = waitpid(child, &status, 0)) < 0 && errno == EINTR) {}
      /* reap any other group members that are our children (none expected) */
      while (waitpid(-child, NULL, WNOHANG) > 0) {}
      /* the group is gone once no member can be signalled (ESRCH); allow a short grace period */
      int gone = 0;
      for (int i = 0; i < 200 && !gone; i++) {
        if (kill(-child, 0) < 0 && errno == ESRCH) gone = 1;
        else { struct timespec g = {0, 5 * 1000000}; nanosleep(&g, NULL); }
      }
      printf("[runner] TIMEOUT step=%s limit_ms=%ld reaped=%d group_gone=%d\n", step, limit, w == child, gone);
      fflush(stdout);
      return 124;
    }
    struct timespec ts = {0, 5 * 1000000};
    nanosleep(&ts, NULL);
  }
  /* kill stragglers left in the group after the leader ended (e.g. a spawner) */
  kill(-child, SIGKILL);
  if (WIFEXITED(status)) {
    printf("[runner] WAIT step=%s exit=%d\n", step, WEXITSTATUS(status));
    fflush(stdout);
    return WEXITSTATUS(status);
  }
  if (WIFSIGNALED(status)) {
    printf("[runner] WAIT step=%s signal=%d core=%d\n", step, WTERMSIG(status), WCOREDUMP(status) ? 1 : 0);
    fflush(stdout);
    return 128 + WTERMSIG(status);
  }
  printf("[runner] WAIT step=%s status=0x%x\n", step, status);
  return 125;
}
