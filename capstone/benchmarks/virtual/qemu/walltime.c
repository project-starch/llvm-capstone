/* Guest-side helper of run-qemu.py (built natively, static): run one command with a timeout and
 * print `WALL ns=<elapsed> rc=<exit status or 128+signal>[ timeout]` when it ends. Under QEMU
 * -icount shift=0 the guest clock advances 1 ns per retired instruction, so ns is the whole run's
 * instruction count (launcher, kernel and image together) -- the "whole-run" estimate. */
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

static pid_t child;
static volatile sig_atomic_t timed_out;
static void on_alarm(int s) {
  (void)s;
  timed_out = 1;
  kill(child, SIGKILL);
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: walltime SECONDS COMMAND [ARG...]\n");
    return 2;
  }
  struct timespec t0, t1;
  fflush(stdout);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  child = fork();
  if (child == 0) {
    execv(argv[2], argv + 2);
    perror("execv");
    _exit(127);
  }
  signal(SIGALRM, on_alarm);
  alarm((unsigned)atoi(argv[1]));
  int status = 0;
  while (waitpid(child, &status, 0) < 0)
    ;
  clock_gettime(CLOCK_MONOTONIC, &t1);
  int rc = WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
  long long ns = (long long)(t1.tv_sec - t0.tv_sec) * 1000000000LL + (t1.tv_nsec - t0.tv_nsec);
  printf("WALL ns=%lld rc=%d%s\n", ns, rc, timed_out ? " timeout" : "");
  fflush(stdout);
  return 0;
}
