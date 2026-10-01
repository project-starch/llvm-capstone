/* Shared by the domain probe and its native test. Observe the worker's actual
 * Linux read before signalling it: elapsed time does not prove it is blocked.
 * An early-delivery arm forces the schedule that used to hang the probe. */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static int pipefd[2], reader_tid, read_rc, read_errno, name_error;
static _Atomic int ready, go, done, signals_seen, signal_tid;

static long now_ms(void)
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1000 + t.tv_nsec / 1000000;
}

static void pause_poll(void)
{
  struct timespec t = {0, 1000000};
  nanosleep(&t, 0);
}

static void caught(int sig)
{
  (void)sig;
  atomic_store(&signal_tid, (int)syscall(SYS_gettid));
  atomic_fetch_add(&signals_seen, 1);
}

static void *reader(void *arg)
{
  (void)arg;
  char c;
  reader_tid = (int)syscall(SYS_gettid);
  name_error = pthread_setname_np(pthread_self(), "probe-reader");
  atomic_store(&ready, 1);
  while (!atomic_load(&go))
    sched_yield(); /* synchronous domain signal delivery before the read */
  read_rc = (int)read(pipefd[0], &c, 1);
  read_errno = errno;
  atomic_store(&done, 1);
  return NULL;
}

static int wait_for(_Atomic int *word, int value)
{
  long until = now_ms() + 3000;
  while (atomic_load(word) != value) {
    if (now_ms() >= until)
      return -1;
    pause_poll();
  }
  return 0;
}

/* The domain tid is virtual. Find its serving Linux thread by the name the
 * worker set, then inspect /proc's syscall snapshot. No other writer or signal
 * can end this empty-pipe read between the observation and pthread_kill. */
static int blocked_reader(void)
{
  DIR *dir = opendir("/proc/self/task");
  if (!dir)
    return 0;
  struct dirent *entry;
  int blocked = 0;
  while (!blocked && (entry = readdir(dir))) {
    char *end;
    long tid = strtol(entry->d_name, &end, 10);
    if (*end || tid <= 0)
      continue;
    char path[96], name[32];
    snprintf(path, sizeof path, "/proc/self/task/%ld/comm", tid);
    FILE *f = fopen(path, "r");
    if (!f)
      continue;
    int match = fgets(name, sizeof name, f) && !strcmp(name, "probe-reader\n");
    fclose(f);
    if (!match)
      continue;
    snprintf(path, sizeof path, "/proc/self/task/%ld/syscall", tid);
    f = fopen(path, "r");
    if (!f)
      continue;
    long nr;
    unsigned long fd;
    blocked = fscanf(f, "%ld %lx", &nr, &fd) == 2 && nr == SYS_read && fd == (unsigned long)pipefd[0];
    fclose(f);
  }
  closedir(dir);
  return blocked;
}

int capstone_probe_kill_thread(int early)
{
  pthread_t thread;
  struct sigaction sa = {0}, old;
  int failed = 0;
  const char *reason = NULL;
  atomic_store(&ready, 0);
  atomic_store(&go, 0);
  atomic_store(&done, 0);
  atomic_store(&signals_seen, 0);
  atomic_store(&signal_tid, 0);
  sa.sa_handler = caught;
  sigemptyset(&sa.sa_mask);
  if (pipe(pipefd))
    return 1;
  if (sigaction(SIGUSR1, &sa, &old)) {
    close(pipefd[0]); close(pipefd[1]);
    return 1;
  }
  if (pthread_create(&thread, NULL, reader, NULL)) {
    sigaction(SIGUSR1, &old, NULL);
    close(pipefd[0]); close(pipefd[1]);
    return 1;
  }
  if (wait_for(&ready, 1) || name_error) {
    reason = "reader did not become ready";
    goto cleanup;
  }
  if (early && (pthread_kill(thread, SIGUSR1) || wait_for(&signals_seen, 1))) {
    reason = "early signal did not run before read";
    goto cleanup;
  }
  atomic_store(&go, 1);
  long until = now_ms() + 3000;
  while (!blocked_reader()) {
    if (atomic_load(&done) || now_ms() >= until) {
      reason = "reader was not observed blocked in read";
      goto cleanup;
    }
    pause_poll();
  }
#ifndef CAPSTONE_KILL_PROBE_NO_SIGNAL
  if (pthread_kill(thread, SIGUSR1)) {
    reason = "pthread_kill failed";
    goto cleanup;
  }
#endif
  if (wait_for(&done, 1))
    reason = "signal did not interrupt the blocked read";
cleanup:
  if (reason) {
    fprintf(stderr, "kill-thread: %s\n", reason);
    failed = 1;
    atomic_store(&go, 1);
    /* End an outstanding (or not yet entered) read even on a broken signal
     * path. The runner's timeout remains the guard against broken joins. */
    if (write(pipefd[1], "x", 1) != 1)
      failed = 1;
  }
  if (pthread_join(thread, NULL))
    failed = 1;
  if (!failed && (atomic_load(&signals_seen) != 1 + early ||
                 atomic_load(&signal_tid) != reader_tid || read_rc != -1 || read_errno != EINTR))
    failed = 1;
  printf("kill-thread: early=%d signals=%d handler=%d reader=%d read=%d errno=%d %s\n",
         early, atomic_load(&signals_seen), atomic_load(&signal_tid), reader_tid,
         read_rc, read_errno, failed ? "FAIL" : "PASS");
  sigaction(SIGUSR1, &old, NULL);
  close(pipefd[0]); close(pipefd[1]);
  return failed;
}

#ifdef CAPSTONE_KILL_PROBE_MAIN
int main(int argc, char **argv)
{
  (void)argv;
  return capstone_probe_kill_thread(argc > 1);
}
#endif
