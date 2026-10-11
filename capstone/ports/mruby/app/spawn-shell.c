/* mruby's POSIX IO HAL, expressed with spawn actions instead of fork. */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <spawn.h>
#include <stdlib.h>
#include <unistd.h>

extern char **environ;

int capstone_mruby_spawn(const char *cmd, int in, int out, int err, int *pid) {
  while (*cmd == ' ' || *cmd == '\t' || *cmd == '\n') ++cmd;
  if (!*cmd) { errno = ENOENT; return -1; }
  int source[3] = {in, out, err}, parked[3] = {-1, -1, -1};
  posix_spawn_file_actions_t actions;
  int error = posix_spawn_file_actions_init(&actions);
  if (error) { errno = error; return -1; }
  /* Keep the original sources when redirections overlap standard descriptors. */
  for (int i = 0; i < 3 && !error; ++i) {
    if (source[i] < 0) continue;
    parked[i] = fcntl(source[i], F_DUPFD_CLOEXEC, 3);
    if (parked[i] < 0) error = errno;
    else error = posix_spawn_file_actions_adddup2(&actions, parked[i], i);
  }
  /* The child must not inherit either end of the parent's other pipes. The
     application is single-threaded, so its descriptor set cannot race here. */
  DIR *directory = error ? NULL : opendir("/proc/self/fd");
  if (!error && !directory) error = errno;
  if (directory) {
    struct dirent *entry;
    errno = 0;
    while (!error && (entry = readdir(directory))) {
      char *end;
      long fd = strtol(entry->d_name, &end, 10);
      if (end != entry->d_name && !*end && fd >= 3 && fd <= INT_MAX && fd != dirfd(directory)) {
        /* /proc also lists the launcher's private descriptors. Its fcntl
           boundary deliberately presents them to the application as EBADF. */
        if (fcntl((int)fd, F_GETFD) >= 0)
          error = posix_spawn_file_actions_addclose(&actions, (int)fd);
        else if (errno != EBADF)
          error = errno;
      }
      if (!error) errno = 0;
    }
    if (!error && errno) error = errno;
    closedir(directory);
  }
  pid_t child;
  char *args[] = {"sh", "-c", (char *)cmd, NULL};
  if (!error) error = posix_spawn(&child, "/bin/sh", &actions, NULL, args, environ);
  posix_spawn_file_actions_destroy(&actions);
  for (int i = 0; i < 3; ++i) if (parked[i] >= 0) close(parked[i]);
  if (error) { errno = error; return -1; }
  *pid = child;
  return 0;
}
