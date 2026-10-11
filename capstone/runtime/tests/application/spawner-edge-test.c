#define _GNU_SOURCE
#include "../../linux/spawner.h"
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char **argv) {
  assert(argc >= 2);
  if (!strcmp(argv[1], "check-fd"))
    return fcntl(atoi(argv[2]), F_GETFD) < 0 && errno == EBADF ? 0 : 1;
  struct capstone_spawner s;
  char block[CAPSTONE_SPAWN_BYTES], script[512], path[512], fd_text[32];
  int stale = open("/dev/null", O_RDONLY);
  assert(stale >= 0);
  assert(realpath(argv[0], path));
  assert(!capstone_spawner_start(&s));
  char *args[] = {"sh", "-c", script, NULL};
  char *env[] = {"PATH=/bin:/usr/bin", NULL};
  const char *program = "/bin/sh";
  struct capstone_spawn_action actions[64] = {{0}};
  unsigned action_count = 0;
  char dir[] = "/tmp/capstone-spawn-XXXXXX";
  if (!strcmp(argv[1], "cwd")) {
    assert(mkdtemp(dir));
    assert(!chdir(dir));
    snprintf(script, sizeof script, "test \"$(pwd)\" = '%s'", dir);
  } else if (!strcmp(argv[1], "umask")) {
    umask(0037);
    strcpy(script, "test \"$(umask)\" = 0037");
  } else if (!strcmp(argv[1], "stale-fd")) {
    close(stale);
    snprintf(fd_text, sizeof fd_text, "%d", stale);
    program = path;
    args[0] = path; args[1] = "check-fd"; args[2] = fd_text;
  } else if (!strcmp(argv[1], "status-close")) {
    for (unsigned i = 0; i < 64; ++i)
      actions[action_count++] = (struct capstone_spawn_action){CAPSTONE_SPAWN_CLOSE, i, 0, 0, 0, 0};
    program = "/no/such/program";
    strcpy(script, "exit 0");
  } else if (!strcmp(argv[1], "fd-overflow")) {
    int fds[64], numbers[64];
    uint64_t flags;
    for (int i = 0; i < 65; ++i) assert(open("/dev/null", O_RDONLY) >= 0);
    int count = capstone_spawner_descriptors(fds, numbers, &flags, 64, s.socket);
    capstone_spawner_stop(&s);
    assert(count == -EMFILE);
    return 0;
  } else abort();
  size_t bytes;
  assert(!capstone_spawn_pack(block, sizeof block, 0, 0, program, args, env,
                              actions, action_count, NULL, &bytes));
  long pid = capstone_spawner_spawn(&s, block, bytes, NULL, NULL, 0, 0, 0, 0);
  int status = 0;
  if (pid > 0) assert(waitpid(pid, &status, 0) == pid);
  capstone_spawner_stop(&s);
  if (!strcmp(argv[1], "status-close")) assert(pid == -ENOENT);
  else assert(pid > 0 && WIFEXITED(status) && !WEXITSTATUS(status));
  if (!strcmp(argv[1], "cwd")) { assert(!chdir("/")); assert(!rmdir(dir)); }
  return 0;
}
