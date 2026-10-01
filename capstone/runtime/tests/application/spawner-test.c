/* Native test of the spawner: a real child of this process, its descriptors
 * where the block says, its status through wait4, and an exec failure. */
#include "../../linux/spawner.h"
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static char block[CAPSTONE_SPAWN_BYTES];

int main(void) {
  struct capstone_spawner s;
  int fds[CAPSTONE_SPAWNER_FDS], numbers[CAPSTONE_SPAWNER_FDS];
  unsigned count;
  size_t bytes;
  int out[2], status;
  char buffer[64];
  assert(!capstone_spawner_start(&s));
  assert(!pipe2(out, O_CLOEXEC));
  /* stdout of the child is our pipe; the pipe itself is not inherited */
  struct capstone_spawn_action actions[] = {{CAPSTONE_SPAWN_DUP2, 1, (uint32_t)out[1], 0, 0, 0}};
  char *argv[] = {"sh", "-c", "echo from a child; exit 3", NULL};
  char *envp[] = {"PATH=/usr/bin:/bin", NULL};
  assert(!capstone_spawn_pack(block, sizeof block, CAPSTONE_SPAWN_SEARCH_PATH, 0, "sh", argv,
                              envp, actions, 1, (const char *[]){NULL}, &bytes));
  /* pass the pipe's write end so the dup2 can find it, plus the standard fds */
  /* the pipe end is close-on-exec, as musl's popen makes it: the dup2 action
     may still use it, and the number itself is gone after exec */
  fds[0] = out[1]; numbers[0] = out[1];
  fds[1] = 0; numbers[1] = 0;
  fds[2] = 2; numbers[2] = 2;
  count = 3;
  long pid = capstone_spawner_spawn(&s, block, bytes, fds, numbers, 1, count, 0, 0);
  assert(pid > 0);
  close(out[1]);
  ssize_t n = read(out[0], buffer, sizeof buffer - 1);
  assert(n > 0);
  buffer[n] = 0;
  assert(!strcmp(buffer, "from a child\n"));
  assert(waitpid((pid_t)pid, &status, 0) == pid);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 3);
  /* a program that does not exist: an error, and no child left behind */
  char *missing[] = {"no-such-program", NULL};
  assert(!capstone_spawn_pack(block, sizeof block, 0, 0, "/no/such/program", missing, envp,
                              NULL, 0, NULL, &bytes));
  assert(capstone_spawner_spawn(&s, block, bytes, NULL, NULL, 0, 0, 0, 0) == -ENOENT);
  /* the failed child was reaped: the only child left is the spawner, still
     running, and cloned without an exit signal so that a plain wait never
     sees it; only __WALL does */
  assert(waitpid(-1, &status, WNOHANG) == -1 && errno == ECHILD);
  /* A child whose exec failed keeps the helper's exit signal, none: under
     CLONE_PARENT a child takes its creator's, and only a successful exec
     resets it to SIGCHLD. Checked for 200 ms, so that an unreaped child has
     exited and shows here, not only when it happens to be fast. */
  for (int i = 0; i < 20; ++i) {
    assert(waitpid(-1, &status, WNOHANG | __WALL) == 0);
    usleep(10000);
  }
  /* the descriptor set carries every open one with its flag, except the skip */
  {
    uint64_t cloexec = 0;
    int seen_pipe = 0;
    count = capstone_spawner_descriptors(fds, numbers, &cloexec, CAPSTONE_SPAWNER_FDS, s.socket);
    for (unsigned i = 0; i < count; ++i) {
      assert(fds[i] != s.socket && fds[i] == numbers[i]);
      if (fds[i] == out[0]) { seen_pipe = 1; assert((cloexec >> i) & 1); }
      if (fds[i] == 0) assert(!((cloexec >> i) & 1));
    }
    assert(seen_pipe);
  }
  /* an unterminated block is refused before any fork */
  assert(capstone_spawner_spawn(&s, block, 8, NULL, NULL, 0, 0, 0, 0) == -EINVAL);
  assert(!capstone_spawner_is_image("/bin/sh"));
  capstone_spawner_stop(&s);
  assert(s.socket == -1);
  puts("spawner-test: ok");
  return 0;
}
