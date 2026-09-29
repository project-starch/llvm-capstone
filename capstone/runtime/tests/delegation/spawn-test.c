/* Native test of the spawn block: pack, unpack, every malformed shape. */
#include "capstone/spawn.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>

static char block[CAPSTONE_SPAWN_BYTES];
static char *argv_out[CAPSTONE_SPAWN_STRINGS + 1], *envp_out[CAPSTONE_SPAWN_STRINGS + 1];
static const char *paths[CAPSTONE_SPAWN_ACTIONS];
static struct capstone_spawn_view view;

static int unpack(size_t bytes) {
  return capstone_spawn_unpack(block, bytes, argv_out, CAPSTONE_SPAWN_STRINGS + 1,
                               envp_out, CAPSTONE_SPAWN_STRINGS + 1, paths,
                               CAPSTONE_SPAWN_ACTIONS, &view);
}

int main(void) {
  char *argv[] = {"sh", "-c", "echo hi", "", "a\nb", NULL};
  char *envp[] = {"PATH=/bin", "X=", NULL};
  struct capstone_spawn_action actions[] = {
    {CAPSTONE_SPAWN_DUP2, 1, 5, 0, 0, 0},
    {CAPSTONE_SPAWN_CLOSE, 5, 0, 0, 0, 0},
    {CAPSTONE_SPAWN_OPEN, 0, 0, O_RDONLY, 0, 0},
    {CAPSTONE_SPAWN_CHDIR, 0, 0, 0, 0, 0},
  };
  const char *action_paths[] = {NULL, NULL, "/dev/null", "/tmp"};
  size_t bytes = 0;
  assert(!capstone_spawn_pack(block, sizeof block, CAPSTONE_SPAWN_SEARCH_PATH, 0, "/bin/sh",
                              argv, envp, actions, 4, action_paths, &bytes));
  assert(bytes > 40 && !unpack(bytes));
  assert(view.flags == CAPSTONE_SPAWN_SEARCH_PATH && view.argc == 5 && view.envc == 2 &&
         view.actions == 4);
  assert(!strcmp(view.path, "/bin/sh"));
  for (unsigned i = 0; i < 5; ++i) assert(!strcmp(argv[i], argv_out[i]));
  assert(!argv_out[5] && !envp_out[2] && !strcmp(envp_out[1], "X="));
  assert(view.action[0].cmd == CAPSTONE_SPAWN_DUP2 && view.action[0].fd == 1 &&
         view.action[0].srcfd == 5);
  assert(!paths[0] && !paths[1] && !strcmp(paths[2], "/dev/null") && !strcmp(paths[3], "/tmp"));
  assert(view.action[2].oflag == O_RDONLY);
  /* every truncation is refused */
  for (size_t n = 0; n < bytes; ++n) assert(unpack(n) == EINVAL);
  /* a corrupted offset, an unknown action, an unterminated string */
  char good[sizeof block];
  memcpy(good, block, sizeof good);
  {
    struct capstone_spawn_header h;
    memcpy(&h, block, sizeof h);
    h.path = h.bytes;
    memcpy(block, &h, sizeof h);
    assert(unpack(bytes) == EINVAL);
    memcpy(block, good, sizeof good);
    struct capstone_spawn_action a;
    size_t at = sizeof h + 7 * sizeof(uint32_t);
    memcpy(&a, block + at, sizeof a);
    a.cmd = 9;
    memcpy(block + at, &a, sizeof a);
    assert(unpack(bytes) == EINVAL);
    memcpy(block, good, sizeof good);
    block[bytes - 1] = 'x';
    assert(unpack(bytes) == EINVAL);
    memcpy(block, good, sizeof good);
  }
  /* pack refuses what it must */
  assert(capstone_spawn_pack(block, sizeof block, 0, 0, "/bin/sh", argv, envp, actions, 4,
                             NULL, &bytes) == EINVAL);            /* open without a path */
  assert(capstone_spawn_pack(block, sizeof block, 0x100, 0, "/bin/sh", argv, envp, NULL, 0,
                             NULL, &bytes) == EINVAL);            /* unknown flag */
  assert(capstone_spawn_pack(block, 64, 0, 0, "/bin/sh", argv, envp, NULL, 0, NULL,
                             &bytes) == E2BIG);                   /* no room */
  /* no environment and no actions is a valid request */
  assert(!capstone_spawn_pack(block, sizeof block, 0, 0, "/bin/true", (char *[]){"true", NULL},
                              NULL, NULL, 0, NULL, &bytes));
  assert(!unpack(bytes) && view.envc == 0 && view.actions == 0 && !envp_out[0]);
  puts("spawn-test: ok");
  return 0;
}
