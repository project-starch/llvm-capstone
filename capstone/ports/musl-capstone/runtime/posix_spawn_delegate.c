/* posix_spawn for the delegated runtime: the task forks and execs the child.
 *
 * musl's posix_spawn clones a vfork child, which a domain cannot do; its
 * posix_spawnp, popen and system all come through here, so they work as
 * soon as this does. The file actions are musl's private list (fdop.h), the
 * attribute's __fn marks posix_spawnp, and the signal fields are not carried:
 * a domain has no signals to reset. The pid returned is the child's real
 * Linux pid, a child of the launcher task, so waitpid and kill apply to it.
 */
#define _GNU_SOURCE
#include "capstone/spawn.h"
#include <errno.h>
#include <spawn.h>
#include <string.h>
#include "fdop.h"

long __capstone_delegate_spawn(const void *block, unsigned long bytes);

int posix_spawn(pid_t *restrict res, const char *restrict path,
                const posix_spawn_file_actions_t *fa,
                const posix_spawnattr_t *restrict attr,
                char *const argv[restrict], char *const envp[restrict]) {
  static char block[CAPSTONE_SPAWN_BYTES];
  struct capstone_spawn_action actions[CAPSTONE_SPAWN_ACTIONS];
  const char *paths[CAPSTONE_SPAWN_ACTIONS];
  unsigned count = 0;
  uint32_t flags = 0, pgroup = 0;
  size_t bytes;
  long pid;
  int error;
  if (attr) {
    if (attr->__fn)
      flags |= CAPSTONE_SPAWN_SEARCH_PATH;
    if (attr->__flags & POSIX_SPAWN_SETPGROUP) {
      flags |= CAPSTONE_SPAWN_SETPGROUP;
      pgroup = (uint32_t)attr->__pgrp;
    }
    if (attr->__flags & POSIX_SPAWN_SETSID)
      flags |= CAPSTONE_SPAWN_SETSID;
  }
  if (fa) {
    /* musl keeps the list newest-first; apply in the order they were added */
    const struct fdop *op = fa->__actions;
    while (op && op->next)
      op = op->next;
    for (; op; op = op->prev) {
      if (count >= CAPSTONE_SPAWN_ACTIONS)
        return E2BIG;
      actions[count] = (struct capstone_spawn_action){(uint32_t)op->cmd, (uint32_t)op->fd,
                                                      (uint32_t)op->srcfd, (uint32_t)op->oflag,
                                                      (uint32_t)op->mode, 0};
      paths[count] = op->cmd == FDOP_OPEN || op->cmd == FDOP_CHDIR ? op->path : NULL;
      ++count;
    }
  }
  error = capstone_spawn_pack(block, sizeof block, flags, pgroup, path, argv,
                              envp ? envp : (char *const[]){NULL}, actions, count, paths, &bytes);
  if (error)
    return error;
  pid = __capstone_delegate_spawn(block, bytes);
  if (pid < 0)
    return (int)-pid;
  if (res)
    *res = (pid_t)pid;
  return 0;
}
