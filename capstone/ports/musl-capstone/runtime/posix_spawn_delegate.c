/* posix_spawn for the delegated runtime: the task forks and execs the child.
 *
 * musl's posix_spawn clones a vfork child, which a domain cannot do; its
 * posix_spawnp, popen and system all come through here, so they work as
 * soon as this does. The file actions are musl's private list (fdop.h), the
 * attribute's __fn marks posix_spawnp, and the signal attributes travel as
 * two 64-bit sets: the launcher's helper applies them before exec, where
 * Linux would have applied them in a vfork child. The pid returned is the child's real
 * Linux pid, a child of the launcher task, so waitpid and kill apply to it.
 */
#define _GNU_SOURCE
#include "capstone/spawn.h"
#include <errno.h>
#include <spawn.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include "fdop.h"
#include <capstone/lock.h>

long __capstone_delegate_spawn(const void *block, unsigned long bytes);
/* delegate.c: the one static request block and its lock, shared with execve */
extern volatile int __capstone_spawn_lock;

int posix_spawn(pid_t *restrict res, const char *restrict path,
                const posix_spawn_file_actions_t *fa,
                const posix_spawnattr_t *restrict attr,
                char *const argv[restrict], char *const envp[restrict]) {
  static char block[CAPSTONE_SPAWN_BYTES];
  struct capstone_spawn_action actions[CAPSTONE_SPAWN_ACTIONS];
  const char *paths[CAPSTONE_SPAWN_ACTIONS];
  unsigned count = 0;
  uint32_t flags = 0, pgroup = 0, sigflags = 0;
  uint64_t sigdefault = 0, sigmask = 0;
  size_t bytes;
  long pid;
  int error;
  if (attr) {
    if (attr->__flags & (POSIX_SPAWN_RESETIDS | POSIX_SPAWN_SETSCHEDPARAM | POSIX_SPAWN_SETSCHEDULER))
      return ENOTSUP;
    if (attr->__fn)
      flags |= CAPSTONE_SPAWN_SEARCH_PATH;
    if (attr->__flags & POSIX_SPAWN_SETPGROUP) {
      flags |= CAPSTONE_SPAWN_SETPGROUP;
      pgroup = (uint32_t)attr->__pgrp;
    }
    if (attr->__flags & POSIX_SPAWN_SETSID)
      flags |= CAPSTONE_SPAWN_SETSID;
    if (attr->__flags & POSIX_SPAWN_SETSIGDEF) {
      sigflags |= CAPSTONE_SPAWN_SETSIGDEF;
      memcpy(&sigdefault, &attr->__def, sizeof sigdefault);
    }
    if (attr->__flags & POSIX_SPAWN_SETSIGMASK) {
      sigflags |= CAPSTONE_SPAWN_SETSIGMASK;
      memcpy(&sigmask, &attr->__mask, sizeof sigmask);
    }
  }
  if (fa) {
    /* musl keeps the list newest-first; apply in the order they were added */
    const struct fdop *op = fa->__actions;
    while (op && op->next)
      op = op->next;
    for (; op; op = op->prev) {
      if (count >= CAPSTONE_SPAWN_ACTIONS)
        return E2BIG;
      /* fdop allocators initialize only the fields used by their command. */
      actions[count] = (struct capstone_spawn_action){.cmd = (uint32_t)op->cmd};
      if (op->cmd != FDOP_CHDIR) actions[count].fd = (uint32_t)op->fd;
      if (op->cmd == FDOP_DUP2) actions[count].srcfd = (uint32_t)op->srcfd;
      if (op->cmd == FDOP_OPEN) {
        actions[count].oflag = (uint32_t)op->oflag;
        actions[count].mode = (uint32_t)op->mode;
      }
      paths[count] = op->cmd == FDOP_OPEN || op->cmd == FDOP_CHDIR ? op->path : NULL;
      ++count;
    }
  }
  /* PATH belongs to the caller, and is independent of the child's envp.
     Resolve each candidate through the service so file actions still run in
     the child before its exec. No domain pointer crosses in this search. */
  const char *search = NULL;
  int denied = 0;
  if ((flags & CAPSTONE_SPAWN_SEARCH_PATH) && !strchr(path, '/')) {
    if (!*path) return ENOENT;
    search = getenv("PATH");
    if (!search) search = "/usr/local/bin:/bin:/usr/bin";
    if (strlen(path) > NAME_MAX) return ENAMETOOLONG;
  }
  flags &= ~CAPSTONE_SPAWN_SEARCH_PATH;
  for (;;) {
    char candidate[PATH_MAX];
    const char *target = path, *end = NULL;
    if (search) {
      end = strchr(search, ':');
      size_t length = end ? (size_t)(end - search) : strlen(search);
      if (length + strlen(path) + 2 > sizeof candidate) {
        if (!end) return denied ? EACCES : ENOENT;
        search = end + 1;
        continue;
      }
      memcpy(candidate, search, length);
      if (length) candidate[length++] = '/';
      strcpy(candidate + length, path);
      target = candidate;
    }
    /* The block is static: one request at a time, from packing to the answer. */
    capstone_lock(&__capstone_spawn_lock);
    error = capstone_spawn_pack(block, sizeof block, flags, pgroup, target, argv,
                                envp ? envp : (char *const[]){NULL}, actions, count, paths, &bytes);
    if (!error) {
      capstone_spawn_set_signals(block, sigflags, sigdefault, sigmask);
      pid = __capstone_delegate_spawn(block, bytes);
    }
    capstone_unlock(&__capstone_spawn_lock);
    if (error) return error;
    if (!search || (pid != -ENOENT && pid != -ENOTDIR && pid != -EACCES)) break;
    if (pid == -EACCES) denied = 1;
    if (!end) return denied ? EACCES : (int)-pid;
    search = end + 1;
  }
  if (pid < 0)
    return (int)-pid;
  if (res)
    *res = (pid_t)pid;
  return 0;
}
