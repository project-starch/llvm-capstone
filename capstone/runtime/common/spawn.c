#include "capstone/spawn.h"
#include <errno.h>
#include <string.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Capstone spawn v1 requires little-endian scalar encoding"
#endif
_Static_assert(sizeof(struct capstone_spawn_header) == 56, "spawn header ABI");
_Static_assert(sizeof(struct capstone_spawn_action) == 24, "spawn action ABI");
static const unsigned char magic[8] = {'C', 'P', 'S', 'P', 'A', 'W', 'N', '1'};

static int append(char *data, size_t capacity, size_t *used, const char *value, uint32_t *offset) {
  size_t left = capacity - *used;
  size_t len = strnlen(value, left);
  if (len == left)
    return E2BIG;
  *offset = (uint32_t)*used;
  memcpy(data + *used, value, len + 1);
  *used += len + 1;
  return 0;
}

int capstone_spawn_pack(void *buffer, size_t capacity, uint32_t flags, uint32_t pgroup,
                        const char *path, char *const argv[], char *const envp[],
                        const struct capstone_spawn_action *actions, unsigned action_count,
                        const char *const action_paths[], size_t *bytes) {
  struct capstone_spawn_header h = {0};
  unsigned argc = 0, envc = 0;
  size_t used;
  int error;
  if (!buffer || !path || !argv || !bytes || action_count > CAPSTONE_SPAWN_ACTIONS ||
      (action_count && !actions) || (flags & ~63u))
    return EINVAL;
  if (capacity > CAPSTONE_SPAWN_BYTES)
    capacity = CAPSTONE_SPAWN_BYTES;
  while (argv[argc]) {
    if (argc >= CAPSTONE_SPAWN_STRINGS)
      return E2BIG;
    ++argc;
  }
  if (envp)
    while (envp[envc]) {
      if (envc >= CAPSTONE_SPAWN_STRINGS - argc)
        return E2BIG;
      ++envc;
    }
  used = sizeof h + ((size_t)argc + envc) * sizeof(uint32_t) +
         (size_t)action_count * sizeof(struct capstone_spawn_action);
  if (used >= capacity)
    return E2BIG;
  memcpy(h.magic, magic, sizeof magic);
  h.version = CAPSTONE_SPAWN_VERSION;
  h.flags = flags;
  h.pgroup = pgroup;
  h.argc = argc;
  h.envc = envc;
  h.actions = action_count;
  if ((error = append(buffer, capacity, &used, path, &h.path)))
    return error;
  for (unsigned i = 0; i < argc + envc; ++i) {
    const char *s = i < argc ? argv[i] : envp[i - argc];
    uint32_t offset;
    if (!s)
      return EINVAL;
    if ((error = append(buffer, capacity, &used, s, &offset)))
      return error;
    memcpy((char *)buffer + sizeof h + i * sizeof offset, &offset, sizeof offset);
  }
  for (unsigned i = 0; i < action_count; ++i) {
    struct capstone_spawn_action a = actions[i];
    a.path = 0;
    if (a.cmd == CAPSTONE_SPAWN_OPEN || a.cmd == CAPSTONE_SPAWN_CHDIR) {
      if (!action_paths || !action_paths[i])
        return EINVAL;
      if ((error = append(buffer, capacity, &used, action_paths[i], &a.path)))
        return error;
    } else if (a.cmd != CAPSTONE_SPAWN_CLOSE && a.cmd != CAPSTONE_SPAWN_DUP2 &&
               a.cmd != CAPSTONE_SPAWN_FCHDIR) {
      return EINVAL;
    }
    memcpy((char *)buffer + sizeof h + (argc + envc) * sizeof(uint32_t) + i * sizeof a, &a, sizeof a);
  }
  h.bytes = (uint32_t)used;
  memcpy(buffer, &h, sizeof h);
  *bytes = used;
  return 0;
}

static const char *string_at(const char *data, size_t bytes, size_t first, uint32_t off) {
  if (off < first || off >= bytes || !memchr(data + off, 0, bytes - off))
    return NULL;
  return data + off;
}

int capstone_spawn_unpack(const void *buffer, size_t bytes, char **argv, size_t argv_slots,
                          char **envp, size_t env_slots, const char **action_paths,
                          size_t action_slots, struct capstone_spawn_view *view) {
  struct capstone_spawn_header h;
  const char *data = buffer;
  size_t first, strings;
  if (!buffer || !argv || !envp || !view || bytes < sizeof h)
    return EINVAL;
  memcpy(&h, buffer, sizeof h);
  if (memcmp(h.magic, magic, sizeof magic) || h.version != CAPSTONE_SPAWN_VERSION ||
      (h.flags & ~63u) || h.argc > CAPSTONE_SPAWN_STRINGS ||
      h.envc > CAPSTONE_SPAWN_STRINGS - h.argc || h.actions > CAPSTONE_SPAWN_ACTIONS ||
      h.argc >= argv_slots || h.envc >= env_slots || h.actions > action_slots ||
      (h.actions && !action_paths) ||
      h.bytes > CAPSTONE_SPAWN_BYTES || h.bytes > bytes)
    return EINVAL;
  first = sizeof h + ((size_t)h.argc + h.envc) * sizeof(uint32_t);
  strings = first + (size_t)h.actions * sizeof(struct capstone_spawn_action);
  if (strings >= h.bytes)
    return EINVAL;
  view->path = string_at(data, h.bytes, strings, h.path);
  if (!view->path || !view->path[0])
    return EINVAL;
  for (unsigned i = 0; i < h.argc + h.envc; ++i) {
    uint32_t off;
    const char *s;
    memcpy(&off, data + sizeof h + i * sizeof off, sizeof off);
    s = string_at(data, h.bytes, strings, off);
    if (!s || (i >= h.argc && (!strchr(s, '=') || s[0] == '=')))
      return EINVAL;
    if (i < h.argc)
      argv[i] = (char *)s;
    else
      envp[i - h.argc] = (char *)s;
  }
  argv[h.argc] = NULL;
  envp[h.envc] = NULL;
  view->action = (const struct capstone_spawn_action *)(data + first);
  for (unsigned i = 0; i < h.actions; ++i) {
    struct capstone_spawn_action a;
    memcpy(&a, data + first + i * sizeof a, sizeof a);
    if (a.cmd == CAPSTONE_SPAWN_OPEN || a.cmd == CAPSTONE_SPAWN_CHDIR) {
      const char *s = string_at(data, h.bytes, strings, a.path);
      if (!s)
        return EINVAL;
      action_paths[i] = s;
    } else if (a.cmd == CAPSTONE_SPAWN_CLOSE || a.cmd == CAPSTONE_SPAWN_DUP2 ||
               a.cmd == CAPSTONE_SPAWN_FCHDIR) {
      action_paths[i] = NULL;
    } else {
      return EINVAL;
    }
    if (a.fd > 65535 || a.srcfd > 65535)
      return EINVAL;
  }
  view->flags = h.flags;
  view->pgroup = h.pgroup;
  view->sigdefault = h.sigdefault;
  view->sigmask = h.sigmask;
  view->argc = h.argc;
  view->envc = h.envc;
  view->actions = h.actions;
  return 0;
}

void capstone_spawn_set_signals(void *buffer, uint32_t flags, uint64_t sigdefault, uint64_t sigmask) {
  struct capstone_spawn_header h;
  memcpy(&h, buffer, sizeof h);
  h.flags |= flags & (CAPSTONE_SPAWN_SETSIGDEF | CAPSTONE_SPAWN_SETSIGMASK);
  h.sigdefault = sigdefault;
  h.sigmask = sigmask;
  memcpy(buffer, &h, sizeof h);
}
