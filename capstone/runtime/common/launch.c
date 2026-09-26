#include "capstone/launch.h"
#include <errno.h>
#include <string.h>

#if __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "Capstone launch v1 requires little-endian scalar encoding"
#endif
_Static_assert(sizeof(struct capstone_launch_header) == 40, "launch header ABI");
static const unsigned char magic[8] = {'C', 'P', 'L', 'A', 'U', 'N', 'C', 'H'};

static int append(char *data, size_t capacity, size_t *used,
                  const char *value, uint32_t *offset) {
  size_t left = capacity - *used;
  size_t len = strnlen(value, left);
  if (len == left)
    return E2BIG;
  *offset = (uint32_t)*used;
  memcpy(data + *used, value, len + 1);
  *used += len + 1;
  return 0;
}

int capstone_launch_pack(void *buffer, size_t capacity, int argc,
                         char *const argv[], char *const envp[],
                         const char *cwd, unsigned stdio_mask) {
  if (!buffer || !argv || !envp || !cwd || argc < 1 ||
      (unsigned)argc > CAPSTONE_LAUNCH_STRINGS || (stdio_mask & ~7u))
    return EINVAL;
  if (capacity > CAPSTONE_LAUNCH_BYTES)
    capacity = CAPSTONE_LAUNCH_BYTES;
  unsigned envc = 0;
  while (envp[envc]) {
    if (envc >= CAPSTONE_LAUNCH_STRINGS - (unsigned)argc)
      return E2BIG;
    ++envc;
  }
  size_t used = sizeof(struct capstone_launch_header) +
                ((unsigned)argc + envc) * sizeof(uint32_t);
  if (used >= capacity)
    return E2BIG;
  struct capstone_launch_header h = {0};
  memcpy(h.magic, magic, sizeof magic);
  h.version = CAPSTONE_LAUNCH_VERSION;
  h.argc = (unsigned)argc;
  h.envc = envc;
  h.stdio_mask = stdio_mask;
  int error = append(buffer, capacity, &used, cwd, &h.cwd);
  if (error)
    return error;
  for (unsigned i = 0; i < (unsigned)argc + envc; ++i) {
    const char *s = i < (unsigned)argc ? argv[i] : envp[i - (unsigned)argc];
    uint32_t offset;
    if (!s)
      return EINVAL;
    error = append(buffer, capacity, &used, s, &offset);
    if (error)
      return error;
    memcpy((char *)buffer + sizeof h + i * sizeof offset, &offset, sizeof offset);
  }
  h.bytes = (uint32_t)used;
  memcpy(buffer, &h, sizeof h);
  return 0;
}

static char *string_at(char *data, size_t bytes, size_t first, uint32_t off) {
  if (off < first || off >= bytes || !memchr(data + off, 0, bytes - off))
    return NULL;
  return data + off;
}

int capstone_launch_unpack(void *buffer, size_t capacity,
                           char **argv, size_t argv_slots,
                           char **envp, size_t env_slots,
                           struct capstone_launch_view *view) {
  struct capstone_launch_header h;
  if (!buffer || !argv || !envp || !view || capacity < sizeof h)
    return EINVAL;
  memcpy(&h, buffer, sizeof h);
  if (memcmp(h.magic, magic, sizeof magic) ||
      h.version != CAPSTONE_LAUNCH_VERSION || h.reserved0 || h.reserved1 ||
      (h.stdio_mask & ~7u) || !h.argc ||
      h.argc > CAPSTONE_LAUNCH_STRINGS ||
      h.envc > CAPSTONE_LAUNCH_STRINGS - h.argc ||
      h.argc >= argv_slots || h.envc >= env_slots ||
      h.bytes > CAPSTONE_LAUNCH_BYTES || h.bytes > capacity)
    return EINVAL;
  size_t first = sizeof h + (h.argc + h.envc) * sizeof(uint32_t);
  if (first >= h.bytes)
    return EINVAL;
  char *cwd = string_at(buffer, h.bytes, first, h.cwd);
  if (!cwd || cwd[0] != '/')
    return EINVAL;
  /* Validate the whole block before publishing any pointers. */
  for (unsigned i = 0; i < h.argc + h.envc; ++i) {
    uint32_t off;
    memcpy(&off, (char *)buffer + sizeof h + i * sizeof off, sizeof off);
    char *s = string_at(buffer, h.bytes, first, off);
    if (!s || (i >= h.argc && (!strchr(s, '=') || s[0] == '=')))
      return EINVAL;
  }
  for (unsigned i = 0; i < h.argc + h.envc; ++i) {
    uint32_t off;
    memcpy(&off, (char *)buffer + sizeof h + i * sizeof off, sizeof off);
    char *s = (char *)buffer + off;
    if (i < h.argc)
      argv[i] = s;
    else
      envp[i - h.argc] = s;
  }
  argv[h.argc] = NULL;
  envp[h.envc] = NULL;
  *view = (struct capstone_launch_view){h.argc, h.envc, h.stdio_mask, cwd};
  return 0;
}
