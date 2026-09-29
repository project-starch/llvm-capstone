#include "capstone/launch.h"
#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char data[CAPSTONE_LAUNCH_BYTES];
static char *args[CAPSTONE_LAUNCH_STRINGS + 1], *env[CAPSTONE_LAUNCH_STRINGS + 1];
static struct capstone_launch_view view;
static int unpack(size_t size) {
  return capstone_launch_unpack(data, size, args, CAPSTONE_LAUNCH_STRINGS + 1,
                                env, CAPSTONE_LAUNCH_STRINGS + 1, &view);
}

int main(void) {
  char *input[] = {"program", "", "two words", "a\nb", "'\"$()`", NULL};
  char *environment[] = {"A=", "B=a\nb", NULL};
  assert(!capstone_launch_pack(data, sizeof data, 5, input, environment, "/tmp/a b", 5));
  assert(!unpack(sizeof data));
  assert(view.argc == 5 && view.envc == 2 && view.stdio_mask == 5);
  assert(!strcmp(view.cwd, "/tmp/a b") && !args[5] && !env[2]);
  for (unsigned i = 0; i < 5; ++i) assert(!strcmp(input[i], args[i]));
  for (unsigned i = 0; i < 2; ++i) assert(!strcmp(environment[i], env[i]));
  char good[sizeof data];
  memcpy(good, data, sizeof good);
  struct capstone_launch_header h;
  memcpy(&h, data, sizeof h);
  for (size_t n = 0; n < h.bytes; ++n) assert(unpack(n));
  for (unsigned field = 0; field < 8; ++field) {
    uint32_t bad = UINT32_MAX;
    memcpy(data, good, sizeof data);
    memcpy(data + 8 + field * sizeof bad, &bad, sizeof bad);
    assert(unpack(sizeof data));
  }
  memcpy(data, good, sizeof data);
  memset(data + sizeof h, 0, sizeof(uint32_t));
  assert(unpack(sizeof data));
  memcpy(data, good, sizeof data);
  data[h.bytes - 1] = 'x';
  assert(unpack(h.bytes));
  assert(capstone_launch_pack(data, sizeof h, 5, input, environment, "/", 7) == E2BIG);
  assert(capstone_launch_pack(data, sizeof data, 0, input, environment, "/", 7) == EINVAL);
  assert(capstone_launch_pack(data, sizeof data, 5, input, environment, "/", 8) == EINVAL);
  char huge[CAPSTONE_LAUNCH_BYTES + 1];
  memset(huge, 'x', sizeof huge - 1); huge[sizeof huge - 1] = 0;
  char *large[] = {huge};
  assert(capstone_launch_pack(data, sizeof data, 1, large, environment, "/", 7) == E2BIG);
  /* Deterministic malformed-block stress: the decoder must reject or produce
     only pointers into the supplied data. Run this under ASan/UBSan as well. */
  uint32_t random = 1;
  for (unsigned round = 0; round < 10000; ++round) {
    memcpy(data, good, sizeof data);
    random = random * 1664525u + 1013904223u;
    unsigned where = random % h.bytes;
    data[where] ^= (char)(random >> 24);
    if (!unpack(h.bytes)) {
      for (unsigned i = 0; i < view.argc; ++i)
        assert(args[i] >= data && args[i] < data + h.bytes);
      for (unsigned i = 0; i < view.envc; ++i)
        assert(env[i] >= data && env[i] < data + h.bytes);
    }
  }
  puts("launch wire contract passed");
}
