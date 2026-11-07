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
  struct capstone_launch_task task = {4242, 1, 1000, 1000, 100, 100,
      1700000000123456789ull, 987654321ull, 55555555ull, 10000000ull, {0}};
  assert(!capstone_launch_pack(data, sizeof data, 5, input, environment, "/tmp/a b", 5, &task));
  assert(!unpack(sizeof data));
  assert(view.argc == 5 && view.envc == 2 && view.stdio_mask == 5);
  assert(!memcmp(&view.task, &task, sizeof task));
  assert(!strcmp(view.cwd, "/tmp/a b") && !args[5] && !env[2]);
  for (unsigned i = 0; i < 5; ++i) assert(!strcmp(input[i], args[i]));
  for (unsigned i = 0; i < 2; ++i) assert(!strcmp(environment[i], env[i]));
  char good[sizeof data];
  memcpy(good, data, sizeof good);
  struct capstone_launch_header h;
  memcpy(&h, data, sizeof h);
  for (size_t n = 0; n < h.bytes; ++n) assert(unpack(n));
  /* version, bytes, argc, envc, cwd, stdio_mask: each rejects UINT32_MAX. The
     task fields that follow are data, not structure, and are not validated. */
  for (unsigned field = 0; field < 6; ++field) {
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
  assert(capstone_launch_pack(data, sizeof h, 5, input, environment, "/", 7, &task) == E2BIG);
  assert(capstone_launch_pack(data, sizeof data, 0, input, environment, "/", 7, &task) == EINVAL);
  assert(capstone_launch_pack(data, sizeof data, 5, input, environment, "/", 8, &task) == EINVAL);
  char huge[CAPSTONE_LAUNCH_BYTES + 1];
  memset(huge, 'x', sizeof huge - 1); huge[sizeof huge - 1] = 0;
  char *large[] = {huge};
  assert(capstone_launch_pack(data, sizeof data, 1, large, environment, "/", 7, &task) == E2BIG);
  /* v3 shifts the offsets table by one fixed, bounded extension. */
  memcpy(data, good, sizeof data);
  memmove(data + sizeof h + 16, data + sizeof h, h.bytes - sizeof h);
  struct capstone_launch_header *v3 = (void *)data;
  v3->version = CAPSTONE_LAUNCH_NATIVE_VERSION;
  v3->bytes += 16; v3->cwd += 16;
  struct capstone_launch_native *native = (void *)(data + sizeof h);
  *native = (struct capstone_launch_native){1, CAPSTONE_LAUNCH_NATIVE_IO, 65536, 0};
  for (unsigned i = 0; i < h.argc + h.envc; ++i) {
    uint32_t off;
    memcpy(&off, data + sizeof h + 16 + i * 4, 4); off += 16;
    memcpy(data + sizeof h + 16 + i * 4, &off, 4);
  }
  assert(!unpack(sizeof data) && view.task.native.abi == 1);
  assert(view.task.native.max_io == 65536 && !strcmp(args[2], "two words"));
  for (size_t n = 0; n < v3->bytes; ++n) assert(unpack(n));
  uint32_t full_bytes = v3->bytes, full_cwd = v3->cwd, first_arg;
  memcpy(&first_arg, data + sizeof h + 16, 4);
  for (unsigned bytes = sizeof h; bytes <= sizeof h + 16 + 7 * 4; ++bytes) {
    v3->bytes = bytes; assert(unpack(sizeof data));
  }
  v3->bytes = full_bytes;
  v3->cwd = sizeof h; assert(unpack(sizeof data)); v3->cwd = full_cwd;
  uint32_t into_extension = sizeof h + 4;
  memcpy(data + sizeof h + 16, &into_extension, 4); assert(unpack(sizeof data));
  memcpy(data + sizeof h + 16, &first_arg, 4);
  native->abi = 99; assert(!unpack(sizeof data) && !view.task.native.abi);
  native->abi = 1; native->features = CAPSTONE_LAUNCH_NATIVE_IO | CAPSTONE_LAUNCH_NATIVE_LOCKS;
  assert(!unpack(sizeof data) && view.task.native.features == 3);
  native->features = CAPSTONE_LAUNCH_NATIVE_LOCKS;
  assert(!unpack(sizeof data) && !view.task.native.abi);
  native->features = 5;
  assert(!unpack(sizeof data) && view.task.native.features == 5);
  native->features = 7;
  assert(!unpack(sizeof data) && view.task.native.features == 7);
  native->features = 9;
  assert(!unpack(sizeof data) && view.task.native.features == 9);
  native->features = 15;
  assert(!unpack(sizeof data) && view.task.native.features == 15);
  native->features = 17;
  assert(!unpack(sizeof data) && view.task.native.features == 17);
  native->features = 31;
  assert(!unpack(sizeof data) && view.task.native.features == 31);
  native->features = 65;
  assert(!unpack(sizeof data) && view.task.native.features == 65);
  native->features = 95;
  assert(!unpack(sizeof data) && view.task.native.features == 95);
  native->features = 129;
  assert(!unpack(sizeof data) && view.task.native.features == 129);
  native->features = 223;
  assert(!unpack(sizeof data) && view.task.native.features == 223);
  native->features = 33;
  assert(!unpack(sizeof data) && !view.task.native.abi);
  native->features = 1; native->max_io = 1;
  assert(!unpack(sizeof data) && !view.task.native.abi);
  native->max_io = 65536; native->reserved = 1;
  assert(!unpack(sizeof data) && !view.task.native.abi);
  native->reserved = 0; assert(!unpack(sizeof data) && view.task.native.abi == 1);
  memcpy(data, good, sizeof data);
  assert(!unpack(sizeof data) && !view.task.native.abi); /* no stale v3 state */
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
