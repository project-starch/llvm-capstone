/* Native test of the msghdr block: a good block, and every malformed part. */
#include "capstone/msghdr.h"
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define EXCHANGE 4096
static char exchange[EXCHANGE];
static struct capstone_msghdr_view view;

static void put(uint64_t offset, const void *data, size_t bytes) {
  memcpy(exchange + offset, data, bytes);
}

int main(void) {
  struct capstone_msghdr_block b = {.name = 512, .namelen = 16, .iov = 128, .iovlen = 2,
                                    .control = 768, .controllen = 24, .flags = 0};
  uint64_t pairs[4] = {1024, 5, 2048, 3};
  put(64, &b, sizeof b);
  put(128, pairs, sizeof pairs);
  assert(!capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view));
  assert(view.block.name == 512 && view.block.iovlen == 2 && view.total == 8);
  assert(view.offsets[0] == 1024 && view.lengths[0] == 5 && view.offsets[1] == 2048 && view.lengths[1] == 3);
  /* no name, no control, no pairs: legal and empty */
  struct capstone_msghdr_block none = {0};
  put(64, &none, sizeof none);
  assert(!capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) && view.total == 0);
  /* the block itself at the edge, and beyond */
  put(EXCHANGE - 64, &none, sizeof none);
  assert(!capstone_msghdr_unpack(exchange, EXCHANGE, EXCHANGE - 64, &view));
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, EXCHANGE - 63, &view) == EFAULT);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, EXCHANGE, &view) == EFAULT);
  assert(capstone_msghdr_unpack(NULL, EXCHANGE, 64, &view) == EFAULT);
  /* a name beyond the region; a length without a name */
  struct capstone_msghdr_block bad = b;
  bad.namelen = EXCHANGE;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EFAULT);
  bad = b; bad.name = 0;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EINVAL);
  bad = b; bad.control = EXCHANGE - 8;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EFAULT);
  bad = b; bad.control = 0;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EINVAL);
  /* more pairs than the kernel takes; a pair table beyond the region; a pair
     beyond the region; a total that wraps */
  bad = b; bad.iovlen = CAPSTONE_MSGHDR_IOVS + 1;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EINVAL);
  bad = b; bad.iov = EXCHANGE - 16;
  put(64, &bad, sizeof bad);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EFAULT);
  uint64_t far[4] = {1024, 5, EXCHANGE - 2, 3};
  put(64, &b, sizeof b);
  put(128, far, sizeof far);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EFAULT);
  uint64_t huge[4] = {1024, UINT64_MAX, 2048, 3};
  put(128, huge, sizeof huge);
  assert(capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) == EFAULT);
  /* a zero-length pair anywhere inside is fine, at the region's end too */
  uint64_t empty[4] = {EXCHANGE, 0, 2048, 3};
  put(128, empty, sizeof empty);
  assert(!capstone_msghdr_unpack(exchange, EXCHANGE, 64, &view) && view.total == 3);
  puts("msghdr-test: ok");
  return 0;
}
