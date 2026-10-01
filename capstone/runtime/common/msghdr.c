#include "capstone/msghdr.h"
#include <errno.h>
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(struct capstone_msghdr_block) == CAPSTONE_MSGHDR_BYTES, "msghdr block ABI");

static int inside(size_t exchange_bytes, uint64_t offset, uint64_t bytes) {
  return offset <= exchange_bytes && bytes <= exchange_bytes - offset;
}

int capstone_msghdr_unpack(const char *exchange, size_t exchange_bytes, uint64_t offset,
                           struct capstone_msghdr_view *view) {
  struct capstone_msghdr_block *b;
  if (!exchange || !view || !inside(exchange_bytes, offset, CAPSTONE_MSGHDR_BYTES))
    return EFAULT;
  b = &view->block;
  memcpy(b, exchange + offset, sizeof *b);
  view->total = 0;
  if (b->name ? !inside(exchange_bytes, b->name, b->namelen) : b->namelen != 0)
    return b->name ? EFAULT : EINVAL;
  if (b->control ? !inside(exchange_bytes, b->control, b->controllen) : b->controllen != 0)
    return b->control ? EFAULT : EINVAL;
  if (b->iovlen > CAPSTONE_MSGHDR_IOVS)
    return EINVAL;
  if (b->iovlen == 0)
    return 0;
  if (!inside(exchange_bytes, b->iov, b->iovlen * 16))
    return EFAULT;
  for (uint64_t i = 0; i < b->iovlen; ++i) {
    uint64_t pair[2];
    memcpy(pair, exchange + b->iov + 16 * i, sizeof pair);
    if (!inside(exchange_bytes, pair[0], pair[1]))
      return EFAULT;
    if (pair[1] > (uint64_t)INT64_MAX - view->total)
      return EINVAL;
    view->offsets[i] = pair[0];
    view->lengths[i] = pair[1];
    view->total += (size_t)pair[1];
  }
  return 0;
}
