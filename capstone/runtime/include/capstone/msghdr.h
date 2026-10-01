/* sendmsg and recvmsg on the wire: struct msghdr holds pointers, so the libc
 * flattens it into one 64-byte block in the exchange region with offsets in
 * place of pointers, the way readv's iovec array crosses. The launcher rebuilds
 * a msghdr over its bounce buffer. The block is the second argument of the
 * sendmsg and recvmsg rows, a fixed 64-byte buffer; recvmsg writes namelen,
 * controllen and flags back into it. Little-endian 64-bit words. */
#ifndef CAPSTONE_MSGHDR_H
#define CAPSTONE_MSGHDR_H

#include <stddef.h>
#include <stdint.h>

#define CAPSTONE_MSGHDR_BYTES 64u
#define CAPSTONE_MSGHDR_IOVS 1024u  /* the kernel's UIO_MAXIOV */

struct capstone_msghdr_block {
  uint64_t name, namelen;       /* exchange offset of the address, 0 for none; its bytes */
  uint64_t iov, iovlen;         /* exchange offset of iovlen pairs {offset, bytes}; 0 pairs: none */
  uint64_t control, controllen; /* exchange offset of the control bytes, 0 for none; its bytes */
  uint64_t flags;               /* msg_flags: written by recvmsg, ignored by sendmsg */
  uint64_t reserved;
};

struct capstone_msghdr_view {
  struct capstone_msghdr_block block;
  uint64_t offsets[CAPSTONE_MSGHDR_IOVS], lengths[CAPSTONE_MSGHDR_IOVS];
  size_t total;                 /* bytes over every pair */
};

/* Read the block at `offset` and check every part against the region: 0, or
 * EFAULT for a part outside it, EINVAL for more pairs than the kernel takes,
 * a total that does not fit a ssize_t, or a length without its buffer. The
 * block and the pairs are copied once; the caller uses the view, never the
 * region again. */
int capstone_msghdr_unpack(const char *exchange, size_t exchange_bytes, uint64_t offset,
                           struct capstone_msghdr_view *view);

#endif
