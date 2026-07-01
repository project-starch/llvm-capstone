// Authority suite: untagged 128-bit round-trip through ldc/stc.
//
// Under test: an *untagged* capability-sized memory word (16 bytes of PLAIN
// data, no tag) must survive an ldc/stc round-trip bit-exact, including its
// high 8 bytes. This is the QEMU untagged-ldc/stc fix (SQLite gaps 3/4 root
// cause): before it, an untagged stc wrote hi = 0, zeroing the high half of
// every non-capability 16-byte chunk, so a capability-sized memcpy silently
// corrupted plain data.
//
// The copy `*(void**)&dst = *(void*const*)&src` lowers to ldc (load 16 bytes
// from src) then stc (store 16 bytes to dst); src holds plain data, so both
// take the untagged path. We then read the two halves back as scalars and
// check both survived.
//
// Oracle: ok, retval = 0x22990003 (bit 0 = low half survived, bit 1 = high
// half survived; both set == pass). A regression that loses the high half
// yields 0x22990001 and fails the oracle.

#include <stdint.h>

static uint64_t src[2] __attribute__((aligned(16)));
static uint64_t dst[2] __attribute__((aligned(16)));

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  src[0] = 0x0123456789abcdefULL;
  src[1] = 0xfedcba9876543210ULL; // non-zero high half — the bits that were lost
  dst[0] = 0;
  dst[1] = 0;

  // Capability-sized copy: ldc from src, stc to dst (untagged path).
  void *tmp = *(void *const *)&src;
  *(void **)&dst = tmp;

  unsigned lo_ok = (dst[0] == 0x0123456789abcdefULL) ? 1u : 0u;
  unsigned hi_ok = (dst[1] == 0xfedcba9876543210ULL) ? 2u : 0u;
  *res = 0x22990000u | lo_ok | hi_ok;
}
