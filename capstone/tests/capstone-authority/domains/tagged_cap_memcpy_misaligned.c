// Authority suite: a TAGGED capability CANNOT survive a memcpy when source and
// destination do NOT share 16-byte alignment — and this is a fundamental limit,
// not a memcpy bug. A capability is only tag-representable at a 16-byte-aligned
// address, so a copy that lands it at a misaligned destination cannot carry the
// tag; memcpy correctly falls back to a plain byte loop there. The copied
// pointer therefore comes back untagged and its deref tag-faults.
//
// This records the boundary of memcpy tag-preservation and, paired with
// tagged_cap_memcpy_aligned, tells us that if SQLite loses a tag across a copy,
// the question to ask is whether its allocator hands back mismatched-alignment
// buffers (an allocator-alignment issue) rather than a memcpy defect.
//
// Oracle: tag-fault (deref of the untagged, misaligned-copied pointer).

typedef __SIZE_TYPE__ bsize_t;
#include "../../../benchmarks/beebs/adapted/beebs_freestanding_string.c"

static long backing[8];
static char src_area[48] __attribute__((aligned(16)));
static char dst_area[48] __attribute__((aligned(16)));

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  *(void **)src_area = &backing[3]; // tagged cap at a 16-aligned source slot

  volatile bsize_t n = 16;
  // Destination offset by 8: src/dst alignments differ -> byte-loop fallback,
  // tag necessarily dropped.
  memcpy(dst_area + 8, src_area, n);

  long *p = (long *)*(void **)(dst_area + 8); // untagged copy of the pointer
  *p = 0x1234;                        // tag-faults here
  *res = 0x22AB0000u;                 // unreachable if the deref faults
}
