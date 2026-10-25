/* main() for the FFmpeg carved corpus. One program per case, run twice against
 * the same binary: the control arm (fixed) first, then the buggy one.
 */
#include "corpus.h"
#if defined(FFC_CARVE_BOUNDS) && defined(__CHERI_PURE_CAPABILITY__)
#include <cheriintrin.h>
#endif
#if defined(FFC_SUBLET_CARVE) && (!defined(__CAPSTONE__) || defined(FFC_CARVE_BOUNDS))
#error "FFC_SUBLET_CARVE is the Capstone Sublet port of the carve, and the only carve switch"
#endif

_Noreturn void ffc_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* an infrastructure failure is never a verdict */
}

/* The labelled crossings, defined ONCE so the symbol resolves from the image. */
__attribute__((noinline, used)) unsigned
ffc_read_probe_u8(const volatile unsigned char *p) {
  return *p;
}
__attribute__((noinline, used)) void
ffc_write_probe_u8(volatile unsigned char *p, unsigned char v) {
  *p = v;
}
__attribute__((noinline, used)) uint32_t
ffc_read_probe_u32(const volatile uint32_t *p) {
  return *p;
}
__attribute__((noinline, used)) void
ffc_write_probe_u32(volatile uint32_t *p, uint32_t v) {
  *p = v;
}

/* An address as an integer, on every target: on CHERI purecap uintptr_t is a
 * capability and the conversion to unsigned long takes its address. */
static unsigned long addr(const void *p) { return (unsigned long)(uintptr_t)p; }

#ifdef FFC_SUBLET_CARVE
#include <capstone/capability.h>
#undef malloc
#undef calloc
#undef free
#define FFC_PIECES 32
/* One carved block at a time. The case receives only its scalar address;
 * the allocator keeps the parent and gives out bounded child lifetimes. */
static struct {
  void *allocation, *life;
  unsigned long base, end;
  int live;
  unsigned pieces;
  unsigned long lo[FFC_PIECES], hi[FFC_PIECES];
  unsigned char *alias[FFC_PIECES];
} blk;

void *ffc_block_alloc(size_t bytes, int zero) {
  CHECK(!blk.live, 961);
  void *p = zero ? calloc(1, bytes) : malloc(bytes);
  if (!p)
    return NULL;
  blk.allocation = p;
  blk.life = capstone_cap_derive(p, 0, bytes);
  blk.base = addr(p);
  blk.end = blk.base + bytes;
  blk.live = 1;
  blk.pieces = 0;
  printf("block mallocng bytes=%zu\n", bytes);
  return (void *)blk.base;
}

void ffc_block_free(void *block) {
  if (!block)
    return;
  CHECK(blk.live && addr(block) == blk.base, 965);
  free(blk.allocation); /* Ends the group and every issued region below it. */
  blk.allocation = blk.life = NULL;
  blk.live = 0;
}

void ffc_recarve(void *region) {
  unsigned long a = addr(region);
  unsigned j = 0;
  while (j < blk.pieces && !(blk.lo[j] <= a && a < blk.hi[j]))
    ++j;
  CHECK(blk.live && j < blk.pieces && blk.alias[j], 966);
  capstone_cap_revoke_child(blk.life, region);
  blk.alias[j] = NULL;
  printf("recarve region=[%lu,%lu)\n", blk.lo[j] - blk.base, blk.hi[j] - blk.base);
}

static unsigned char *sublet_region(size_t off, size_t len, const char *name,
                                    unsigned long *lo, unsigned long *hi) {
  (void)name;
  CHECK(blk.live && len && off <= blk.end - blk.base &&
        len <= blk.end - blk.base - off, 963);
  unsigned long s = blk.base + off, e = s + len;
  unsigned j;
  for (j = 0; j < blk.pieces; ++j) {
    if (s == blk.lo[j] && e == blk.hi[j])
      break;
    CHECK(e <= blk.lo[j] || s >= blk.hi[j], 964);
  }
  if (j == blk.pieces) {
    CHECK(j < FFC_PIECES, 964);
    ++blk.pieces;
    blk.lo[j] = s;
    blk.hi[j] = e;
    blk.alias[j] = NULL;
  }
  if (!blk.alias[j])
    blk.alias[j] = capstone_cap_derive(blk.life, off, len);
  *lo = off;
  *hi = off + len;
  return blk.alias[j];
}
#endif

void *ffc_carve(void *block, size_t off, size_t len, const char *name) {
#ifdef FFC_SUBLET_CARVE
  /* No arithmetic on `block` here: it is an untagged address, and offsetting an untagged
   * capability faults on Capstone (cause 24) -- found by the carve control, 2026-10-09. */
  unsigned long lo, hi;
  CHECK(addr(block) == blk.base, 962);
  unsigned char *p = sublet_region(off, len, name, &lo, &hi);
  printf("carve %s off=%zu len=%zu bounds=%llu base_eq=%d region=[%lu,%lu)\n", name, off, len,
         (unsigned long long)(__builtin_capstone_cap_get_end(p) - __builtin_capstone_cap_get_base(p)),
         __builtin_capstone_cap_get_base(p) == blk.base + off, lo, hi);
  return p;
#else
  unsigned char *p = (unsigned char *)block + off;
#if defined(FFC_CARVE_BOUNDS) && defined(__CHERI_PURE_CAPABILITY__)
  p = cheri_bounds_set(p, len);
  printf("carve %s off=%zu len=%zu bounds=%zu base_eq=%d\n", name, off, len,
         (size_t)cheri_length_get(p), (unsigned long)cheri_base_get(p) == addr(p));
#elif defined(FFC_CARVE_BOUNDS) && defined(__CAPSTONE__)
  uint64_t base = __builtin_capstone_cap_get_cursor(p);
  p = __builtin_capstone_cap_shrink(p, base, base + len);
  printf("carve %s off=%zu len=%zu bounds=%llu base_eq=%d\n", name, off, len,
         (unsigned long long)(__builtin_capstone_cap_get_end(p) -
                              __builtin_capstone_cap_get_base(p)),
         __builtin_capstone_cap_get_base(p) == base);
#elif defined(FFC_CARVE_BOUNDS)
#error "FFC_CARVE_BOUNDS needs a capability target: CHERI purecap or Capstone"
#else
  printf("carve %s off=%zu len=%zu bounds=allocation\n", name, off, len);
#endif
  return p;
#endif
}

void ffc_note(struct ffc_outcome *o, const void *block, size_t blen,
              const void *region, size_t rlen, const void *at, size_t span) {
  unsigned long b = addr(block), r = addr(region), a = addr(at);
  unsigned long reach = a + span; /* one past the last byte the step covers */
  o->noted = 1;
  o->block = blen;
  o->region = rlen;
  o->touched = (long)(a - r);
  o->crossed = a < r || a >= r + rlen;
  o->extent = reach > r + rlen ? (long)(reach - (r + rlen)) : 0;
  o->contained = a >= b && reach <= b + blen;
}

int main(int argc, char **argv) {
  int fixed = argc > 1 && !strcmp(argv[1], "fixed");
  if (argc > 2 && atoi(argv[2]) != ffc_case_number) {
    fprintf(stderr, "CONTROL-FAILED fixture is case %d, run asked for %s\n",
            ffc_case_number, argv[2]);
    return 75;
  }
  struct ffc_outcome o = {0};
#ifdef FFC_CARVE_BOUNDS
  const char *carve = "bounded";
#elif defined(FFC_SUBLET_CARVE)
  const char *carve = "sublet";
#else
  const char *carve = "pointer";
#endif
  printf("case=%d arm=%s carve=%s\n", ffc_case_number, fixed ? "fixed" : "buggy", carve);
  ffc_case_run(fixed, &o);
  printf("block=%lu region=%lu touched=%ld extent=%ld noted=%d crossed=%d contained=%d damage=%d\n",
         o.block, o.region, o.touched, o.extent, o.noted, o.crossed, o.contained, o.damage);
  /* A crossing that leaves the allocation is not this corpus's defect: the
   * reduction is wrong, and it would be read as a catch by every bound. */
  if (o.crossed && !o.contained) {
    fprintf(stderr, "CONTROL-FAILED the overshoot leaves the allocation: not a nested case\n");
    return 75;
  }
  if (!fixed && o.crossed)
    printf("VERDICT DEFECT-REPRODUCED %s\n", o.defect_text);
  else if (fixed && !o.crossed)
    printf("VERDICT FIXED %s\n", o.fixed_text);
  else
    printf("VERDICT INCONCLUSIVE\n");
  return fixed ? !!o.crossed : !o.crossed;
}
