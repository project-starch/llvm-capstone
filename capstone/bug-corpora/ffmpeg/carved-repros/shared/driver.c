/* main() for the FFmpeg carved corpus. One program per case, run twice against
 * the same binary: the control arm (fixed) first, then the buggy one.
 */
#include "corpus.h"
#if defined(FFC_CARVE_BOUNDS) && defined(__CHERI_PURE_CAPABILITY__)
#include <cheriintrin.h>
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

void *ffc_carve(void *block, size_t off, size_t len, const char *name) {
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
