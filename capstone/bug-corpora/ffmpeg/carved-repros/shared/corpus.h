/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * CARVED corpus: FFmpeg's NESTED spatial row. Each case is an upstream fix for
 * an access that left a REGION which FFmpeg had carved out of ONE allocation by
 * pointer arithmetic -- `ubuf = buf + 18 * linesize`, `quant_cof[c] = buffer +
 * c * max_order`, VP9's `assign()` -- and landed in a NEIGHBOURING region of
 * the same allocation. The carving code is FFmpeg's own allocator one level
 * down: nobody but it knows where a region ends.
 *
 * It is the sibling of ../plain-heap-repros (the access leaves the malloc bound
 * itself), ../subobject-repros (the access leaves a struct member) and
 * ../plane-repros (one av_frame_get_buffer block carved into planes). The
 * crossing here never leaves the allocation -- each case CHECKs that the WHOLE
 * unreduced overshoot stays inside it -- which is exactly why a per-allocation
 * bound, a redzone allocator and a revoking free all see nothing.
 *
 * WHY THE PLATFORM ALLOCATOR. As in ../plain-heap-repros/shared/corpus.h: the
 * one block is the platform's calloc, so the allocation bound the arms enforce
 * is real and exact, and a silence is about the carve, not about an arena.
 *
 * THE CARVE IS ONE FUNCTION, ffc_carve(), so one build switch decides what a
 * region is. Without FFC_CARVE_BOUNDS it is `block + off`, upstream's
 * behaviour, on every arm. With it the region is a capability narrowed to
 * [block + off, block + off + len) -- the carve-bounds arms: the remedy at the
 * source, the carving code stating each region's extent, which is the only
 * place that knows it. The fixed arms run under it too, so a region length
 * that is wrong shows up as a fault on a FIXED arm rather than as a catch.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case
 * supplies its sequence inside FFC_CASE(NN) and records the first crossing
 * access with ffc_note().
 */
#ifndef FFC_CORPUS_H
#define FFC_CORPUS_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FFC_ALIGN(x, a) (((x) + (a) - 1) & ~((size_t)(a) - 1))

#define FFC_CASE(n)                                                            \
  const int ffc_case_number = (n);                                             \
  void ffc_case_run(int fixed, struct ffc_outcome *o)

/* What a case observed. Offsets are in BYTES from the region's start, because
 * the regions hold int8, int32, float and structs and the comparison is about
 * where the access went, not how many elements it was. */
struct ffc_outcome {
  int noted;              /* ffc_note ran: the case reached its access step */
  int crossed;            /* the access left its carved region */
  int contained;          /* ... and the whole unreduced overshoot stayed in the block */
  int damage;             /* the consequence the upstream report describes */
  unsigned long block;    /* bytes in the one allocation */
  unsigned long region;   /* bytes in the region the access belongs to */
  long touched;           /* offset of the first crossing byte from the region's start */
  long extent;            /* bytes past the region's end the unreduced access reaches */
  const char *defect_text, *fixed_text;
};

extern const int ffc_case_number;
void ffc_case_run(int fixed, struct ffc_outcome *o);

_Noreturn void ffc_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      ffc_fail(n);                                                             \
  } while (0)

/* A region of `block`: `block + off`, or under FFC_CARVE_BOUNDS a capability
 * narrowed to exactly [block + off, block + off + len). Every carve is printed,
 * with the bounds the platform actually granted, so a representability
 * round-up on CHERI cannot pass silently as a catch or a miss. */
void *ffc_carve(void *block, size_t off, size_t len, const char *name);

/* Record the access step: `at` is the first byte the step touches that lies
 * outside [region, region + rlen) -- or, when nothing does (the fixed arm), the
 * byte the same step touches last -- and `span` how many bytes the unreduced
 * access covers from `at`. A fixed arm whose fix returns before the access
 * calls nothing, and the driver reads that as "did not cross". */
void ffc_note(struct ffc_outcome *o, const void *block, size_t blen,
              const void *region, size_t rlen, const void *at, size_t span);

/* The crossings, labelled so a sanitiser's report or a capability fault can be
 * required to land HERE rather than merely somewhere in the program. Defined
 * ONCE in shared/driver.c, not static here, so the symbol is resolvable from
 * the image (../plain-heap-repros/shared/corpus.h explains why that matters). */
unsigned ffc_read_probe_u8(const volatile unsigned char *p);
void ffc_write_probe_u8(volatile unsigned char *p, unsigned char v);
uint32_t ffc_read_probe_u32(const volatile uint32_t *p);
void ffc_write_probe_u32(volatile uint32_t *p, uint32_t v);

#define read_probe_u8 ffc_read_probe_u8
#define write_probe_u8 ffc_write_probe_u8
#define read_probe_u32 ffc_read_probe_u32
#define write_probe_u32 ffc_write_probe_u32

#endif
