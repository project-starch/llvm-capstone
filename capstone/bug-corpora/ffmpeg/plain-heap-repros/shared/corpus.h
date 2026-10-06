/* What a case in this corpus needs, so a case.c is a complete translation unit.
 *
 * PLAIN-HEAP corpus, and deliberately a sibling of ../subobject-repros and
 * ../pool-repros rather than part of either. Those boundaries are "inside one
 * allocation" and "storage an AVBufferPool handed out"; the defects here cross
 * the `malloc` bound ITSELF -- each one runs off a buffer FFmpeg obtained
 * directly from av_malloc_array or av_calloc, with no inner layer.
 *
 * It is the third of its kind, after ../../memcached/plain-heap-repros and
 * ../../wireshark/plain-heap-repros, and exists for the same reason: the
 * inventory's not-nested spatial row was thin because the hunt had required its
 * candidates to be LIVE AT THE PIN. They need not be -- the convention is at
 * ../../memcached/allocator-repros/README.md:132-135, and most cases in this
 * tree are fix-reversals.
 *
 * WHY THE PLATFORM ALLOCATOR AND NOT THE PORT'S av_malloc. The buffer-pool
 * port's av_malloc (ports/ffmpeg/buffer-pool/src/shared/metadata-allocator.c:26)
 * is a bump/freelist carve out of ONE arena: it returns a raw interior pointer
 * and rounds every request up to 64 bytes. Routing these cases through it would
 * leave no per-allocation bound to cross and would absorb every small crossing
 * in the round-up, so a reading would be about the arena rather than about the
 * defect. These cases therefore call the platform's own calloc/malloc, exactly
 * as the two sibling plain-heap corpora do, which is what gives CHERI and
 * per-object bounds a real edge to enforce.
 *
 * The contract is ../../SCHEMA.md. shared/driver.c supplies main(); a case
 * supplies its sequence inside FFH_CASE(NN) and fills the outcome the driver
 * prints.
 */
#ifndef FFH_CORPUS_H
#define FFH_CORPUS_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FFH_CASE(n)                                                            \
  const int ffh_case_number = (n);                                             \
  void ffh_case_run(int fixed, struct ffh_outcome *o)

/* What a case observed. `crossed` is the claim: the access left the allocation.
 * `extent` is how far past it would run unreduced, because the reduction probes
 * the first crossing element and the magnitude is part of the finding. A
 * NEGATIVE `touched` is meaningful here and is why it is signed: two of these
 * defects cross BELOW the allocation's base. */
struct ffh_outcome {
  int crossed;            /* the access left the allocation */
  int damage;             /* the consequence the upstream report describes */
  unsigned long cap;      /* the allocation's size, in elements */
  long touched;           /* the element index the consumer reached */
  long extent;            /* elements past the allocation the unreduced run spans */
  const char *defect_text, *fixed_text;
};

extern const int ffh_case_number;
void ffh_case_run(int fixed, struct ffh_outcome *o);

_Noreturn void ffh_fail(unsigned code);
#define CHECK(c, n)                                                            \
  do {                                                                         \
    if (!(c))                                                                  \
      ffh_fail(n);                                                             \
  } while (0)

/* The crossings, labelled so a sanitiser's report or a capability fault can be
 * required to land HERE rather than merely somewhere in the program. */
__attribute__((noinline, used)) static float
read_probe(const volatile float *p) {
  return *p;
}
__attribute__((noinline, used)) static unsigned
read_probe_u8(const volatile unsigned char *p) {
  return *p;
}
__attribute__((noinline, used)) static void
write_probe_u8(volatile unsigned char *p, unsigned char v) {
  *p = v;
}

#endif
