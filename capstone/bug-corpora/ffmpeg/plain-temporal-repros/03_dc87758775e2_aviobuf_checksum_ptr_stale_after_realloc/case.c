#include "corpus.h"

/* libavformat/aviobuf.c, ffio_ensure_seekback, fix dc87758775e2.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(3) {
  /* Case 3 -- ffio_ensure_seekback's buffer replacement, fix dc87758775e2. The
   * freed object is the old IO buffer; the pointer left holding its address is
   * the FIELD s->checksum_ptr, saved for a later checksum update.
   *
   * At the fix's parent the replacement is:
   *
   *     s->buf_end = buffer + (s->buf_end - s->buffer);
   *     s->buffer = buffer;
   *     s->buffer_size = buf_size;
   *
   * with nothing done about checksum_ptr. The fix records it as an offset:
   *
   *     ptrdiff_t checksum_ptr_offset = s->checksum_ptr ? s->checksum_ptr - s->buffer : -1;
   *     ...
   *     if (checksum_ptr_offset >= 0)
   *         s->checksum_ptr = s->buffer + checksum_ptr_offset;
   */
  const unsigned long n = 48;
  const unsigned long off = 12;                   /* where checksum_ptr sat in the old buffer */
  unsigned char *old = malloc((size_t)n);
  CHECK(old, 831);
  memset(old, 0x11, (size_t)n);
  volatile unsigned char *checksum_ptr = old + off;

  /* The fix's bookkeeping, taken BEFORE the buffer is replaced. */
  const long saved = fixed ? (long)off : -1;

  unsigned char *buffer = malloc((size_t)n * 4);  /* the larger replacement */
  CHECK(buffer, 832);
  free(old);                                      /* av_free(s->buffer) */
  o->freed = 1;
  if (saved >= 0)
    checksum_ptr = buffer + saved;                /* rebased across the replacement */

  unsigned char *fresh = malloc((size_t)n);       /* reuses the old buffer's chunk */
  CHECK(fresh, 833);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = read_probe(checksum_ptr);         /* update_checksum's read */
  o->aliased = o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "checksum_ptr still pointed into the IO buffer that the seekback growth freed, "
                   "so the checksum update reads storage that now belongs to another object";
  o->fixed_text = "the fix saves checksum_ptr as an offset and restores it against the new buffer";
  free(fresh);
  free(buffer);
}
