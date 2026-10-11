#include "corpus.h"

/* libavcodec/jpeg2000dec.c, get_ppt, fix 4b2248594c7f.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(4) {
  /* Case 4 -- get_ppt's packed-headers reader, fix 4b2248594c7f. The freed
   * object is the old packed_headers block; what is left holding its address is
   * a GetByteContext -- a reader struct of pointers INTO that block.
   *
   * The fix adds, right after the realloc:
   *
   *     memset(&tile->packed_headers_stream, 0, sizeof(tile->packed_headers_stream));
   *
   * so the context cannot be used until it is re-created against the new base.
   * A context carrying the old base passes its OWN bounds checks while reading
   * freed storage, which is what makes this one quiet. */
  const unsigned long n = 48;
  unsigned char *headers = malloc((size_t)n);
  CHECK(headers, 841);
  memset(headers, 0x11, (size_t)n);
  /* A blocker right after it, so realloc cannot grow the chunk in place. Without
   * this nothing is freed and the stale base stays valid. */
  unsigned char *blocker = malloc((size_t)n);
  CHECK(blocker, 842);
  unsigned char *old_base = headers;

  /* The reader context, reduced to the base pointer it caches. */
  struct { const volatile unsigned char *buffer; unsigned long size; } stream;
  stream.buffer = headers;
  stream.size = n;

  unsigned char *grown = realloc(headers, (size_t)n * 64);
  CHECK(grown, 843);
  CHECK(grown != old_base, 844);   /* the premise: the block really moved and was freed */
  o->freed = 1;
  if (fixed)
    memset(&stream, 0, sizeof stream);            /* the fix's memset */

  unsigned char *fresh = malloc((size_t)n);       /* reuses the old block's chunk */
  CHECK(fresh, 843);
  memset(fresh, 0xAA, (size_t)n);

  o->bytes = n;
  o->marker = 0xAA;
  o->observed = stream.buffer ? read_probe(stream.buffer) : 0u;   /* the labelled access */
  o->aliased = stream.buffer && o->observed == o->marker;
  o->damage = o->aliased;
  o->defect_text = "the packed-headers reader kept the base of the block av_realloc freed, so it "
                   "reads storage that now belongs to another object while passing its own bounds "
                   "checks";
  o->fixed_text = "the fix zeroes the reader context, so it must be re-created against the new base";
  free(fresh);
  free(grown);
  free(blocker);
}
