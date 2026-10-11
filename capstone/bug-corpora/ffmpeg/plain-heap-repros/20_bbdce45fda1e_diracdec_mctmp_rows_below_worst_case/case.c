#include "corpus.h"

/* libavcodec/diracdec.c, the motion-compensation row writes, fix bbdce45fda1e. The crossed buffer is ONE direct av_malloc-family
 * allocation; no AVBufferPool or AVRefStructPool is in the path. */

FFH_CASE(20) {
  /* Case 20 -- diracdec's mctmp, fix bbdce45fda1e. PLAIN HEAP: the scratch
   * buffer's height term is h + MAX_BLOCKSIZE, while motion compensation writes
   * rows up to blheight * ybsep + yblen.
   *
   * At the fix's parent:
   *
   *     s->mctmp = av_malloc_array((stride+MAX_BLOCKSIZE), (h+MAX_BLOCKSIZE) * sizeof(*s->mctmp));
   *
   * and the fix widens the margin:
   *
   *     s->mctmp = av_malloc_array((stride+MAX_BLOCKSIZE), (h + 5*MAX_BLOCKSIZE) * sizeof(*s->mctmp));
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const unsigned long rowbytes = 8;               /* the reduced row width */
  const unsigned long rows_alloc = fixed ? 8 : 2; /* h + MAX_BLOCKSIZE vs h + 5*MAX_BLOCKSIZE */
  const unsigned long rows_written = 8;           /* blheight * ybsep + yblen */
  const unsigned long bytes = rows_alloc * rowbytes;   /* buggy arm: 16 B, a size class */
  CHECK(rows_written > 2, 1101);                  /* the premise, asserted */

  unsigned char *mctmp = calloc((size_t)bytes, 1);
  CHECK(mctmp, 1102);

  long touched = 0;
  int crossed = 0;
  for (unsigned long y = 0; y < rows_written; y++) {
    const unsigned long at = y * rowbytes;
    if (at >= bytes) {
      touched = (long)at;
      crossed = 1;
      write_probe_u8(mctmp + at, 0xA5);           /* the labelled crossing */
      break;
    }
    memset(mctmp + at, 0x11, (size_t)rowbytes);
  }

  o->cap = bytes;
  o->touched = touched;
  o->crossed = crossed;
  o->extent = (long)(rows_written * rowbytes) - (long)bytes;
  o->damage = crossed;

  o->defect_text = "mctmp's height term is h + MAX_BLOCKSIZE while motion compensation writes rows "
                   "up to blheight * ybsep + yblen, so the later rows land past the allocation";
  o->fixed_text = "the fix allocates h + 5*MAX_BLOCKSIZE rows, covering the writer's worst case";
  free(mctmp);
}
