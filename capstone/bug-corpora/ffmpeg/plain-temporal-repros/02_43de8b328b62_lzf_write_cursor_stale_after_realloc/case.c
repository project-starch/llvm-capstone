#include "corpus.h"

/* libavcodec/lzf.c, ff_lzf_uncompress, fix 43de8b328b62.
 * The object is ONE direct av_malloc-family allocation -- no AVBufferPool and no
 * AVRefStructPool -- which is what puts this row in the plain-temporal corpus
 * rather than in ../pool-repros. */

FFT_CASE(2) {
  /* Case 2 -- ff_lzf_uncompress's write cursor, fix 43de8b328b62. The lifetime
   * ender is a REALLOC: growing the buffer frees the old block, and the cursor
   * `p` still points into it.
   *
   * At the fix's parent:
   *
   *     ret = av_reallocp(buf, *size);
   *     if (ret < 0)
   *         return ret;
   *     }
   *
   *     bytestream2_get_buffer(gb, p, s);
   *
   * and the fix rebases the cursor:
   *
   *     p = *buf + len;
   *
   * THIS ONE WRITES through the stale cursor. */
  const unsigned long n = 48;
  const unsigned long len = 8;                    /* how far the cursor had advanced */
  unsigned char *buf = malloc((size_t)n);
  CHECK(buf, 821);
  memset(buf, 0x11, (size_t)n);
  /* A blocker immediately after buf, so buf is NOT at the top of the heap and
   * realloc CANNOT grow it in place. Without this the realloc extends the chunk,
   * nothing is freed, and the cursor stays valid -- the case then reports
   * INCONCLUSIVE for a reason that has nothing to do with the defect. The
   * reduction has to CREATE the triggering condition, not merely contain it. */
  unsigned char *blocker = malloc((size_t)n);
  CHECK(blocker, 822);
  unsigned char *old_base = buf;
  volatile unsigned char *p = buf + len;          /* the cursor into the old block */

  unsigned char *grown = realloc(buf, (size_t)n * 64);
  CHECK(grown, 823);
  CHECK(grown != old_base, 824);   /* the premise: the block really moved and was freed */
  o->freed = 1;                                   /* the old block's lifetime ended */
  if (fixed)
    p = grown + len;                              /* the fix's `p = *buf + len` */

  unsigned char *fresh = malloc((size_t)n);       /* reuses the old block's chunk */
  CHECK(fresh, 823);
  memset(fresh, 0xAA, (size_t)n);

  /* bytestream2_get_buffer's write, reduced to one byte at the cursor. */
  write_probe(p, (unsigned char)0x5A);            /* the labelled access */
  o->bytes = n;
  o->marker = 0xAA;
  o->observed = fresh[len];                       /* did the write land in the NEW object? */
  o->aliased = o->observed == 0x5A;
  o->damage = o->aliased;
  o->defect_text = "the write cursor was not rebased after av_reallocp moved the buffer, so the "
                   "copy writes into storage that now belongs to another object";
  o->fixed_text = "the fix rebases the cursor to *buf + len, so the write lands in the grown buffer";
  free(fresh);
  free(grown);
  free(blocker);
}
