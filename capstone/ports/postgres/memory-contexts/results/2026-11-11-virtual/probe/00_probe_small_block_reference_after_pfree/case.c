#include "blockref.h"
/* A context whose chunk limit is small, so a 600-byte request gets a single-chunk block of under 1 KiB: malloc serves
 * it from a group of several slots, and freeing it unmaps nothing. After pfree, is the reference malloc returned for
 * that block still live? */
PG_CASE(0) {
  MemoryContext c = AllocSetContextCreateInternal(pg_root, "small-limit", 0, 1024, 1024);
  unsigned char *keep = MemoryContextAlloc(c, 600);      /* a neighbour that keeps the malloc group in use */
  unsigned char *p = MemoryContextAlloc(c, 600);
  p[0] = 7; keep[0] = 9;
  unsigned char *block = block_reference(c, p);
  (void)pg_probe(block);
  printf("PROBE block reference read before pfree\n");
  pfree(p);
  pg_held = block;
  pg_mark();
  (void)pg_probe(pg_held);
  printf("PROBE block reference STILL LIVE after pfree\n");
}
