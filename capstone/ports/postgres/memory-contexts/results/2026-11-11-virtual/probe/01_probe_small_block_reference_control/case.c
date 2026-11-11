#include "blockref.h"
/* The same block freed through the context's own list (reset), that is through the reference malloc returned. */
PG_CASE(1) {
  MemoryContext c = AllocSetContextCreateInternal(pg_root, "small-limit", 0, 1024, 1024);
  unsigned char *keep = MemoryContextAlloc(pg_root, 600);
  unsigned char *p = MemoryContextAlloc(c, 600);
  p[0] = 7; keep[0] = 9;
  unsigned char *block = block_reference(c, p);
  (void)pg_probe(block);
  printf("PROBE block reference read before reset\n");
  MemoryContextReset(c);
  pg_held = block;
  pg_mark();
  (void)pg_probe(pg_held);
  printf("PROBE block reference STILL LIVE after context reset\n");
}
