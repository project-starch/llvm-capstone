/* mruby Array#sort! use-after-realloc, upstream fix eb398971b (issue #6649). Standalone
 * reduction of the allocator shape; the measured multi-platform instrument is ../probe.c
 * (CheriBSD) and ../probe-capstone.c (Capstone virtual), scenario mruby-sort-bang-shrink.
 * See PROVENANCE.md for the upstream hunk. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
  const unsigned long elems = 64, shrunk = 32, old_index = 63; /* mrb_value ~ 8 bytes */
  unsigned char *base = malloc(elems * 8);
  if (!base) return 2;
  memset(base, 0x5a, elems * 8);
  unsigned char *captured = base;                     /* sort_cmp's p/a */
  unsigned char *after = realloc(base, shrunk * 8);   /* ary_shrink_capa */
  if (!after) return 2;
  /* pre-fix: the pointer-equality guard misses a same-address shrink, so the sort reads the
   * captured base at the old index, in the trimmed tail. */
  volatile unsigned char *p = captured + old_index * 8;
  unsigned v = *p;                                    /* stale read; faults under revocation */
  printf("same_addr=%d stale_read=0x%02x\n", after == captured, v);
  free(after);
  return 0;
}
