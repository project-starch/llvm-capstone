/* Probe geometry for the silicon prediction table (docs/plans/2026-10-15-bug-corpus-on-fpga.md).
 *
 * Built only with -DCORPUS_GEOMETRY (and -I to this directory); a default build never includes this file, and
 * each driver defines CORPUS_GEOM as ((void)0) instead, so default images are byte-identical with or without it.
 *
 * Just before a labelled probe's access, prints the probe's capability as the virtual profile holds it:
 *   GEOM probe=<name> type=<LCC 1> base=<LCC 3> end=<LCC 4> cursor=<address> width=<bytes>
 * Selector 1 (type) first: 7 is an untagged value, whose bounds are not read (selectors 3 and 4 would fault).
 * Selector 0 (validity) is never used: on QEMU it is a stub (ISSUES Q-11). Under exact bounds, [base, end) is the
 * allocation the probe's pointer came from; tools/predict-silicon.py turns it into the silicon bounds. */
#ifndef CORPUS_GEOMETRY_H
#define CORPUS_GEOMETRY_H
#include <stdio.h>

static inline void corpus_geometry(const char *probe, const volatile void *p, unsigned long width) {
  unsigned long type, base = 0, end = 0;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x1" : "=r"(type) : "r"(p));
  if (type != 7) {
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x3" : "=r"(base) : "r"(p));
    __asm__ volatile(".insn r 0x5b, 0x1, 0x04, %0, %1, x4" : "=r"(end) : "r"(p));
  }
  printf("GEOM probe=%s type=%lu base=%#lx end=%#lx cursor=%#lx width=%lu\n", probe, type, base, end,
         (unsigned long)p, width);
  fflush(stdout);
}
#define CORPUS_GEOM(probe, p) corpus_geometry((probe), (p), sizeof *(p))
#endif
