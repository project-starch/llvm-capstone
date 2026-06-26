// Authority suite: out-of-bounds read that runs PAST the loadable segment.
//
// Companion to global_oob.c. Here `a` is the only/last object in its segment,
// so a[100] runs past the segment end. The capability `a` inherits is bounded
// to the segment, so this over-read IS caught today.
//
// This documents the coarse spatial protection that already exists: capabilities
// are segment-bounded, so cross-segment over-reads trap even without object
// SHRINK. (Contrast global_oob.c, which stays inside the segment and does NOT
// trap today.)
//
// Oracle: bounds-fault (traps today and after Step-3 SHRINK).

static unsigned char a[64];

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned idx = 100;          // a[100]: past the end of a's segment
  *res = 0x00B00000u | (unsigned)a[idx]; // over-read past segment -> bounds fault
}
