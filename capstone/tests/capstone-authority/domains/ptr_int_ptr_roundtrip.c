// Authority suite: round-trip a valid pointer through an integer and back.
//
// Provenance invariant under test: laundering a pointer through an integer
// must not preserve authority. ptrtoint strips the tag; the intervening
// `volatile` integer forces the value through a plain-integer path (preventing
// the optimiser from folding inttoptr(ptrtoint(p)) back to p and restoring
// provenance). The reconstructed pointer is UNTAGGED, so the load must trap.
//
// (Note for the paper: WITHOUT the volatile barrier, LLVM may fold the
// round-trip back to the original capability and NOT trap. Both outcomes are
// fail-safe -- neither forges new authority -- but they differ; this case
// pins the integer-path behaviour.)
//
// Oracle: tag-fault.

static int g = 0x1234;

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  int *p = &g;                          // valid capability (provenance: &g)
  volatile unsigned long u = (unsigned long)p; // ptrtoint -> integer, tag dropped
  int *q = (int *)u;                    // inttoptr -> untagged scalar
  *res = (unsigned)(*q);                // load through untagged value -> tag fault
}
