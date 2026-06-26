// Authority suite: forge a capability from a raw integer, then dereference it.
//
// Provenance invariant under test: an integer must not become authority.
// `(int *)x` is an inttoptr of a plain integer -> the result is UNTAGGED.
// Dereferencing an untagged value must trap with the QEMU tag-fault diagnostic
// "Cap mem access requires capability".
//
// `volatile` keeps the integer in a register/memory so the optimiser cannot
// reason about its (absent) provenance and elide the load.
//
// Oracle: tag-fault (expected to trap today and forever).

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  volatile unsigned long x = 0x4000UL; // arbitrary address; tag check fires first
  volatile int *p = (int *)x;          // inttoptr -> untagged scalar
  *res = (unsigned)(*p);               // load through untagged value -> tag fault
}
