#ifndef GPN2USE0_KERNEL_H
#define GPN2USE0_KERNEL_H
/* Mirror of gpn2use1: two globals in the descriptor, only ONE accessed -- but this
   time the accessed one lands at cap-table index 0 instead of index 1. Together the
   pair separates three things the count>1 failure could be:
     both pass          -> a single-slot access is always fine; the fault needs TWO
                           distinct slots live in one domain.
     use0 pass/use1 fail-> slot 1 itself is wrong, i.e. the carve loop's second
                           iteration or the second table store.
     both fail          -> merely BUILDING a 2-entry table breaks the domain, and the
                           access pattern is irrelevant.
   Both globals have external linkage so neither can be optimised away; which one gets
   index 0 is decided by module emission order and is verified by disassembly, not
   assumed. */
unsigned gpn2use0_a[4];
unsigned gpn2use0_b[4];
static unsigned gpn2use0_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<4;i++) gpn2use0_a[i] = (unsigned)(i + 1);
  for (int i=0;i<4;i++) { h ^= gpn2use0_a[i]; h *= 16777619u; }
  return h;
}
#endif
