#ifndef GPN2USE1_KERNEL_H
#define GPN2USE1_KERNEL_H
/* SEPARATES "the glue mis-BUILDS entry 1" from "the domain mis-READS slot 1".
 *
 * Every silicon domain with one global passes and every one with more than one fails,
 * but two different mechanisms fit that split equally well:
 *   (a) the glue's carve loop is wrong on its second and later iterations, so the
 *       cap table is already corrupt before the domain runs; or
 *   (b) the carve is fine and the fault is in the ACCESS -- domain code reaches
 *       global i with `ldc rd, (i*16)(gp)`, and a count-1 domain only ever emits
 *       offset 0, so a mishandled nonzero immediate would be invisible until count 2.
 *
 * This rung has TWO globals in the descriptor -- so the glue runs its loop twice and
 * builds a 2-entry table, exactly as gpn2 does -- but g1 is `used` and never
 * referenced, so the compiler emits NO `ldc 16(gp)`. Every access is offset 0.
 *
 *   PASSES  -> building the second entry is fine; the fault is the nonzero ldc
 *              immediate. Fix is compiler-side, in lowerGlobalAddress.
 *   HANGS   -> the carve loop itself is broken on iteration 2. Fix is in the glue.
 *
 * Deliberately identical to gpn2 in every other respect (same shapes, same fold) so
 * the pair is a clean one-variable comparison.
 */
/* g1 has EXTERNAL linkage rather than __attribute__((used)) so it survives without
   referencing it. `used` would emit llvm.compiler.used, which is an appending-linkage
   marker rather than data -- and getGpCaptableIndex used to hand it a cap-table slot,
   which fails the link with "undefined symbol: llvm.compiler.used". That filter is
   fixed separately; this rung just avoids needing it. */
static unsigned g0[4];
unsigned gpn2use1_g1[4];                       /* in the descriptor, never accessed */
static unsigned gpn2use1_compute(void) {
  unsigned h = 2166136261u;
  for (int i=0;i<4;i++) g0[i] = (unsigned)(i + 0*7 + 1);
  for (int i=0;i<4;i++) { h ^= g0[i]; h *= 16777619u; }
  return h;
}
#endif
