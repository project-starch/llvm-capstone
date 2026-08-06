#ifndef LOC1_KERNEL_H
#define LOC1_KERNEL_H
/* Smallest possible form of the construct that wedges: ONE local function-pointer variable,
   assigned from a runtime-selected target, then called. No array, no struct.
   Board-measured 2026-08-06: locnp8 (local struct array, NO pointers) returns correctly, and
   locfl8 (flat local array of fn pointers, CALLED) wedges -- so neither the struct element
   type nor the string pointer is required. This asks whether the ARRAY is required either. */
static volatile unsigned loc1_gate = 1u;   /* satisfies the ldc gp[i] build gate */
static int loc1_f0(void) { return 3; }
static int loc1_f1(void) { return 5; }

static unsigned loc1_compute(void) {
  unsigned i, s = 0;
  for (i = 0; i < 8; i++) {
    int (*fp)(void) = (i & 1) ? loc1_f0 : loc1_f1;   /* local fn ptr, runtime-selected */
    s += (unsigned)fp() + i;                          /* called through the local      */
  }
  return s + loc1_gate - 1u;
}
#endif
