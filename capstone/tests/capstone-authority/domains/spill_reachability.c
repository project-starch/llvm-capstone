// Authority suite: capabilities survive a register spill (PI Q1, T2 class).
//
// Under test: holding more live pointers than there are callee-saved registers
// across a call forces some capabilities to be spilled to the stack (stc) and
// reloaded (ldc). The spill/reload must PRESERVE the tag and bounds, so the
// pointers still work after the call. This demonstrates the mechanism the PI
// asked about: spilled capabilities are stored tagged and remain usable.
//
// (The complementary security concern -- whether an attacker can READ a spill
// slot via an over-broad capability -- is the T3 granularity problem covered by
// global_oob.c, not a spill defect. Spilling itself is fail-safe here.)
//
// Oracle: ok, retval = 0x59110037 (sum of 10 spilled-and-reloaded loads == 55).

static int v[12];

__attribute__((noinline)) static void clobber(void) {
  // A call between defining the pointers and using them forces spills.
  for (int k = 0; k < 12; k++)
    v[k] += 0;
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  for (int k = 0; k < 12; k++)
    v[k] = k + 1;
  int *p0 = &v[0], *p1 = &v[1], *p2 = &v[2], *p3 = &v[3], *p4 = &v[4];
  int *p5 = &v[5], *p6 = &v[6], *p7 = &v[7], *p8 = &v[8], *p9 = &v[9];
  clobber(); // forces the ten live capabilities to be spilled across the call
  int s = *p0 + *p1 + *p2 + *p3 + *p4 + *p5 + *p6 + *p7 + *p8 + *p9; // == 55
  *res = 0x59110000u | (unsigned)s;
}
