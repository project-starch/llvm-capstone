/* Stage-3 proof app for -capstone-gp-captable: one .bss global array, filled and
   summed at runtime (all access via gp[0]). Addition-only (no mul helper).
   s = sum(i+10, 0..7) = 28 + 80 = 108 = 0x6c  ->  retval 0x2110C06C. */
static int acc[8]; // .bss, index 0, size 32
void domain_main(unsigned *res, unsigned func) {
  (void)func;
  int s = 0;
  for (int i = 0; i < 8; i++) { acc[i] = i + 10; s += acc[i]; } // gp[0] store+load
  *res = 0x2110C000u | (unsigned)(s & 0xff); // 0x2110C06C = 554745964
}
