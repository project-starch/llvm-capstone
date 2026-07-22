// gp-free QEMU proof domain: a real integer app with data globals AND a real
// (non-inlined) call graph, exercising scc-based global addressing + plain
// inter-function calls/returns under -capstone-gp-free.
//
// Oracle: s = sum(tbl) + sum(0..7) = 77 + 28 = 105 = 0x69  ->  retval 0x2110C069.

static const int tbl[8] = {2, 3, 5, 7, 11, 13, 17, 19}; // .rodata global
static int acc[8];                                       // .data/.bss global

__attribute__((noinline)) static int helper(int i) {
  return tbl[i] + acc[i]; // reads both globals via scc gp
}

void domain_main(unsigned *res, unsigned func) {
  (void)func;
  int s = 0;
  for (int i = 0; i < 8; i++) {
    acc[i] = i;        // store to a global via scc gp
    s += helper(i);    // plain call (gp-free), not cjalr
  }
  *res = 0x2110C000u | (unsigned)(s & 0xff);
}
