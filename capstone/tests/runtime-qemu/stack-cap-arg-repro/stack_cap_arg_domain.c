/*
 * Minimal reproducer: capability (pointer) arguments passed on the STACK are
 * delivered untagged on the Capstone domain.
 *
 * RISC-V passes the first 8 integer/pointer args in a0-a7; further args go on
 * the stack. On Capstone a pointer is a 128-bit capability, so a stack-passed
 * pointer argument must use a 16-byte tagged capability slot (lc/sc), not a
 * plain 8-byte ld/sd. The backend currently passes/loads it as a plain 8-byte
 * value, so the callee receives an UNTAGGED pointer; dereferencing it traps:
 *
 *   [CAPSTONE] Cap mem access requires capability: pc = ..., rs1 = x.., imm = 0
 *
 * `f` below takes 8 longs (a0-a7) then two `int *` (the 9th/10th args -> stack).
 * Writing through them faults. This is the root cause of the RV8 `norx` failure
 * (cf_norx32_encrypt has 10 args; its ciphertext/tag are stack-passed caps).
 * It is the same class as the already-fixed va_list capability bug (capabilities
 * in the calling-convention memory path).
 *
 * Expected once fixed: domain returns BEEBS_RET_CORRECT (g8==11 && g9==22).
 * Build/run like the other reduced runtime-qemu domains (start.S + link.ld +
 * beebs_simple_domain.c harness).
 */
typedef unsigned BeebsRet;

static int g8, g9;

__attribute__((noinline)) static void f(long a, long b, long c, long d, long e,
                                        long g, long h, long i, int *p8,
                                        int *p9) {
  (void)a; (void)b; (void)c; (void)d; (void)e; (void)g; (void)h; (void)i;
  *p8 = 11; /* p8 is the 9th arg -> stack-passed capability */
  *p9 = 22; /* p9 is the 10th arg -> stack-passed capability */
}

void initialise_benchmark(void) {}

int benchmark(void) {
  f(0, 0, 0, 0, 0, 0, 0, 0, &g8, &g9);
  return (g8 == 11 && g9 == 22) ? 1 : 0;
}

int verify_benchmark(int result) { return result == 1; }
