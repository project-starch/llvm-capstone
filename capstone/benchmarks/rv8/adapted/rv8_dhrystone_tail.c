/*
 * Capstone adapted oracle for RV8 `dhrystone`.
 *
 * rv8-bench's dhrystone is a hosted program: main() -> Proc0() runs the LOOPS
 * kernel under gettimeofday timing and prints DMIPS (no correctness check). This
 * tail provides the BEEBS-style domain entry points: it runs the same Proc0()
 * kernel (timing/printf are no-op stubs) and verifies the **canonical Dhrystone
 * end-state self-check** of the global state, which is deterministic.
 *
 * The build pins LOOPS to 100000 (the perf count is irrelevant to us; we only
 * need to exercise the kernel and validate its deterministic result), so the
 * documented self-check values are:
 *   IntGlob==5, BoolGlob==1, Char1Glob=='A', Char2Glob=='B',
 *   Array1Glob[8]==7, Array2Glob[8][7]==LOOPS+10 (==100010).
 * Confirmed by a native gcc reference build of the same source at LOOPS=100000.
 */
#include "rv8_capstone_preamble.h"

typedef int Array1Dim[51];
typedef int Array2Dim[51][51];

extern int IntGlob;
extern int BoolGlob;
extern char Char1Glob;
extern char Char2Glob;
extern Array1Dim Array1Glob;
extern Array2Dim Array2Glob;

extern void Proc0(void);
extern void rv8_arena_init(void);

void initialise_benchmark(void) { rv8_arena_init(); }

int benchmark(void) {
  Proc0();
  return 0; /* result carried in global state; checked below */
}

int verify_benchmark(int result) {
  (void)result;
  return (IntGlob == 5 && BoolGlob == 1 && Char1Glob == 'A' &&
          Char2Glob == 'B' && Array1Glob[8] == 7 &&
          Array2Glob[8][7] == 100010)
             ? 1
             : 0;
}
