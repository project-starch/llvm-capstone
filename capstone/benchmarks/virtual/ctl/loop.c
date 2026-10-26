/* ctl/loop and ctl/empty (-DCTL_EMPTY): the two controls share this unit and differ ONLY in the
 * instructions inside the asm statement.
 *
 *   ctl/loop:   li t, CTL_LOOP_N        (lui + addi(w): 2 instructions for N = 10,000,000)
 *            1: addi t, t, -1
 *               bnez t, 1b              (2 instructions per iteration)
 *   ctl/empty:  (no instructions)
 *
 * So instret(ctl/loop) - instret(ctl/empty) = 2*N + 2 exactly, whatever the marks cost: ctl/empty
 * is the cost of one bracket (two marks, two clock_gettime calls, the unit's call, the store of
 * `left` and the return), the same in both arms. The prediction is printed (BENCH-INFO). Under
 * QEMU -icount shift=0 the same difference holds for ns. */
#include <stdio.h>
#include "bench-harness.h"

#ifndef CTL_LOOP_N
#define CTL_LOOP_N 10000000
#endif
#define STR2(x) #x
#define STR(x) STR2(x)

static volatile unsigned long left; /* volatile: the store stays in both variants */

static void unit(void) {
  unsigned long t;
#ifdef CTL_EMPTY
  __asm__ volatile("" : "=r"(t)::"memory");
#else
  __asm__ volatile("li %0, " STR(CTL_LOOP_N) "\n"
                   "1: addi %0, %0, -1\n"
                   "bnez %0, 1b"
                   : "=&r"(t)::"memory");
#endif
  left = t;
}

static int post(unsigned long long *check) {
#ifdef CTL_EMPTY
  *check = 0;
  return 1;
#else
  *check = left;
  return left == 0;
#endif
}

int main(int argc, char **argv) {
#ifdef CTL_EMPTY
  return bench_run("ctl/empty", argc, argv, unit, post);
#else
  printf("BENCH-INFO ctl/loop predicted_instret_minus_empty=%lu n=%lu\n", 2UL * CTL_LOOP_N + 2,
         (unsigned long)CTL_LOOP_N);
  return bench_run("ctl/loop", argc, argv, unit, post);
#endif
}
