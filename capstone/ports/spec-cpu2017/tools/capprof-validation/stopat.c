/* capprof work-bounded window test (stopat-run.sh, stopatcheck.py). stopat_f is called 1200 times,
 * 1000 of them inside two windows; its body and the loop around the call carry instructions with
 * unique immediates, so the window files show them by their bytes alone:
 *   stopat_f   addi t3,t3,0x3d1 (its first instruction: the stop site) / addi t3,t3,0x3d2
 *   run loop   addi t3,t3,0x3e2 before the call / addi t3,t3,0x3e3 after it
 *   main       0x3e1 / 0x3e4 around window 0's loop, 0x3e5 / 0x3e6 around window 1's
 * main: run(100) START run(300) STOP run(100) START run(700) STOP. Entries inside windows are
 * 1..300 in window 0 and 301..1000 in window 1; the 100 calls before each window count nothing. */
#include <stdio.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define MARK(imm) asm volatile("addi t3, t3, " #imm ::: "t3", "memory")

__attribute__((noinline)) long stopat_f(long x) {
    MARK(0x3d1); MARK(0x3d2);
    return x * 3 + 1;
}

__attribute__((noinline)) long run(long n, long acc) {
    for (long i = 0; i < n; i++) {
        MARK(0x3e2);
        acc = stopat_f(acc);
        MARK(0x3e3);
    }
    return acc;
}

int main(void) {
    long acc = run(100, 0);
    START(); MARK(0x3e1); acc = run(300, acc); MARK(0x3e4); STOP();
    acc = run(100, acc);
    START(); MARK(0x3e5); acc = run(700, acc); MARK(0x3e6); STOP();
    printf("%ld\n", acc);
    return 0;
}
