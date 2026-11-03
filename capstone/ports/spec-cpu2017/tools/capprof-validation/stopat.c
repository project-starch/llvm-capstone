/* capprof work-bounded window test (stopat-run.sh, stopatcheck.py). stopat_f is called 1200 times,
 * 1000 of them inside two windows; its body and the loop around the call carry instructions with
 * unique immediates, so the window files show them by their bytes alone:
 *   stopat_f   addi t3,t3,0x3d1 (its first instruction: the stop site) / addi t3,t3,0x3d2
 *   run loop   addi t3,t3,0x3e2 before the call / addi t3,t3,0x3e3 after it
 *   main       0x3e1 / 0x3e4 around window 0's loop, 0x3e5 / 0x3e6 around window 1's
 * main: run(100) START run(300) STOP run(100) START run(700) STOP. Entries inside windows are
 * 1..300 in window 0 and 301..1000 in window 1; the 100 calls before each window count nothing.
 *
 * `stopat fault`: START, 64 calls of stopat_g(page i of a fresh mapping) with addi t3 0x3f2 before
 * and 0x3f3 after each call, STOP. stopat_g's first instruction is a store that faults on its
 * fresh page and is retried, and it is the last instruction of its page, so its block ends at it
 * and the instruction after it (ret, the next page) is in another block: an arrival that faults
 * must count as one entry with its retry. */
#include <stdio.h>
#include <stdlib.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define MARK(imm) asm volatile("addi t3, t3, " #imm ::: "t3", "memory")

__attribute__((noinline)) long stopat_f(long x) {
    MARK(0x3d1); MARK(0x3d2);
    return x * 3 + 1;
}

__asm__(".text\n.balign 4096\n.skip 4092\n.globl stopat_g\n.type stopat_g,@function\n"
        "stopat_g:\n sd zero, 0(a0)\n ret\n.size stopat_g, .-stopat_g\n");
void stopat_g(char *p);

__attribute__((noinline)) long run(long n, long acc) {
    for (long i = 0; i < n; i++) {
        MARK(0x3e2);
        acc = stopat_f(acc);
        MARK(0x3e3);
    }
    return acc;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        long pages = 64;
        char *p = malloc(pages * 4096 + 4096);
        if (!p) return 1;
        char *q = p + ((4096 - ((unsigned long)p & 4095)) & 4095);  /* keeps p's capability */
        START();
        for (long i = 0; i < pages; i++) {
            MARK(0x3f2);
            stopat_g(q + i * 4096);
            MARK(0x3f3);
        }
        STOP();
        printf("%d\n", q[8]);
        return 0;
    }
    long acc = run(100, 0);
    START(); MARK(0x3e1); acc = run(300, acc); MARK(0x3e4); STOP();
    acc = run(100, acc);
    START(); MARK(0x3e5); acc = run(700, acc); MARK(0x3e6); STOP();
    printf("%ld\n", acc);
    return 0;
}
