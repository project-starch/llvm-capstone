/* capprof validation program. Each loop's body has instructions with unique
 * immediates, so the window file shows them by their bytes alone:
 *   loopA: addi t3,t3,0x3a1 / xori t4,t4,0x5a1 / addi %0,%0,-1 / bnez  (body 4)
 *   loopB: addi t3,t3,0x3b1 / xori t4,t4,0x5b1 / addi %0,%0,-1 / bnez
 *   loopC: addi t3,t3,0x3c1 / xori t4,t4,0x5c1 / addi %0,%0,-1 / bnez
 * Markers: slti x0,x0,0x7a1 (start), 0x7a2 (stop).
 * usage: vt MODE
 *   window   loopA(1000) START loopA(12345) loopB(777) STOP loopA(5000)
 *   long     START loopC(50000000) STOP
 *   fault    p = 64 fresh pages; START touch each once (sd, unique offset) STOP
 *   prefault the same, every page touched once before START
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define START() asm volatile(".insn i 0x13, 2, x0, x0, 0x7a1" ::: "memory")
#define STOP()  asm volatile(".insn i 0x13, 2, x0, x0, 0x7a2" ::: "memory")
#define LOOP(name, a, x) \
__attribute__((noinline)) void name(long n) { \
    asm volatile("1:\n addi t3, t3, " #a "\n xori t4, t4, " #x "\n addi %0, %0, -1\n bnez %0, 1b\n" \
                 : "+r"(n) :: "t3", "t4"); }
LOOP(loopA, 0x3a1, 0x5a1)
LOOP(loopB, 0x3b1, 0x5b1)
LOOP(loopC, 0x3c1, 0x5c1)

__attribute__((noinline)) void touch(char *p, long pages) {
    /* one store per page; li t5,0x6d1 makes the store recognisable. The page
     * pointer is advanced in C, so it stays a capability on Capstone. */
    for (long i = 0; i < pages; i++)
        asm volatile("li t5, 0x6d1\n sd t5, 0(%0)\n" :: "r"(p + i * 4096) : "t5", "memory");
}

int main(int argc, char **argv) {
    const char *m = argc > 1 ? argv[1] : "window";
    if (!strcmp(m, "window")) {
        loopA(1000); START(); loopA(12345); loopB(777); STOP(); loopA(5000);
    } else if (!strcmp(m, "long")) {
        START(); loopC(50000000); STOP();
    } else if (!strcmp(m, "fault") || !strcmp(m, "prefault")) {
        long pages = 64;
        char *p = malloc(pages * 4096 + 4096);
        if (!p) return 1;
        char *q = p + ((4096 - ((unsigned long)p & 4095)) & 4095);  /* keeps p's capability */
        if (!strcmp(m, "prefault")) touch(q, pages);
        START(); touch(q, pages); STOP();
        printf("%d\n", q[8]);
    } else return 2;
    return 0;
}
