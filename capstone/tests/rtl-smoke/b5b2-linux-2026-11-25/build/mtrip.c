/* mtrip: the cost of one M round trip on the board, as seen from U. On this CVA6 there is no time CSR, so rdtime
 * traps to M and OpenSBI emulates it; rdcycle does not trap (it is the reference clock). Prints, per variant, the
 * mean cycles per op over N ops, best of R repeats:
 *   rdtime   one M trip (illegal-instruction emulation), the variable
 *   rdcycle  no trap, the control: a few cycles
 *   getppid  a U->S syscall round trip, no M involvement, the second control */
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>
#define N 20000
#define R 5
static inline unsigned long cyc(void) { unsigned long c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
static inline unsigned long tim(void) { unsigned long t; __asm__ volatile("rdtime %0" : "=r"(t)); return t; }
int main(void) {
    unsigned long best[3] = {~0UL, ~0UL, ~0UL}, sink = 0;
    for (int r = 0; r < R; r++) {
        unsigned long c0 = cyc(); for (int i = 0; i < N; i++) sink += tim(); unsigned long c1 = cyc();
        for (int i = 0; i < N; i++) sink += cyc(); unsigned long c2 = cyc();
        for (int i = 0; i < N; i++) sink += syscall(SYS_getppid); unsigned long c3 = cyc();
        if (c1 - c0 < best[0]) best[0] = c1 - c0;
        if (c2 - c1 < best[1]) best[1] = c2 - c1;
        if (c3 - c2 < best[2]) best[2] = c3 - c2;
    }
    printf("MTRIP rdtime %lu.%02lu rdcycle %lu.%02lu getppid %lu.%02lu cycles/op (N=%d best of %d) sink=%lu\n",
           best[0] / N, best[0] % N * 100 / N, best[1] / N, best[1] % N * 100 / N, best[2] / N, best[2] % N * 100 / N,
           N, R, sink & 1);
    return 0;
}
