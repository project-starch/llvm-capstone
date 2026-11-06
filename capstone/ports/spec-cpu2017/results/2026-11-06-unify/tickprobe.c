/* tickprobe <seconds>: how much of a Plain user thread's time the kernel takes, measured from inside it.
   Spins on rdcycle; any step between two reads longer than GAP cycles is time spent outside the thread
   (a tick, an interrupt, another task). Prints one RESULT line: total cycles, cycles in gaps, gap count,
   their mean, and the gap share. The loop's own step is ~10-20 cycles, so GAP=2000 cannot be the loop. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static inline unsigned long rdcycle(void) { unsigned long c; __asm__ volatile("rdcycle %0" : "=r"(c)); return c; }
int main(int argc, char **argv)
{
    double secs = argc > 1 ? atof(argv[1]) : 30.0;
    const unsigned long GAP = 2000;
    struct timespec t0, t; clock_gettime(CLOCK_MONOTONIC, &t0);
    unsigned long start = rdcycle(), prev = start, gap_cyc = 0, gaps = 0, maxgap = 0, n = 0;
    for (;;) {
        unsigned long c = rdcycle(), d = c - prev;
        if (d > GAP) { gap_cyc += d; ++gaps; if (d > maxgap) maxgap = d; }
        prev = c;
        if ((++n & 0xfffff) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t);
            if ((t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) / 1e9 >= secs) break;
        }
    }
    unsigned long total = prev - start;
    printf("RESULT tickprobe retval=0 total=%lu gapcyc=%lu gaps=%lu meangap=%lu maxgap=%lu share=%.4f\n",
           total, gap_cyc, gaps, gaps ? gap_cyc / gaps : 0, maxgap, total ? (double)gap_cyc / total : 0.0);
    return 0;
}
