/* Board probe for the perf instrument (CAPSTONE_VM_BENCH_COUNTERS): reads the module's cumulative
 * bench_{cycles,instret,entries} around (a) an empty region and (b) a loop of a known instruction count,
 * five times each. Prediction for (b): instret delta = 2*N + small constant (addi + bnez per iteration),
 * identical across repeats; cycles delta > 0. Absent files print counters=absent (QEMU, non-perf module). */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#define N 10000000UL
static const char *path[3] = {"/sys/module/capstone_vm/parameters/bench_cycles",
                              "/sys/module/capstone_vm/parameters/bench_instret",
                              "/sys/module/capstone_vm/parameters/bench_entries"};
static int fd[3];
static char buf[3][32];
static void mark(void) { for (int i = 0; i < 3; i++) { ssize_t n = pread(fd[i], buf[i], 31, 0); buf[i][n > 0 ? n : 0] = 0; } }
static void take(unsigned long long v[3]) { for (int i = 0; i < 3; i++) v[i] = strtoull(buf[i], 0, 10); }
static void spin(unsigned long n) { __asm__ volatile("1: addi %0, %0, -1\n bnez %0, 1b" : "+r"(n)); }
int main(void)
{
    for (int i = 0; i < 3; i++) if ((fd[i] = open(path[i], O_RDONLY)) < 0) { puts("PROBE counters=absent"); return 2; }
    unsigned long long a[3], b[3];
    for (int r = 0; r < 5; r++) {
        mark(); take(a); mark(); take(b);
        printf("PROBE empty r=%d cycles=%llu instret=%llu entries=%llu\n", r, b[0]-a[0], b[1]-a[1], b[2]-a[2]);
        mark(); take(a); spin(N); mark(); take(b);
        printf("PROBE loop r=%d n=%lu cycles=%llu instret=%llu entries=%llu predicted_instret_min=%lu\n",
               r, N, b[0]-a[0], b[1]-a[1], b[2]-a[2], 2*N);
    }
    puts("PROBE done");
    return 0;
}
