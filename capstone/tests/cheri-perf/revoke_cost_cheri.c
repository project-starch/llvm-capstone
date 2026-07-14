/* Temporal-safety overhead microbenchmark -- CHERI side.
 *
 * The CHERI arm of the QEMU-to-QEMU CHERI-vs-Capstone performance comparison
 * (plans/perf-cheri-vs-capstone-qemu.md; PI 2026-07-14). The Capstone arm lives
 * in tests/runtime-qemu/revoke-cost-probe/ and measures the SAME workload -- a
 * malloc(BLOCK) -> touch -> free() loop, ITERS times -- so the two per-op
 * instruction counts are directly comparable.
 *
 * This is an ordinary CheriBSD *purecap* userspace program. The revocation
 * policy (off / async-quarantine / revoke-every-free) is chosen by the CALLER
 * via `sysctl security.cheri.runtime_revocation_default` /
 * `..._every_free_default` BEFORE exec (see run-in-guest.sh) -- exactly the knob
 * the cheri-baseline task used. So one binary, three configs, no recompile.
 *
 * Instrumentation: we bracket the loop with the RISC-V retired-instruction
 * counter (rdinstret; rdcycle with -DUSE_CYCLE). instret counts retirements in
 * ALL privilege modes, so the delta across the loop INCLUDES the kernel-side
 * revocation sweep -- which is the whole point on CHERI: its temporal cost is
 * paid in a kernel quarantine sweep, not at the free. Capstone's is an inline
 * capability op; here it is amortized kernel work. Counting user+kernel captures
 * both fairly. Under QEMU `-icount shift=0` the count is deterministic; without
 * -icount it is a real retired-instruction count with mild interrupt jitter that
 * we damp with warmup + a large ITERS + median-of-trials in the driver.
 *
 * Output: one structured line the host driver greps:
 *   RCPERF mode=<name> iters=<n> block=<b> empty=<c> allocfree=<c> perop=<f>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef RC_ITERS
#define RC_ITERS 200000UL
#endif
#ifndef RC_BLOCK
#define RC_BLOCK 64UL
#endif

/* Read the retired-instruction counter (or cycle counter). Plain scalar CSR
 * read -- no capability operand -- so it is identical purecap or not. Requires
 * userspace access (scounteren IR/CY); CheriBSD/FreeBSD RISC-V enables it. If it
 * traps SIGILL we will see the fault in the run and fall back to the other CSR. */
static inline unsigned long rd_count(void) {
  unsigned long v;
#ifdef USE_CYCLE
  __asm__ volatile("rdcycle %0" : "=r"(v));
#else
  __asm__ volatile("rdinstret %0" : "=r"(v));
#endif
  return v;
}

static volatile unsigned long sink;

/* Empty calibration loop: same loop control + sink feed as the real loop, no
 * allocation. Subtracting it removes shared loop overhead (matches the Capstone
 * arm's measure_empty). */
static unsigned long measure_empty(unsigned long iters) {
  unsigned long i, t0, t1, x = 0;
  t0 = rd_count();
  for (i = 0; i < iters; i++) {
    x ^= i;
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_count();
  sink ^= x;
  return t1 - t0;
}

/* The measured workload: malloc a block, touch it, free it -- iters times.
 * Byte-identical in shape to the Capstone arm's measure_alloc_free. */
static unsigned long measure_alloc_free(unsigned long iters, unsigned long block) {
  unsigned long i, t0, t1;
  t0 = rd_count();
  for (i = 0; i < iters; i++) {
    volatile char *p = (volatile char *)malloc(block);
    if (p) {
      p[0] = (char)i;
      sink ^= (unsigned char)p[0];
      free((void *)p);
    }
  }
  t1 = rd_count();
  return t1 - t0;
}

int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "unknown";
  unsigned long iters = argc > 2 ? strtoul(argv[2], NULL, 0) : RC_ITERS;
  unsigned long block = argc > 3 ? strtoul(argv[3], NULL, 0) : RC_BLOCK;

  /* Warm the allocator + page in the counter path so steady-state dominates. */
  for (int w = 0; w < 3; w++) {
    void *p = malloc(block);
    if (p) { memset(p, 0, block); free(p); }
  }
  (void)measure_empty(1024);
  (void)measure_alloc_free(1024, block);

  unsigned long empty = measure_empty(iters);
  unsigned long allocfree = measure_alloc_free(iters, block);

  double perop = (double)(allocfree - empty) / (double)iters;
  printf("RCPERF mode=%s iters=%lu block=%lu empty=%lu allocfree=%lu perop=%.3f\n",
         mode, iters, block, empty, allocfree, perop);
  fflush(stdout);
  return 0;
}
