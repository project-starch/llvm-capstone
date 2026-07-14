/* Temporal-safety overhead microbenchmark domain -- Capstone side.
 *
 * Built with the Capstone clang, loaded as a domain_main .dom. Receives one
 * LINEAR arena on the REGION_SHARE entry and, on the CALL entry, measures the
 * dynamic instruction count of a malloc/touch/free loop under one allocator
 * config selected at compile time by -DROF_COST_MODE. See revoke_cost_probe.h
 * for the three modes and the overhead breakdown they yield.
 */
#include "revoke_cost_probe.h"
#include "../../../benchmarks/sqlite/revoke_on_free_alloc.h"

#ifndef ROF_COST_MODE
#define ROF_COST_MODE ROF_COST_MODE_REVOKE
#endif

/* ---- instrumentation ---- */
/* Count via rdcycle, not the csrdicount custom op: under -icount rdcycle
 * advances one-per-retired-instruction, so a delta across a region is that
 * region's dynamic instruction count -- verified against csrdicount in the
 * rtl-smoke run (raw 2 / borrow 6, identical to task-014). Using rdcycle also
 * (a) keeps this identical to the FPGA borrow-cost port's readout, and (b) does
 * not depend on a csrdicount-enabled capstone-qemu binary (the checked-in build
 * predates that op). csdebugcount* below ARE in this binary. */
static inline unsigned long rd_icount(void) {
  unsigned long v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}
static inline void dbg_count_add(unsigned long slot, unsigned long delta) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x45, x0, %0, %1"
                   : : "r"(slot), "r"(delta) : "memory");
}
static inline void dbg_count_clear(void) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x47, x0, x0, x0" : : : "memory");
}
static inline void dbg_count_print(void) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x46, x0, x0, x0" : : : "memory");
}

/* ---- domain entry glue ---- */
static void *grant; /* the monitor-granted LINEAR arena, parked between entries */

static inline int receive(void *arg, unsigned func) {
  if (func == ROF_COST_DPI_REGION_SHARE) {
    grant = arg;
    return 1;
  }
  return 0;
}

static volatile unsigned long sink;

/* ---- allocator under test, selected at compile time ----
 *
 * The three configs share the SAME loop body (measure_alloc_free); only these
 * three inlines differ, so the measured delta is purely the allocator. */
#if ROF_COST_MODE == ROF_COST_MODE_BUMP
/* Plain bump allocator over a broad NONLIN heap capability: one delin of the
 * arena up front (a normal allocator holding a heap cap), then interior pointers
 * by offset. No SPLIT/mrev/delin per object, no revoke. free is a no-op. */
static void *bump_base;
static unsigned long bump_off;
static inline void alloc_init(void *g) {
  bump_base = __builtin_capstone_cap_delin(g); /* NONLIN, copyable, broad bounds */
  bump_off = 0;
}
static inline void *alloc_malloc(unsigned long n) {
  n = (n + 15UL) & ~15UL;
  void *p = (char *)bump_base + bump_off; /* cincoffset on NONLIN base */
  bump_off += n;
  return p;
}
static inline void alloc_free(void *p) { (void)p; }
#else /* NOREVOKE or REVOKE: the real revoke-on-free allocator */
static inline void alloc_init(void *g) {
  rof_init(g);
#if ROF_COST_MODE == ROF_COST_MODE_NOREVOKE
  rof_no_revoke = 1; /* SPLIT+mrev+delin per malloc, but free does not revoke */
#else
  rof_no_revoke = 0; /* full temporal safety: revoke per free */
#endif
}
static inline void *alloc_malloc(unsigned long n) { return rof_malloc(n); }
static inline void alloc_free(void *p) { rof_free(p); }
#endif

/* ---- measured loops ---- */

/* Empty calibration loop: same loop control + sink feed as the real loop, no
 * allocation. Subtracting it removes the shared loop overhead. */
static unsigned long measure_empty(void) {
  unsigned long i, t0, t1, x = 0;
  t0 = rd_icount();
  for (i = 0; i < ROF_COST_ITERS; i++) {
    x ^= i;
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_icount();
  sink ^= x;
  return t1 - t0;
}

/* The measured workload: malloc a block, touch it, free it -- ITERS times. */
static unsigned long measure_alloc_free(void) {
  unsigned long i, t0, t1;
  t0 = rd_icount();
  for (i = 0; i < ROF_COST_ITERS; i++) {
    volatile char *p = (volatile char *)alloc_malloc(ROF_COST_BLOCK);
    p[0] = (char)i;
    sink ^= (unsigned char)p[0];
    alloc_free((void *)p);
  }
  t1 = rd_icount();
  return t1 - t0;
}

void domain_main(void *arg, unsigned func) {
  if (receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  alloc_init(grant);

  unsigned long empty = measure_empty();
  unsigned long allocfree = measure_alloc_free();

  dbg_count_clear();
  dbg_count_add(ROF_COST_SLOT_ITERS, ROF_COST_ITERS);
  dbg_count_add(ROF_COST_SLOT_EMPTY, empty);
  dbg_count_add(ROF_COST_SLOT_ALLOCFREE, allocfree);
  dbg_count_add(ROF_COST_SLOT_MODE, ROF_COST_MODE);
  dbg_count_print();

  *res = ROF_COST_RET_OK;
}
