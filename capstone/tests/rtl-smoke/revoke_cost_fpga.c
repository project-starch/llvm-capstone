/* Temporal-safety overhead microbenchmark domain -- RTL/FPGA variant (task-016).
 *
 * Hardware port of tests/runtime-qemu/revoke-cost-probe/revoke_cost.c. The
 * allocator configs and the measured loops (measure_empty / measure_alloc_free)
 * are COPIED VERBATIM from that file and MUST STAY IN SYNC with it -- the point
 * of this variant is to measure the SAME code on cycle-accurate silicon. Only
 * the instrumentation differs:
 *   - counting: rd_icount() -> rd_cycles() (mcycle by default, fpga_instrument.h),
 *               not the QEMU rdcycle-under-icount / csrdicount readout;
 *   - output:   the 4 counters are written into a RETAINED (REV_SHARED) results
 *               region for the controller to printf over UART, not dumped via the
 *               QEMU csdebugcount* ops (which do not exist on the CVA6 core).
 *
 * The allocator config is selected at compile time by -DROF_COST_MODE
 * (BUMP/NOREVOKE/REVOKE); three .dom builds, one per config, exactly as the QEMU
 * probe. See revoke_cost_probe.h for the three modes and the overhead breakdown.
 *
 * UNTESTED ON HARDWARE: the QEMU plumbing validation (mcycle-in-domain + the
 * two-region result hand-off) is in RESULTS.md.
 */
#include "../runtime-qemu/revoke-cost-probe/revoke_cost_probe.h"
#include "fpga_instrument.h"
#include "../../benchmarks/sqlite/revoke_on_free_alloc.h"

#ifndef ROF_COST_MODE
#define ROF_COST_MODE ROF_COST_MODE_REVOKE
#endif

/* ---- result hand-off (RTL/FPGA) ----
 * Write the 4 counters into the retained results region for the controller to
 * read back and printf over UART. Slots defined in revoke_cost_probe.h. */
static inline void fpga_write_revoke_results(void *out, unsigned long iters,
                                             unsigned long empty,
                                             unsigned long allocfree,
                                             unsigned long mode) {
  volatile unsigned long *r = (volatile unsigned long *)out;
  r[ROF_COST_SLOT_ITERS] = iters;
  r[ROF_COST_SLOT_EMPTY] = empty;
  r[ROF_COST_SLOT_ALLOCFREE] = allocfree;
  r[ROF_COST_SLOT_MODE] = mode;
}

/* ---- domain entry glue ----
 *
 * Two regions arrive as REGION_SHARE entries, distinguished by ARRIVAL ORDER:
 *   regions[0] = the LINEAR arena (REV_TRANSFERRED) the allocator SPLIT/mrev/
 *                revokes; regions[1] = the results region (REV_SHARED) the host
 *                RETAINS so it can read the 4 results back after the call.
 * A single REV_TRANSFERRED region cannot be read back by the host (task-007
 * host-landmine; see the borrow-cost port RESULTS.md). */
static void *regions[2];
static int share_count;
#define arena regions[0]
#define results_region regions[1]

static inline int receive(void *arg, unsigned func) {
  if (func == ROF_COST_DPI_REGION_SHARE) {
    regions[share_count++ & 1] = arg; /* [0]=arena, [1]=results, by arrival order */
    return 1;
  }
  return 0;
}

static volatile unsigned long sink;

/* ---- allocator under test, selected at compile time (VERBATIM from revoke_cost.c) ---- */
#if ROF_COST_MODE == ROF_COST_MODE_BUMP
/* Plain bump allocator over a broad NONLIN heap cap: one delin of the arena up
 * front, then interior pointers by offset. No SPLIT/mrev/delin per object, no
 * revoke. free is a no-op. The UNPROTECTED baseline. */
static void *bump_base;
static unsigned long bump_off;
static inline void alloc_init(void *g) {
  bump_base = __builtin_capstone_cap_delin(g);
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

/* ---- measured loops (VERBATIM from revoke_cost.c) ---- */

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

  alloc_init(arena);

  unsigned long empty = measure_empty();
  unsigned long allocfree = measure_alloc_free();

  /* Results go to the RETAINED results region, NOT the transferred arena. */
  fpga_write_revoke_results(results_region, ROF_COST_ITERS, empty, allocfree,
                            ROF_COST_MODE);

  *res = ROF_COST_RET_OK;
}
