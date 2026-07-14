/* Real-workload temporal-safety overhead -- Capstone side. Runs the shared
 * binary-search-tree build/lookup/destroy workload (tests/shared/tree_workload.h)
 * under the three allocator configs (-DROF_COST_MODE: bump/norevoke/revoke), so
 * the Capstone revoke-at-free cost is measured on the SAME real pointer-chasing
 * allocation pattern as the CHERI side (tests/cheri-perf/tree_cheri.c), not a
 * synthetic malloc/free loop.
 *
 * Reuses the microbench's controller + serial protocol (revoke_cost_probe.h);
 * only the measured region differs (tw_run instead of measure_alloc_free). The
 * revoke-vs-norevoke delta isolates the revoke-at-free op: both modes pay the
 * identical alloc-side bookkeeping (SPLIT+mrev+delin, slot table, rof_find), so
 * their difference is purely the free-time revoke -- workload-independent by
 * construction, which is exactly what we expect to confirm.
 *
 * Total allocation is TW_KEYS*TW_ROUNDS; kept within the 256 KiB Phase-0 arena
 * (the rof allocator never coalesces). TW_KEYS=2000, TW_ROUNDS=2 -> 4000 nodes
 * (~192 KiB at 48 B/node).
 */
#include "revoke_cost_probe.h"
#include "../../../benchmarks/sqlite/revoke_on_free_alloc.h"

#ifndef ROF_COST_MODE
#define ROF_COST_MODE ROF_COST_MODE_REVOKE
#endif

/* ---- instrumentation (identical to revoke_cost.c) ---- */
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

static void *grant;
static inline int receive(void *arg, unsigned func) {
  if (func == ROF_COST_DPI_REGION_SHARE) {
    grant = arg;
    return 1;
  }
  return 0;
}

static volatile unsigned long sink;

/* ---- allocator under test, selected at compile time (as in revoke_cost.c) ---- */
#if ROF_COST_MODE == ROF_COST_MODE_BUMP
static void *bump_base;
static unsigned long bump_off;
static inline void alloc_init(void *g) {
  bump_base = __builtin_capstone_cap_delin(g);
  bump_off = 0;
}
static inline void *alloc_malloc(unsigned long n) {
  n = (n + 15UL) & ~15UL;
  void *p = (char *)bump_base + bump_off;
  bump_off += n;
  return p;
}
static inline void alloc_free(void *p) { (void)p; }
#else
static inline void alloc_init(void *g) {
  rof_init(g);
#if ROF_COST_MODE == ROF_COST_MODE_NOREVOKE
  rof_no_revoke = 1;
#else
  rof_no_revoke = 0;
#endif
}
static inline void *alloc_malloc(unsigned long n) { return rof_malloc(n); }
static inline void alloc_free(void *p) { rof_free(p); }
#endif

/* ---- the shared workload, driven through the selected allocator ---- */
#define TW_KEYS 2000u
#define TW_ROUNDS 2u
#define TW_MALLOC(n) alloc_malloc(n)
#define TW_FREE(p) alloc_free(p)
#include "../../shared/tree_workload.h"

void domain_main(void *arg, unsigned func) {
  if (receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  alloc_init(grant);

  unsigned long t0 = rd_icount();
  unsigned long ops = tw_run(&sink);
  unsigned long t1 = rd_icount();

  dbg_count_clear();
  dbg_count_add(ROF_COST_SLOT_ITERS, ops);        /* total node lifecycles */
  dbg_count_add(ROF_COST_SLOT_EMPTY, 0);          /* no calibration subtract */
  dbg_count_add(ROF_COST_SLOT_ALLOCFREE, t1 - t0);/* total instrs over the workload */
  dbg_count_add(ROF_COST_SLOT_MODE, ROF_COST_MODE);
  dbg_count_print();

  *res = ROF_COST_RET_OK;
}
