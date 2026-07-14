/* Borrow-path cost measurement domain -- RTL/FPGA variant (task-016).
 *
 * Hardware port of tests/runtime-qemu/borrow-cost-probe/borrow_cost.c. The
 * measured loops (measure_empty/raw/borrow/copy) are COPIED VERBATIM from that
 * file and MUST STAY IN SYNC with it -- they are the artifact whose cost the
 * paper quotes, and the whole point of this variant is to measure the SAME code
 * on cycle-accurate silicon. Only the instrumentation differs:
 *   - counting: rd_icount() -> rdcycle (fpga_instrument.h), not QEMU csrdicount;
 *   - output:   results are written into the shared region for the controller
 *               to printf over UART, not dumped via QEMU csdebugcount.
 *
 * UNTESTED IN-SANDBOX: build + run on the caplifive toolchain / FPGA. See
 * README.md for the build recipe and the open items (rdcycle-in-domain, etc.).
 */
#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"
#include "fpga_instrument.h"

/* ---- domain entry glue (identical to the QEMU probe) ---- */

/* Two regions arrive, both as REGION_SHARE entries, distinguished by ORDER:
 *   regions[0] = the LINEAR arena (REV_TRANSFERRED) the borrow loop mrev/revokes;
 *   regions[1] = the results region (REV_SHARED) the host RETAINS, so it can read
 *                the eight results back after the call.
 * The single-region design (write results through the reclaimed borrow handle,
 * host reads the same region back) is UNSOUND: the arena is REV_TRANSFERRED, so
 * after the call the monitor has dropped the host's mapping and the host readback
 * traps (helper_cslcc tag assertion -- the task-007 host-landmine). Validated
 * under QEMU: see RESULTS.md.
 *
 * The array-indexed store below is DELIBERATE: a conditional store of the
 * delivered capability into two *distinct named* globals (arena=arg / else
 * results=arg) ICEs the Capstone backend at -O2 (segfault in codegen; -O0/-O1
 * fine). The indexed store compiles. Flagged to the codegen lane; see RESULTS.md. */
static void *regions[2];
static int share_count;
#define arena regions[0]
#define results_region regions[1]

static volatile unsigned long raw_src[BORROW_COST_BUF_BYTES / sizeof(unsigned long)]
    __attribute__((aligned(16)));
static volatile unsigned long copy_dst[BORROW_COST_BUF_BYTES / sizeof(unsigned long)]
    __attribute__((aligned(16)));
static volatile unsigned long sink;

static inline int receive(void *arg, unsigned func) {
  if (func == BORROW_COST_DPI_REGION_SHARE) {
    regions[share_count++ & 1] = arg; /* [0]=arena, [1]=results, by arrival order */
    return 1;
  }
  return 0;
}

/* ==== measured operations -- KEEP IN SYNC with borrow-cost-probe/borrow_cost.c ==== */

static unsigned long measure_empty(void) {
  unsigned long i, t0, t1;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_icount();
  return t1 - t0;
}

static unsigned long measure_raw(volatile unsigned long *word) {
  unsigned long i, t0, t1, acc = 0;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    acc ^= word[0];
  }
  t1 = rd_icount();
  sink ^= acc;
  return t1 - t0;
}

static unsigned long measure_borrow(void *lin) {
  unsigned long i, t0, t1, acc = 0;
  void *c = lin;
  *(volatile unsigned long *)lin = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    void *rev = __builtin_capstone_cap_mrev(c);
    void *del = __builtin_capstone_cap_delin(c);
    acc ^= *(volatile unsigned long *)del;
    c = __builtin_capstone_cap_revoke(rev);
  }
  t1 = rd_icount();
  sink ^= acc;
  return t1 - t0;
}

#define BORROW_COST_COPY_CHUNK 256UL
static unsigned long measure_copy(volatile unsigned long *src, unsigned long bytes) {
  unsigned long i, off, t0, t1, acc = 0;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    for (off = 0; off < bytes; off += BORROW_COST_COPY_CHUNK) {
      __builtin_memcpy((void *)(copy_dst + off / sizeof(unsigned long)),
                       (const void *)(src + off / sizeof(unsigned long)),
                       BORROW_COST_COPY_CHUNK);
    }
    acc ^= copy_dst[0];
  }
  t1 = rd_icount();
  sink ^= acc;
  return t1 - t0;
}

/* ==== end measured operations ==== */

void domain_main(void *arg, unsigned func) {
  if (receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg;

  const unsigned long words = BORROW_COST_BUF_BYTES / sizeof(unsigned long);
  unsigned long k;
  for (k = 0; k < words; k++) {
    raw_src[k] = 0x5E5E5E5Eu + k;
  }

  unsigned long empty = measure_empty();
  unsigned long raw = measure_raw(raw_src);
  unsigned long copy = measure_copy(raw_src, BORROW_COST_COPY_BYTES);
  unsigned long copy2 = measure_copy(raw_src, BORROW_COST_COPY_BYTES_2);
  unsigned long borrow = measure_borrow(arena);

  /* Results go to the RETAINED results region, NOT the reclaimed arena handle:
   * the arena is REV_TRANSFERRED and the host can no longer read it back. */
  fpga_write_results(results_region, BORROW_COST_ITERS, empty, raw, borrow, copy,
                     BORROW_COST_COPY_BYTES, copy2, BORROW_COST_COPY_BYTES_2);

  *res = BORROW_COST_RET_OK;
}
