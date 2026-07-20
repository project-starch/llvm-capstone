/* Borrow-path cost measurement domain -- GLOBAL-FREE RTL/FPGA variant (2026-07-20).
 *
 * WHY: the earlier borrow_cost_fpga.c uses module-level statics (regions[],
 * raw_src[], copy_dst[], sink). Our LLVM Capstone backend reaches those via
 * `cincoffset X, gp, <absolute>` assuming gp = PCC(cursor 0) -- a form only our
 * QEMU fork fabricates and that is NOT representable on silicon (Jason, 2026-07-20:
 * "store on stack"). So on the FPGA those globals fault (gp=0 -> `delin gp` stalls).
 *
 * This variant is GLOBAL-FREE: no module statics, so the compiler emits no gp use.
 *   - scratch buffers (raw_src/copy_dst) live INSIDE the delivered region;
 *   - the DCE sink is a stack local passed by pointer (kept out of the timed loop);
 *   - measure_borrow runs on a LINEAR capability the entry glue carves off the
 *     stack top (start-fpga-nogp.S), delivered as the 3rd argument.
 * The revoke cost is provenance-independent (that is precisely the O(1) claim), so
 * measuring the borrow/revoke ops on a stack-carved linear cap is faithful; the
 * per-op numbers are cross-checked against the QEMU probe.
 *
 * The measured loop bodies (t0..t1) are BYTE-IDENTICAL to
 * runtime-qemu/borrow-cost-probe/borrow_cost.c -- only the sink store (outside the
 * timed region) changed from a global to `*sinkp`. Entry model: the controller
 * shares ONE REV_SHARED region; the whole measurement runs in that single
 * REGION_SHARE entry (no cross-entry state), writing 8 results into the region
 * base which the host reads back.
 */
#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"
#include "fpga_instrument.h"

/* Region layout (REV_SHARED, 4096 B): [0..63] results (8 slots), [64] sink escape,
 * [512..] raw_src (1024 B), [1536..] copy_dst (1024 B). All well within 4096. */
#define OFF_RAW_SRC  512UL
#define OFF_COPY_DST 1536UL
#define OFF_SINK_ESC 8UL   /* slot index (8*8 = byte 64), just past the 8 results */

/* ==== measured operations -- KEEP THE TIMED BODY IN SYNC with borrow_cost.c ==== */

static unsigned long measure_empty(void) {
  unsigned long i, t0, t1;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_icount();
  return t1 - t0;
}

static unsigned long measure_raw(volatile unsigned long *word,
                                 volatile unsigned long *sinkp) {
  unsigned long i, t0, t1, acc = 0;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    acc ^= word[0];
  }
  t1 = rd_icount();
  *sinkp ^= acc;
  return t1 - t0;
}

static unsigned long measure_borrow(void *lin, volatile unsigned long *sinkp) {
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
  *sinkp ^= acc;
  return t1 - t0;
}

static unsigned long measure_copy(volatile unsigned long *src,
                                  volatile unsigned long *copy_dst,
                                  unsigned long bytes,
                                  volatile unsigned long *sinkp) {
  unsigned long i, off, t0, t1, acc = 0;
  const unsigned long nwords = bytes / sizeof(unsigned long);
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    /* Volatile word copy, not __builtin_memcpy: an external memcpy libcall would
     * force a gp-based code-cap (`cincoffset gp,memcpy`), reintroducing the very
     * gp dependence this variant removes. Volatile also blocks loop-idiom from
     * re-synthesising memcpy. Still O(size), so the copy-vs-borrow point stands. */
    for (off = 0; off < nwords; off++) {
      copy_dst[off] = src[off];
    }
    acc ^= copy_dst[0];
  }
  t1 = rd_icount();
  *sinkp ^= acc;
  return t1 - t0;
}

/* ==== end measured operations ==== */

/* domain_main(region, func, lin_scratch): see start-fpga-nogp.S. The single
 * REGION_SHARE entry both receives the region and runs the measurement. */
void domain_main(void *arg, unsigned func, void *lin_scratch) {
  volatile unsigned long sink = 0;
  unsigned char *region = (unsigned char *)arg;
  volatile unsigned long *res = (volatile unsigned long *)region;

  if (func != BORROW_COST_DPI_REGION_SHARE) {
    /* Only the region-share entry is expected; nothing to do otherwise. */
    return;
  }

  volatile unsigned long *raw_src =
      (volatile unsigned long *)(region + OFF_RAW_SRC);
  volatile unsigned long *copy_dst =
      (volatile unsigned long *)(region + OFF_COPY_DST);

  const unsigned long words = BORROW_COST_BUF_BYTES / sizeof(unsigned long);
  unsigned long k;
  for (k = 0; k < words; k++) {
    raw_src[k] = 0x5E5E5E5Eu + k;
  }

  unsigned long empty = measure_empty();
  unsigned long raw = measure_raw(raw_src, &sink);
  unsigned long copy = measure_copy(raw_src, copy_dst, BORROW_COST_COPY_BYTES, &sink);
  unsigned long copy2 = measure_copy(raw_src, copy_dst, BORROW_COST_COPY_BYTES_2, &sink);
  unsigned long borrow = measure_borrow(lin_scratch, &sink);

  fpga_write_results(region, BORROW_COST_ITERS, empty, raw, borrow, copy,
                     BORROW_COST_COPY_BYTES, copy2, BORROW_COST_COPY_BYTES_2);
  res[OFF_SINK_ESC] = sink; /* force sink to escape so nothing is DCE'd */
}
