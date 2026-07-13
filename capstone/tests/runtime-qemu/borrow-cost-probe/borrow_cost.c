/* Borrow-path cost measurement domain (task-014, paper deliverable 2).
 *
 * Built with the Capstone clang (capability builtins available), loaded as a
 * domain_main .dom. Receives a LINEAR arena on the REGION_SHARE entry and, on
 * the CALL entry, measures the dynamic instruction count of three variants of
 * the same borrow-one-word boundary operation. See borrow_cost_probe.h.
 */
#include "borrow_cost_probe.h"

/* ---- emulator instrumentation ops (custom opcode 0x5b, funct3 0x1) ---- */

/* csrdicount rd -- read QEMU's raw retired-instruction count (icount). Only a
 * true instruction count under -icount; returns 0 otherwise. */
static inline unsigned long rd_icount(void) {
  unsigned long v;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x48, %0, x0, x0" : "=r"(v) : : "memory");
  return v;
}
/* csdebugcount slot, delta -- capstone_debug_counters[slot] += delta. */
static inline void dbg_count_add(unsigned long slot, unsigned long delta) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x45, x0, %0, %1"
                   :
                   : "r"(slot), "r"(delta)
                   : "memory");
}
/* csdebugcountclear / csdebugcountprint. */
static inline void dbg_count_clear(void) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x47, x0, x0, x0" : : : "memory");
}
static inline void dbg_count_print(void) {
  __asm__ volatile(".insn r 0x5b, 0x1, 0x46, x0, x0, x0" : : : "memory");
}

/* ---- domain entry glue (see intra-domain probe's probe_domain.h) ---- */

/* The monitor-granted arena, parked between the REGION_SHARE and CALL entries.
 * 16-byte aligned by the i128 pointer ABI, as store_capregval requires. */
static void *arena;

/* Plain source for the raw and copy variants: the domain's own .bss, reached
 * through its broad ambient data capability -- "today's raw pointer", with no
 * per-object capability machinery. (Only the borrow variant needs the LINEAR
 * arena, so raw/copy use a separate source and leave `arena` pristine LIN.) */
static volatile unsigned long raw_src[BORROW_COST_BUF_BYTES / sizeof(unsigned long)]
    __attribute__((aligned(16)));
/* Private destination for the copy baseline. */
static volatile unsigned long copy_dst[BORROW_COST_BUF_BYTES / sizeof(unsigned long)]
    __attribute__((aligned(16)));

/* Sink that every variant feeds, kept live so the optimiser cannot delete the
 * accesses it is meant to measure. */
static volatile unsigned long sink;

/* domain_main is re-entered once per DPI operation. Returns 1 on the
 * REGION_SHARE delivery (nothing else to do that entry). */
static inline int receive(void *arg, unsigned func) {
  if (func == BORROW_COST_DPI_REGION_SHARE) {
    arena = arg; /* stc: the delivered LINEAR capability, tag intact */
    return 1;
  }
  return 0;
}

/* ---- the measured operations ----
 *
 * Each returns the instruction-count delta for BORROW_COST_ITERS iterations.
 * The loops thread real data dependencies so nothing hoists out of the loop.
 */

/* Empty calibration loop: same shape (memory barrier body), no operation. Its
 * count is the loop-control + bracket overhead, subtracted from the others. */
static unsigned long measure_empty(void) {
  unsigned long i, t0, t1;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_icount();
  return t1 - t0;
}

/* Variant 1 -- RAW pointer (today's zero-copy path). `word` is a plain
 * (delinearised) alias of the arena; each op just dereferences it. */
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

/* Variant 2 -- CAPABILITY BORROW: mint revocation cap + delegate working cap +
 * access + revoke. The reclaimed LINEAR handle that revoke hands back is
 * threaded forward as the next iteration's lend, so the sequence is a valid,
 * repeatable borrow/return lifecycle rather than a one-shot. */
static unsigned long measure_borrow(void *lin) {
  unsigned long i, t0, t1, acc = 0;
  void *c = lin;
  /* Initialise the region through the LINEAR capability before the loop reads
   * it, so the delegated read is not denied as a load through uninitialised
   * capability (cause 26). A store does not delinearise the handle. Not
   * counted -- outside the bracket. */
  *(volatile unsigned long *)lin = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BORROW_COST_ITERS; i++) {
    void *rev = __builtin_capstone_cap_mrev(c);   /* mint revocation cap */
    void *del = __builtin_capstone_cap_delin(c);  /* delegate working cap */
    acc ^= *(volatile unsigned long *)del;        /* access through the lend */
    /* revoke returns the reclaimed capability: with the delegated/working caps
     * non-linear, the sweep retains the data and hands back a fresh LINEAR
     * handle over the region, which becomes the next iteration's lend. */
    c = __builtin_capstone_cap_revoke(rev);
  }
  t1 = rd_icount();
  sink ^= acc;
  return t1 - t0;
}

/* Variant 3 -- COPY baseline (TRANSIENT-style defensive copy). Copy the payload
 * word-by-word into a private buffer, then read one word out. O(payload). */
/* Copy `bytes` (a multiple of 256) in fixed 256-byte __builtin_memcpy chunks.
 * A constant-size memcpy inlines to wide (16-byte ldc/stc) transfers with no
 * libcall -- memcpy-quality, so the copy baseline is not artificially inflated
 * -- and the fixed chunk makes the cost scale cleanly with payload size. */
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

void domain_main(void *arg, unsigned func) {
  if (receive(arg, func))
    return;

  unsigned *res = (unsigned *)arg; /* CALL entry: arg is the retval slot */

  const unsigned long words = BORROW_COST_BUF_BYTES / sizeof(unsigned long);
  unsigned long k;
  for (k = 0; k < words; k++) {
    raw_src[k] = 0x5E5E5E5Eu + k; /* seed the plain source through the ambient cap */
  }

  unsigned long empty = measure_empty();
  unsigned long raw = measure_raw(raw_src);
  unsigned long copy = measure_copy(raw_src, BORROW_COST_COPY_BYTES);
  unsigned long copy2 = measure_copy(raw_src, BORROW_COST_COPY_BYTES_2);
  unsigned long borrow = measure_borrow(arena);

  dbg_count_clear();
  dbg_count_add(BORROW_COST_SLOT_ITERS, BORROW_COST_ITERS);
  dbg_count_add(BORROW_COST_SLOT_EMPTY, empty);
  dbg_count_add(BORROW_COST_SLOT_RAW, raw);
  dbg_count_add(BORROW_COST_SLOT_BORROW, borrow);
  dbg_count_add(BORROW_COST_SLOT_COPY, copy);
  dbg_count_add(BORROW_COST_SLOT_COPY_BYTES, BORROW_COST_COPY_BYTES);
  dbg_count_add(BORROW_COST_SLOT_COPY2, copy2);
  dbg_count_add(BORROW_COST_SLOT_COPY2_BYTES, BORROW_COST_COPY_BYTES_2);
  dbg_count_print();

  *res = BORROW_COST_RET_OK;
}
