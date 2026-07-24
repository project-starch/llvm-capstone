/* Per-primitive capability-cost BREAKDOWN domain -- GLOBAL-FREE RTL/FPGA variant
 * (2026-07-21). Companion to borrow_cost_fpga_nogp.c.
 *
 * Times the elementary capability primitives that make up the boundary borrow,
 * plus SHRINK (the spatial-safety bounds-narrowing primitive), each in its own
 * bracketed loop, in one domain entry. Directly measured on silicon:
 *
 *   load       : load only                        -> LOAD
 *   shrink     : shrink(c, base, base+64)         -> SHRINK  (idempotent after 1)
 *   mrev_only  : rev = mrev(c);                   -> MREV    (c stays LIN)
 *   mrd        : mrev(c); delin(c); revoke(rev);  -> MREV + DELIN + REVOKE
 *   full       : mrev; delin; load; revoke;       -> the whole borrow
 *
 * Derived by the controller:  DELIN+REVOKE = mrd - mrev_only ;  BORROW = full.
 *
 * Two ISA facts fix what is separable:
 *  - `delin` is load-bearing for the reclaim: mrev+revoke WITHOUT delin returns
 *    UNINIT (cap_rev_tree_revoke keeps data only for a non-linear subtree), so it
 *    cannot loop. Hence delin and revoke are measured together, never apart.
 *  - `drop`/csdrop is not implemented on this core, so the revocation tree cannot
 *    be pruned in software; each mrev leaks a node for the domain-call lifetime.
 *
 * Node budget (one domain call): mrev_only accumulates BREAKDOWN_MREV_ITERS
 * un-revoked nodes; mrd and full accumulate BREAKDOWN_ITERS each (revoke
 * invalidates but never releases). Total = MREV_ITERS + 2*ITERS. With 16 + 2*64
 * = 144 this stays well under the ~256 that exits cleanly (1024 resets the board).
 * mrev_only uses a SMALL count because it never revokes.
 *
 * Same single-REGION_SHARE, gp-free, plain-call/ret ABI as borrow_cost_fpga_nogp
 * (start-fpga-nogp.S). All measure_* inline into domain_main (leaf: one cjalr
 * return, retargeted to plain ret by the build script).
 */
#include "../runtime-qemu/borrow-cost-probe/borrow_cost_probe.h"
#include "fpga_instrument.h"

#ifndef BREAKDOWN_ITERS
#define BREAKDOWN_ITERS 64UL
#endif
/* mrev_only never revokes -> one leaked node per iter; keep it small. */
#ifndef BREAKDOWN_MREV_ITERS
#define BREAKDOWN_MREV_ITERS 16UL
#endif
/* SHRINK target window (bytes) carved from the linear scratch base; the borrow
 * only touches 8 bytes, so a small window is safe and never faults. */
#define BREAKDOWN_SHRINK_WIN 64UL

/* ---- breakdown result slots (repurpose the 8-slot region) ---- */
#define BD_SLOT_ITERS      0  /* BREAKDOWN_ITERS (the count for all but mrev)   */
#define BD_SLOT_EMPTY      1
#define BD_SLOT_RAW        2  /* load only                                      */
#define BD_SLOT_SHRINK     3  /* shrink                                         */
#define BD_SLOT_MREV       4  /* mrev only, TOTAL over BREAKDOWN_MREV_ITERS      */
#define BD_SLOT_MRD        5  /* mrev+delin+revoke                              */
#define BD_SLOT_FULL       6  /* full borrow                                    */
#define BD_SLOT_MAGIC      7  /* 0xB2EA marker                                  */
#define BD_MAGIC 0xB2EAUL

/* ==== measured operations -- all bracketed by an mcycle read ==== */

static unsigned long measure_empty(void) {
  unsigned long i, t0, t1;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    __asm__ volatile("" : : : "memory");
  }
  t1 = rd_icount();
  return t1 - t0;
}

/* LOAD alone. Byte-identical timed body to the borrow-cost probe's measure_raw. */
static unsigned long measure_raw(volatile unsigned long *word,
                                 volatile unsigned long *sinkp) {
  unsigned long i, t0, t1, acc = 0;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    acc ^= word[0];
  }
  t1 = rd_icount();
  *sinkp ^= acc;
  return t1 - t0;
}

/* SHRINK: narrow the linear cap to [base, base+WIN]. The first iteration
 * narrows; the rest re-shrink to the same window (base>=base, end<=end -> valid,
 * idempotent) so every iteration executes one shrink instruction. No tree node
 * is allocated. The narrowed cap is threaded back (bounds change is permanent),
 * and is still valid for the 8-byte borrow that follows. */
static unsigned long measure_shrink(void **linp) {
  unsigned long i, t0, t1;
  void *c = *linp;
  unsigned long b = (unsigned long)__builtin_capstone_cap_get_base(c);
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    c = __builtin_capstone_cap_shrink(c, b, b + BREAKDOWN_SHRINK_WIN);
  }
  t1 = rd_icount();
  *linp = c;
  return t1 - t0;
}

/* MREV alone: mrev(c) each iteration. c stays LINEAR (mrev does not consume its
 * source), so the loop round-trips; one revocation node accumulates per iter
 * (small count). Measures mrev in isolation. mrev is IntrInaccessibleMemOnly
 * (read-write) -> preserved through DCE, never CSE'd/hoisted even with the result
 * unused and c loop-invariant. */
static unsigned long measure_mrev_only(void *lin) {
  unsigned long i, t0, t1;
  void *c = lin;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_MREV_ITERS; i++) {
    (void)__builtin_capstone_cap_mrev(c);
  }
  t1 = rd_icount();
  return t1 - t0;
}

/* MREV + DELIN + REVOKE (the reclaim unit), NO load. The first revoke invalidates
 * *linp's node, so the freshly-reclaimed LINEAR cap is threaded back out. */
static unsigned long measure_mrd(void **linp) {
  unsigned long i, t0, t1;
  void *c = *linp;
  *(volatile unsigned long *)c = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    void *rev = __builtin_capstone_cap_mrev(c);
    (void)__builtin_capstone_cap_delin(c);   /* load-bearing: revoke needs it */
    c = __builtin_capstone_cap_revoke(rev);
  }
  t1 = rd_icount();
  *linp = c;
  return t1 - t0;
}

/* FULL borrow: mrev + delin + load + revoke. Timed body BYTE-IDENTICAL to
 * borrow_cost_fpga_nogp.c measure_borrow. Threads the reclaimed cap back out. */
static unsigned long measure_full(void **linp, volatile unsigned long *sinkp) {
  unsigned long i, t0, t1, acc = 0;
  void *c = *linp;
  *(volatile unsigned long *)c = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    void *rev = __builtin_capstone_cap_mrev(c);
    void *del = __builtin_capstone_cap_delin(c);
    acc ^= *(volatile unsigned long *)del;
    c = __builtin_capstone_cap_revoke(rev);
  }
  t1 = rd_icount();
  *linp = c;
  *sinkp ^= acc;
  return t1 - t0;
}

/* ==== end measured operations ==== */

void domain_main(void *arg, unsigned func, void *lin_scratch) {
  volatile unsigned long sink = 0;
  unsigned char *region = (unsigned char *)arg;
  volatile unsigned long *res = (volatile unsigned long *)region;

  if (func != BORROW_COST_DPI_REGION_SHARE) {
    return;
  }

  /* Thread the live LINEAR cap through the measurements. raw reads it at full
   * bounds; shrink narrows it (permanent, threaded); mrd/full reclaim it and hand
   * back the fresh cap. mrev_only leaves it LINEAR (never revoked). */
  void *lin = lin_scratch;
  unsigned long empty = measure_empty();
  unsigned long raw = measure_raw((volatile unsigned long *)lin, &sink);
  unsigned long shrink = measure_shrink(&lin);
  unsigned long mrev_only = measure_mrev_only(lin);
  unsigned long mrd = measure_mrd(&lin);
  unsigned long full = measure_full(&lin, &sink);

  res[BD_SLOT_ITERS] = BREAKDOWN_ITERS;
  res[BD_SLOT_EMPTY] = empty;
  res[BD_SLOT_RAW] = raw;
  res[BD_SLOT_SHRINK] = shrink;
  res[BD_SLOT_MREV] = mrev_only;   /* TOTAL over BREAKDOWN_MREV_ITERS */
  res[BD_SLOT_MRD] = mrd;
  res[BD_SLOT_FULL] = full;
  res[BD_SLOT_MAGIC] = BD_MAGIC;
  (void)sink;   /* volatile: writes in measure_* are already kept */
}
