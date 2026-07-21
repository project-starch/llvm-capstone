/* Per-primitive borrow-cost BREAKDOWN domain -- GLOBAL-FREE RTL/FPGA variant
 * (2026-07-21). Companion to borrow_cost_fpga_nogp.c.
 *
 * WHY: the borrow-cost table reports the borrow *sequence* (mrev+delin+load+
 * revoke) as one number. This probe attributes that number to its primitives,
 * as far as the platform allows. Two hard facts constrain the decomposition:
 *
 *  (1) `drop`/csdrop is NOT implemented on this core (funct7 0001011 is absent
 *      from the QEMU decode table and there is no helper; the RTL is a subset).
 *      So we CANNOT prune the revocation tree in software -> no "reset the tree
 *      to size-1" trick, and no clean single-op borrow via pruning. That needs
 *      an RTL primitive (implement drop, or auto-release on revoke).
 *
 *  (2) `mrev` + `revoke` WITHOUT `delin` does not round-trip: revoke returns
 *      UNINIT (not a reusable LINEAR cap) because the revoked subtree still held
 *      a *linear* node (cap_rev_tree_revoke: retain_data = !linear). So `delin`
 *      is load-bearing for the reclaim, not optional -- it cannot be removed
 *      from the loop. revoke therefore cannot be isolated from delin.
 *
 * The VALID, round-tripping loops (each verified not to fault under QEMU) are
 * therefore exactly these, and they isolate the primitives this way:
 *
 *   raw        : load only                        -> LOAD
 *   mrev_only  : rev = mrev(c);                   -> MINT (mrev), c stays LIN,
 *                                                    accumulates one node/iter
 *   mrd        : mrev(c); delin(c); revoke(rev);  -> MINT + RECLAIM (delin+revoke)
 *   full       : mrev; delin; load; revoke;       -> the whole borrow sequence
 *
 * Attribution (per op, computed by the controller):
 *   LOAD          = raw
 *   MINT (mrev)   = mrev_only
 *   RECLAIM       = mrd - mrev_only        (delin + revoke, inseparable here)
 *   BORROW total  = full  (= mrev + delin + revoke + load)
 *   cross-check   : full - mrd  should  ~=  raw (the load)
 *
 * Node-budget note: run at a SMALL iteration count. mrev_only accumulates one
 * un-revoked node per iter; mrd/full accumulate one per iter too (revoke
 * invalidates but never releases -- see the ~1024-revoke reset ceiling). Total
 * nodes in one domain call = ITERS*(mrev_only + mrd + full) = ITERS*3. Keep
 * ITERS*3 well under the safe budget (a single 256-iter borrow call exits
 * cleanly; 1024 resets). BREAKDOWN_ITERS=64 -> 192 nodes total: safe.
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

/* mrev-in-isolation requires a loop that mrevs WITHOUT revoking (revoke needs
 * delin to round-trip, so it cannot be dropped -- see header), which accumulates
 * one un-reclaimed revocation node per iter. On SILICON that resets the board
 * (2026-07-21 board run: the domain reset during setup/first-entry; the un-
 * revoked accumulation is the same resource stress as the ~1024-revoke ceiling,
 * reached far earlier because nothing is released and there is no `drop` to
 * prune). So mrev-alone isolation is DISABLED by default: the safe breakdown
 * uses only the proven revoke-per-iteration loops (mrd, full). Set
 * -DBREAKDOWN_WITH_MREV_ONLY=1 to re-enable the (board-resetting) probe. */
#ifndef BREAKDOWN_WITH_MREV_ONLY
#define BREAKDOWN_WITH_MREV_ONLY 0
#endif

/* Region layout (REV_SHARED, 4096 B): [0..63] = 8 result slots. The linear
 * scratch cap (lin_scratch) is carved off the stack top by start-fpga-nogp.S. */

/* ---- breakdown result slots (repurpose the 8-slot region) ---- */
#define BD_SLOT_ITERS      0
#define BD_SLOT_EMPTY      1
#define BD_SLOT_RAW        2  /* load only                     */
#define BD_SLOT_MREV_ONLY  3  /* mrev(c) only  (mint)          */
#define BD_SLOT_MRD        4  /* mrev+delin+revoke (mint+reclaim) */
#define BD_SLOT_FULL       5  /* full borrow                   */
#define BD_SLOT_MAGIC      6  /* 0xB2EA marker so the ctl knows it's breakdown */
#define BD_SLOT_SPARE      7

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

/* LOAD alone: dereference the borrowed word. Byte-identical timed body to the
 * borrow-cost probe's measure_raw. */
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

/* MINT alone: mrev(c) each iteration. c stays LINEAR (mrev does not consume its
 * source), so the loop round-trips; one revocation node accumulates per iter
 * (no revoke). Measures mrev in isolation (plus any per-node accumulation cost). */
#if BREAKDOWN_WITH_MREV_ONLY
static unsigned long measure_mrev_only(void *lin, volatile unsigned long *sinkp) {
  unsigned long i, t0, t1;
  void *c = lin;
  *(volatile unsigned long *)lin = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    /* mrev is IntrInaccessibleMemOnly (read-write) -> preserved through DCE and
     * never CSE'd/hoisted, even with the result unused and c loop-invariant.
     * WARNING: never revokes -> accumulates nodes -> resets the board (see the
     * BREAKDOWN_WITH_MREV_ONLY note at the top). Kept for the record only. */
    (void)__builtin_capstone_cap_mrev(c);
  }
  t1 = rd_icount();
  (void)sinkp;
  return t1 - t0;
}
#endif

/* MINT + RECLAIM: mrev + delin + revoke, NO load. delin is required for revoke
 * to return a reusable LINEAR cap (retain_data). If the compiler wrongly DCE'd
 * delin, revoke would return UNINIT and the next mrev would fault under QEMU --
 * so the QEMU functional run guards this.
 *
 * Threading: the FIRST revoke INVALIDATES *linp's original node, so we must hand
 * the caller the freshly-reclaimed LINEAR cap (revoke's result) for the next
 * measurement to use -- reusing the stale *linp would fault (cap-required). */
static unsigned long measure_mrd(void **linp, volatile unsigned long *sinkp) {
  unsigned long i, t0, t1;
  void *c = *linp;
  *(volatile unsigned long *)c = 0x5E5E5E5Eu;
  t0 = rd_icount();
  for (i = 0; i < BREAKDOWN_ITERS; i++) {
    void *rev = __builtin_capstone_cap_mrev(c);
    (void)__builtin_capstone_cap_delin(c); /* delin (InaccessibleMemOnly): kept;
                                            * required so revoke returns LIN     */
    c = __builtin_capstone_cap_revoke(rev);
  }
  t1 = rd_icount();
  *linp = c;          /* hand back the live reclaimed LINEAR cap */
  (void)sinkp;
  return t1 - t0;
}

/* FULL borrow: mrev + delin + load + revoke. Timed body BYTE-IDENTICAL to
 * borrow_cost_fpga_nogp.c measure_borrow -- this is the number the table reports.
 * Threads the reclaimed LINEAR cap back out like measure_mrd. */
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

  /* Thread the live LINEAR cap through the destructive measurements: raw and
   * mrev_only leave it valid (read-only / never revoked), but measure_mrd and
   * measure_full invalidate the incoming node on their first revoke and hand
   * back the freshly-reclaimed cap, which the next measurement must use. */
  void *lin = lin_scratch;
  unsigned long empty = measure_empty();
  unsigned long raw = measure_raw((volatile unsigned long *)lin, &sink);
#if BREAKDOWN_WITH_MREV_ONLY
  unsigned long mrev_only = measure_mrev_only(lin, &sink);
#else
  unsigned long mrev_only = empty; /* disabled on silicon (resets); mint=0 -> see header */
#endif
  unsigned long mrd = measure_mrd(&lin, &sink);
  unsigned long full = measure_full(&lin, &sink);

  res[BD_SLOT_ITERS] = BREAKDOWN_ITERS;
  res[BD_SLOT_EMPTY] = empty;
  res[BD_SLOT_RAW] = raw;
  res[BD_SLOT_MREV_ONLY] = mrev_only;
  res[BD_SLOT_MRD] = mrd;
  res[BD_SLOT_FULL] = full;
  res[BD_SLOT_MAGIC] = BD_MAGIC;
  res[BD_SLOT_SPARE] = sink; /* force sink escape */
}
