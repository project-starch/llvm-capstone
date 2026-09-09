/* R-31 sufficiency probe: is the LSU capability-type check INERT for a plain scalar load?
 *
 * WHY THIS EXISTS. R-31 says REVOKE's inverted permission clause skips the reinitialisation step and
 * lets an owner read a borrower's data. Its fix makes revoke return an UNINITIALISED capability, whose
 * whole purpose is that it carries no read authority. But UNINIT only demonstrably blocks LDC
 * (capstone_dyn_unit.anvil:332). A plain scalar `ld` is type-checked in a DIFFERENT place --
 * load_store_unit.sv:975-977 at 66c4e7517 -- inside a block gated on `capmode_i && ld_st_priv_lvl_i ==
 * PRIV_LVL_M` (:947-950). That block was measured INERT in our domains on 2026-08-04 (seven probes; a
 * store through a base carrying no capability metadata, which trips the block's FIRST clause, did not
 * trap) and has NOT been re-measured since, across three bitstreams.
 *
 * If it is still inert, an UNINIT capability in a domain is readable by ordinary loads and R-31's fix
 * does NOT close the disclosure. That is the difference between "necessary" and "sufficient", and it
 * is not something to assume in either direction.
 *
 * THE READING, and both outcomes are informative:
 *   0x31_TT_BB              TT = the type LCC saw, BB = the byte read. TT must be 3 (LCC returns
 *                           cap_type - 1, so RTL UNINIT 4 reads back as 3) and BB must be 0xA5.
 *                           0x3103A5 = the scalar load RETURNED through a capability that was still
 *                           UNINIT at the load -> the check is INERT -> R-31's fix is necessary but
 *                           NOT sufficient. Any other TT means the probe lost its condition and the
 *                           run carries no verdict.
 *   the domain WEDGES       the load trapped (a domain fault is a wedge on this RTL, M-1)
 *                           -> the check is ACTIVE for UNINIT -> R-31's fix does close the read path
 *
 * A wedge is a real result here, not a lost run, which is why this arm goes LAST in its boot.
 *
 * THE CONTROL IS INSIDE THE PROBE, and it is what makes a return meaningful. Before retyping, the
 * SAME address is read by the SAME instruction through the still-LINEAR capability. If that read does
 * not produce the sentinel we planted, the probe is broken and its second reading means nothing --
 * reported as 0x310000FF rather than as a verdict. Without it, "the load returned" is equally
 * consistent with "the load never happened".
 *
 * CAPTYPE (Custom3 funct7 5, in place on rd, no privilege gate) is how the UNINIT capability is made.
 * Type number 4 is the RTL's UNINIT, NOT the spec's 3 -- the spec numbers types with no NOT_CAP and the
 * RTL inserts it at 0 and shifts. Using 3 here would make a REVOKE handle and measure nothing; that
 * exact confusion cost boot sw39.
 */
#include "../intra-domain-mrev-revoke-probe/intra_domain_mrev_revoke_probe.h"

#define LSU_RET_BASE        0x31000000u
#define LSU_RET_CTL_BROKEN  0x310000FFu   /* the pre-retype control read did not see the sentinel */
#define LSU_SENTINEL        0xA5u         /* low byte planted, read back, and returned */
#define CAP_TYPE_UNINIT_VAL 4u            /* RTL numbering; see the header */

static void *lsu_lin;
static unsigned lsu_calls;
static unsigned long lsu_result;

void domain_main(unsigned long *res, unsigned func) {
  if (func == PROBE_DPI_REGION_SHARE) {
    lsu_lin = (void *)res;        /* keep it LINEAR */
    return;
  }
  if (lsu_calls != 0) {
    return;
  }
  lsu_calls = 1;

  void *A = lsu_lin;
  /* Plant a sentinel through the LINEAR capability and read it straight back with the SAME
   * scalar-load shape the probe uses after retyping. This is the positive control. */
  volatile unsigned long *P = (volatile unsigned long *)((char *)A + 64);
  *P = (unsigned long)LSU_SENTINEL;
  unsigned long control = *P;
  if ((control & 0xffu) != LSU_SENTINEL) {
    lsu_result = LSU_RET_CTL_BROKEN;    /* the probe cannot measure anything; do not read on */
    *res = lsu_result;
    return;
  }

  /* Retype the SAME pointer in place to UNINIT. Nothing else changes: same address, same
   * instruction, same region -- the capability's type is the only variable. */
  void *U = (void *)P;
  unsigned long ty = CAP_TYPE_UNINIT_VAL;
  __asm__ volatile(".insn r 0x7B, 0x0, 0x5, %0, %1, %0" : "+r"(U) : "r"(ty));

  /* SECOND CONTROL, and it is what makes a RETURN interpretable. The compiler spills the retyped
   * capability through memory (stc/ldc) before using it, so "the load returned" is also consistent
   * with "the type did not survive the round trip and this was an ordinary LINEAR load". Read the
   * type back with LCC (field 1, the type) immediately before the measurement and carry it in the
   * result, so the reading cannot be mistaken for the thing it is not. rs2 is an immediate encoded
   * in the register field, hence `x1` for field 1. */
  unsigned long ty_read;
  __asm__ volatile(".insn r 0x5B, 0x1, 0x4, %0, %1, x1" : "=r"(ty_read) : "r"(U));

  /* THE MEASUREMENT. A plain scalar load through an UNINIT capability. If the LSU check is live this
   * traps and the domain wedges; if it is inert we return the byte we planted. */
  unsigned long after = *(volatile unsigned long *)U;

  /* 0x31 TT BB : TT = the type LCC saw (expect 3 -- LCC returns cap_type - 1, so RTL UNINIT 4 reads
   * back as 3), BB = the byte read through it. A TT of anything else means the probe did not hold an
   * UNINIT capability at the load and the reading carries NO verdict about the check. */
  lsu_result = LSU_RET_BASE + ((ty_read & 0xffu) << 8) + (after & 0xffu);
  *res = lsu_result;
}
