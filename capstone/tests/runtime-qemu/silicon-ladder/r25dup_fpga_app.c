/* r25dup: the R-25 board probe -- does INIT with rd != rs1 leave a usable duplicate in rs1?
 *
 * R-25 (ISSUES.md): the RTL's INIT writes the NEW linear capability to BOTH rs1 and rd when they
 * differ (capstone_flu_unit.anvil:147, unguarded); the spec (INIT = MOVC + set type/cursor) nulls
 * rs1. Two live linear capabilities to one region. Silicon-only: QEMU nulls rs1.
 *
 * Operand construction, exactly the RTL lane's self-checking test (init-rs1-ne-rd.S): INIT accepts
 * only an UNINIT capability whose cursor is STRICTLY past its end (spec cap-man-insn.adoc "Illegal
 * operand value: cursor <= end"; anvil :139-141), which no initialising store can produce (each
 * advances the cursor by 16 and the last lands exactly ON end) and which revoke() does not produce
 * either (cursor = start on RTL, end on QEMU). So: take the TRANSFERRED linear region A (4 KiB),
 * move its cursor past the end (allowed on LIN: cincoffset does not check bounds), then retype it
 * IN PLACE to UNINIT with the Capstone debug instruction CAPTYPE (custom-3, funct7 5; no privilege
 * gate in decoder.sv, bounds and cursor survive; QEMU does not decode it, so under QEMU this image
 * halts at that instruction and the QEMU run is a BUILD check only). Then INIT(X, U, 0) with X != U
 * (inline asm; "=&r" makes rd a different register, "+r"(U) makes the -O0 spill keep the register
 * value of U AFTER the INIT rather than the pre-INIT copy from its stack slot).
 *
 * The non-trapping detector (the RTL lane's phase 4): store a capability THROUGH U (the rs1 operand)
 * and load it back through the region. On R-25 silicon U is a second LINEAR capability with
 * cursor == base (new_cursor = val + start, val = 0), the store lands in the region's first 16
 * bytes, the load through D at base returns a tagged capability -> retval 0x25000001. On
 * conformant hardware U is NOT_CAP after INIT and the store traps (UNEXPECTED_OPERAND); the monitor
 * returns the fault to the host, which is the "consumed" reading. Prediction on
 * caplifive_s12fix_5097eb166 (RTL lane, from records/r25/test-prefix.log): 0x25000001; on the
 * R-25-fixed bitstream: a trap at the store through U.
 *
 * Idempotent across CALL entries (the host program rtpc calls twice and reports the second): the
 * first CALL computes and caches, later CALLs return the cache. Built with the interp glue
 * (globals survive re-entry) like rev_transferred_probe; the LINEAR capability is loaded from its
 * global exactly once (a linear ldc MOVES on silicon). Entry VA 0xC0000. */
#include "../intra-domain-mrev-revoke-probe/intra_domain_mrev_revoke_probe.h"

#define R25_RET_DUP_PRESENT 0x25000001u
#define R25_RET_NO_TAG      0x25000000u   /* the store landed but the load came back untagged */
#define R25_REGION_BYTES    4096u
/* The RTL's capability-type numbering is NOT the spec's prose numbering: capstone-ariane
 * verif/tests/custom/capstone/asm_insn.h:77-83 and the cap_type_t enum are NOT_CAP 0, LIN 1, NONLIN 2,
 * REV 3, UNINIT 4, SEALED 5, SEALEDRET 6, EXIT 7 (the spec says linear 0, non-linear 1, uninitialised 3).
 * CAPTYPE takes the low three bits of rs1 as the RTL number. Boot sw39 (2026-09-09) used 3, which made
 * the capability a REVOKE handle, and INIT raised UNEXPECTED_CAP_TYPE (27) as it must. */
#define CAP_TYPE_UNINIT_VAL 4u

static void *r25_lin;          /* the transferred LINEAR region capability, loaded ONCE */
static unsigned r25_calls;
static unsigned long r25_result;

void domain_main(unsigned long *res, unsigned func) {
  if (func == PROBE_DPI_REGION_SHARE) {
    r25_lin = (void *)res;      /* keep it LINEAR */
    return;
  }
  if (r25_calls != 0) {
    *res = r25_result;
    return;
  }
  r25_calls = 1;
  void *A = r25_lin;                                        /* the one load of the linear capability */
  void *U = (void *)((char *)A + (R25_REGION_BYTES + 16));  /* cursor past end, still LIN */
  unsigned long ty = CAP_TYPE_UNINIT_VAL;
  /* CAPTYPE in place: .insn r opcode=0x7B funct3=0 funct7=5, rd=U, rs1=type, rs2=U */
  __asm__ volatile(".insn r 0x7B, 0x0, 0x5, %0, %1, %0" : "+r"(U) : "r"(ty));
  void *X;
  __asm__ volatile("init %0, %1, %2" : "=&r"(X), "+r"(U) : "r"(0UL));   /* rd != rs1 by =& */
  void *D = __builtin_capstone_cap_delin(X);                /* NONLIN alias of the region (consumes X) */
  *(void *volatile *)U = D;                                 /* STC through rs1: R-25 -> lands; conformant -> trap */
  void *y = *(void *volatile *)D;                           /* LDC back through the alias at base */
  r25_result = __builtin_capstone_cap_get_tag(y) ? R25_RET_DUP_PRESENT : R25_RET_NO_TAG;
  *res = r25_result;
}
