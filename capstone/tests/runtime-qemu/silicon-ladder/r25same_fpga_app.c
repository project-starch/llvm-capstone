/* r25same: the control for r25dup -- the same operand construction (transferred LIN region, cursor
 * moved past end, CAPTYPE in place to UNINIT) and INIT with rd == rs1 (the shape every compiled INIT
 * uses), then the same delin -> stc -> ldc -> tag chain through the result. Proves that the whole
 * chain, including the debug CAPTYPE and INIT's cursor > end precondition, works in a domain on
 * this board, so a fault in r25dup is attributable to its one difference: the store THROUGH rs1
 * after an INIT with rd != rs1. Expected on any hardware: 0x25000001 (under QEMU the CAPTYPE
 * instruction is not decoded, so the QEMU run is a build check only). Interp glue, entry VA 0xD0000. */
#include "../intra-domain-mrev-revoke-probe/intra_domain_mrev_revoke_probe.h"

#define R25_RET_DUP_PRESENT 0x25000001u
#define R25_RET_NO_TAG      0x25000000u
#define R25_REGION_BYTES    4096u
/* The RTL's capability-type numbering is NOT the spec's prose numbering: capstone-ariane
 * verif/tests/custom/capstone/asm_insn.h:77-83 and the cap_type_t enum are NOT_CAP 0, LIN 1, NONLIN 2,
 * REV 3, UNINIT 4, SEALED 5, SEALEDRET 6, EXIT 7 (the spec says linear 0, non-linear 1, uninitialised 3).
 * CAPTYPE takes the low three bits of rs1 as the RTL number. Boot sw39 (2026-09-09) used 3, which made
 * the capability a REVOKE handle, and INIT raised UNEXPECTED_CAP_TYPE (27) as it must. */
#define CAP_TYPE_UNINIT_VAL 4u

static void *r25c_lin;
static unsigned r25c_calls;
static unsigned long r25c_result;

void domain_main(unsigned long *res, unsigned func) {
  if (func == PROBE_DPI_REGION_SHARE) {
    r25c_lin = (void *)res;
    return;
  }
  if (r25c_calls != 0) {
    *res = r25c_result;
    return;
  }
  r25c_calls = 1;
  void *A = r25c_lin;
  void *U = (void *)((char *)A + (R25_REGION_BYTES + 16));  /* cursor past end, still LIN */
  unsigned long ty = CAP_TYPE_UNINIT_VAL;
  __asm__ volatile(".insn r 0x7B, 0x0, 0x5, %0, %1, %0" : "+r"(U) : "r"(ty));   /* CAPTYPE -> UNINIT */
  __asm__ volatile("init %0, %0, %1" : "+r"(U) : "r"(0UL));   /* rd == rs1: U becomes the LIN result */
  void *D = __builtin_capstone_cap_delin(U);                 /* NONLIN alias (consumes U) */
  *(void *volatile *)D = D;                                  /* store the alias through itself at base */
  void *y = *(void *volatile *)D;                            /* load it back */
  r25c_result = __builtin_capstone_cap_get_tag(y) ? R25_RET_DUP_PRESENT : R25_RET_NO_TAG;
  *res = r25c_result;
}
