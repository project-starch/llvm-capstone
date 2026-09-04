/* QEMU arm for the S-12 minimal repro.
 *
 * Arms 0, 1 and 3 are QEMU-safe and expected to return bad == 0. QEMU's capability load and
 * store are atomic 16-byte-plus-tag operations with no write buffer, no per-word entries, no
 * drain arbiter and no forwarding network, so none of the mechanisms under investigation can
 * occur under emulation. That makes QEMU the NO-DEFECT BASELINE rather than a weaker copy of
 * the board: if silicon and QEMU differ on arm 1, the difference IS the mechanism.
 *
 * ARM 2 IS NOT BUILDABLE FOR QEMU, for the same reason wbuf's arm 2 is not: it clears the tag
 * deliberately, and op_helper.c:719 asserts rs1_v->tag before any selector check, so the type
 * query ABORTS under emulation where silicon returns 7. Arm 2 is therefore board-only. That is
 * a documented divergence, not a gap in this test -- and it is why arm 2's positive control can
 * only be validated on silicon. */
#include "s12_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = s12_compute(); }
