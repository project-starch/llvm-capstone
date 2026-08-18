/* QEMU arm. Arms 0/1/3/4 are QEMU-safe and expected to return 0 lost: QEMU's capability store
 * is one atomic 16-byte-plus-tag operation with no write buffer, no per-word entries and no
 * drain arbiter, so the reordering under test CANNOT occur under emulation. That makes QEMU
 * the no-reorder baseline rather than a weaker copy of the board -- if silicon and QEMU differ
 * on arm 1, the difference IS the mechanism.
 *
 * ARM 2 IS NOT BUILDABLE FOR QEMU. It clears the tag deliberately, and op_helper.c:719 asserts
 * rs1_v->tag before any selector check, so the type query aborts where silicon returns 7. Its
 * board-only status is the same divergence tagsweep records for its SEED arm. */
#include "wbuf_kernel.h"
void domain_main(unsigned *res, unsigned func) { (void)func; *res = wbuf_compute(); }
