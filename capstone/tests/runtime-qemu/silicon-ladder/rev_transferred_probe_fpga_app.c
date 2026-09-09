/* rev_transferred_probe: the first transfer-annotated share on silicon (Phase B item 2,
 * docs/plans/monitor-unification.md; Q-05 close-out in ISSUES.md).
 *
 * The host creates one region, TRANSFERS it (annotation PERM_INOUT + REV_TRANSFERRED: the
 * monitor hands the domain the LINEAR capability and keeps no revocation handle), then calls
 * the domain twice. Entry 1 is the share itself: func == PROBE_DPI_REGION_SHARE and `res` IS the
 * delivered capability. Entry 2 (first CALL) writes the sentinel through the domain's alias and
 * returns PROBE_RET_NO_REVOKE_OK. Entry 3 (second CALL) reads the byte back through the same
 * alias and returns 0x22400000 | byte -- the observer is the domain, never the host's mmap,
 * which after a transfer has no authority (Q-05) and on the board reaches cap_base(null) in
 * M-mode if it ever faults on those pages.
 *
 * Two things the QEMU probe framework (probe_domain.h) gets away with and silicon does not,
 * which is why this file does not include it:
 *   - it parks the LINEAR capability and reloads it on every call: a linear ldc DUPLICATES on
 *     QEMU but MOVES on silicon (nulls its source, Q-04 family), so the second call would load
 *     an untagged slot. Here the capability is DELINEARISED ONCE, at receipt, and the NONLIN
 *     alias is what survives in the global.
 *   - its glue keeps globals across entries; the generated silicon glue rebuilds every global
 *     on re-entry. This image must be built with DOMAIN_GLUE=interp (build script), the glue
 *     SQLite proves for two REGION_SHARE entries on the board.
 * Consequences: this probe cannot MREV the arena (mrev needs LIN), and the arena stays owned by
 * the domain -- exactly the "receive / use / return / read back" shape item 2 needs.
 * Silicon config: -capstone-gp-captable, shrink off, +m (build-ladder-domain.sh). Two globals ->
 * the static gate sees `ldc ...(gp)`. */
#include "../intra-domain-mrev-revoke-probe/intra_domain_mrev_revoke_probe.h"

#define REVXFER_RET_READBACK 0x22400000u   /* | the byte read back on the second CALL */

static void *probe_arena;      /* the NONLIN alias of the transferred region */
static unsigned probe_calls;   /* CALL entries seen */

void domain_main(unsigned long *res, unsigned func) {
  if (func == PROBE_DPI_REGION_SHARE) {
    /* delin ONCE: from here on every load of probe_arena duplicates on both targets */
    probe_arena = __builtin_capstone_cap_delin((void *)res);
    return;
  }
  volatile unsigned char *buf = (volatile unsigned char *)probe_arena;
  if (probe_calls == 0) {
    probe_calls = 1;
    buf[PROBE_OFFSET] = (unsigned char)PROBE_SENTINEL_LIVE;
    *res = PROBE_RET_NO_REVOKE_OK;                 /* 0x22300000 */
    return;
  }
  *res = REVXFER_RET_READBACK | buf[PROBE_OFFSET]; /* expected 0x2240005e */
}
