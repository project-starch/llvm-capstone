#include "corpus.h"

WM_CASE(6) {
/* Row 6 -- x509if under EAP, #18622, fix a8b16d74e1.
 * dissect_x509if_RDNSequence builds the distinguished name in a pinfo->pool
 * string buffer and keeps it in the file-level static last_dn_buf, which a
 * frame-end routine is meant to clear. EAP handed its sub-dissectors a stack
 * COPY of pinfo, so the routine was registered on the copy and never ran on
 * the real one. The pool is reset between packets; a later SubjectName
 * appends the stale buffer to its item text with "%s". */
  static unsigned char *last_dn_buf; /* packet-x509if.c:279 */
  unsigned char *dn = wmem_alloc(wm_packet, 2600); /* wmem_strbuf_new(actx->pinfo->pool, ""), :911 */
  CHECK(dn, 1);
  memset(dn, 'C', 2599);
  dn[2599] = 0;
  last_dn_buf = dn;
  /* register_frame_end_routine(actx->pinfo, x509if_frame_end), :912 -- on the
   * copy, so nothing clears last_dn_buf when the packet ends. */
  wm_next_packet(); /* epan_dissect_reset, epan.c:591 */
  /* THE FIX, a8b16d74e1: EAP no longer hands its sub-dissectors a stack copy of pinfo, so the
   * frame-end routine is registered on the real one and clears last_dn_buf when the packet ends. */
  if (wm_fixed)
    last_dn_buf = NULL;
  wm_reoccupy(dn, 2600); /* native observer only */
  wm_held = last_dn_buf; /* x509if_get_last_dn(), read by "%s" at x509af:323-324 */
  if (!wm_held)
    return; /* the fix: the frame-end routine ran, nothing stale to append */
  wm_mark();
  WM_READ(wm_held, WM_MARKER);
}
