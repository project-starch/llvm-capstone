#include "corpus.h"

/* The resend-detection record kept per call, as far as this case needs it. */
struct sip_hash_value {
  unsigned cseq;
  unsigned char *method;
};

WM_CASE(8) {
/* Row 8 -- SIP, #18735, fix 31ab1a0a17.
 * A fix for string truncation turned the CSeq method from an inline buffer
 * in the file-scope resend record into a pointer, and stored the pinfo->pool
 * string proto_tree_add_item_ret_string returned. The pool is reset between
 * packets; the next SIP packet's resend check compares against it. */
  struct sip_hash_value *p_val =
      wmem_alloc(wm_file_scope(), sizeof *p_val); /* wmem_new0(wmem_file_scope()), packet-sip.c:5338 */
  CHECK(p_val, 1);
  unsigned char *cseq_method = wmem_alloc(wm_packet, 48); /* ret_string(..., pinfo->pool, ...), :4096 */
  CHECK(cseq_method, 2);
  memset(cseq_method, 0, 48);
  memcpy(cseq_method, "INVITE", 7);
  p_val->cseq = 1;
  p_val->method = cseq_method; /* :5352 */
  wm_next_packet(); /* epan_dissect_reset, epan.c:591 */
  /* The next SIP packet checks whether it is a resend. */
  wm_held = p_val->method;
  wm_mark();
  (void)wm_probe(wm_held); /* strcmp(cseq_method, p_val->method) in sip_is_packet_resend, :5375 */
}
