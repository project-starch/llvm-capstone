#include "corpus.h"

/* The reassembly table's fragment head, as far as this case needs it. */
struct fd_head {
  unsigned char *tvb_data;
  unsigned len;
};

WM_CASE(10) {
/* Row 10 -- T.38, #19695, fix 6fd3af5e99.
 * force_reassemble_seq builds the forced-reassembly buffer in pinfo->pool and
 * wraps it in a tvb that the persistent reassembly table keeps. The pool is
 * reset between packets; a later frame's fragment_add_seq compares its bytes
 * against the stored buffer through tvb_memeql. */
  struct fd_head *head = wmem_alloc(wm_file_scope(), sizeof *head); /* the table */
  CHECK(head, 1);
  unsigned char *data = wmem_alloc(wm_packet, 1); /* wmem_alloc(pinfo->pool, size), packet-t38.c:358 */
  CHECK(data, 2);
  data[0] = 0x4c;
  head->tvb_data = data; /* fd_head->tvb_data = tvb_new_real_data(data, ...), :359 */
  head->len = 1;
  wm_next_packet(); /* epan_dissect_reset, epan.c:602 */
  /* The next T.38 frame retrieves the head from the table. */
  wm_held = head->tvb_data;
  wm_mark();
  (void)wm_probe(wm_held); /* memcmp in tvb_memeql, reassemble.c:2064 */
}
