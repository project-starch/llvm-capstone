/* CONTROL, not a defect: a bucket piece read through its old pointer after
 * apr_bucket_free filed it on its allocator's freelist. The piece stays inside
 * its block and the block inside a live node, so on `virtual-malloc` (the
 * pointer adapter over virtual mallocng) the read must COMPLETE; on
 * `virtual-nested-pools` the file revokes the piece (CREVOKE) and the read must
 * FAULT at read_probe. No corpus case files a piece and reads it again, which
 * is why this control exists: it is the one path of the nested bucket adapter
 * the cases do not reach. Run as `buggy 90`. */
#include "../../../shared/corpus.h"

APRB_CASE(90) {
  o->defect_text = "control: a filed bucket piece was read through its old pointer";
  o->fixed_text = "control: no fixed sequence";
  apr_bucket_alloc_t *ba = apr_bucket_alloc_create(root);
  CHECK(ba, 790);
  unsigned char *data = apr_bucket_alloc(64, ba);
  CHECK(data, 791);
  memset(data, 0xA7, 64);
  held = data;
  if (fixed) {
    o->held_up = 1;
    return;
  }
  apr_bucket_free(data);
  o->now = read_probe(held);
  o->defect = 1;
}
