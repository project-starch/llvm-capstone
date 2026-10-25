/* CONTROL, not a defect: a byte palloc'd from a pool, read through its old
 * pointer after the pool was destroyed. APR files the pool's node on its own
 * free list and never calls free(), so on `virtual-malloc` (the stock adapter,
 * every node one virtual-mallocng object) the read must COMPLETE; on
 * `virtual-nested-pools` the node's release revokes the child the pool held
 * (CREVOKE) and the read must FAULT at read_probe. It is case 0's destroy
 * without case 0's reuse story, run as `buggy 90`. */
#include "../../../shared/corpus.h"

APR_CASE(90) {
  o->defect_text = "control: a destroyed pool's byte was read through its old pointer";
  o->fixed_text = "control: no fixed sequence";
  apr_pool_t *p = NULL;
  CHECK(apr_pool_create(&p, root) == APR_SUCCESS, 790);
  unsigned char *b = apr_palloc(p, 64);
  CHECK(b, 791);
  memset(b, 0xA7, 64);
  held = b;
  if (fixed)
    return;
  apr_pool_destroy(p);
  (void)read_probe(held);
  o->pool_struct_reissued = 1;
  o->allocated_through_stale = 1;
}
