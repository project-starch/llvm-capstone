/* The chunk port's authority on a hosted build: no capabilities, so a block is
 * one allocation and every chunk is an offset into it. The allocator above is
 * the same code as in the domain, which is what lets the native replay check
 * the header move against unmodified upstream. See src/shared/chunks.h. */
#include "chunks.h"
static struct wm_chunk_counts counts = {.magic = WM_CHUNK_COUNTS_MAGIC};
void wm_chunks_init(unsigned temporal) { (void)temporal; }
void wm_block_open(struct wm_block_auth *b, size_t size, size_t hdr) {
  (void)hdr;
  ++counts.opens;
  b->size = size;
  b->alias = wm_sys_alloc(size);
  b->base = wm_addr(b->alias);
}
void wm_block_reset(struct wm_block_auth *b, size_t hdr, size_t dropped) {
  (void)b;
  (void)hdr;
  ++counts.resets;
  counts.dropped += dropped;
}
void wm_chunk_adopt(struct wm_block_auth *b, struct wm_chunk_auth *first) {
  (void)b;
  first->wide = NULL;
}
void wm_block_close(struct wm_block_auth *b) {
  ++counts.closes;
  wm_sys_free(b->alias);
  b->alias = NULL;
}
void wm_chunk_split(struct wm_chunk_auth *c, size_t at,
                    struct wm_chunk_auth *upper) {
  (void)c;
  (void)at;
  ++counts.splits;
  upper->wide = NULL;
}
void wm_chunk_issue(struct wm_block_auth *b, struct wm_chunk_auth *c,
                    size_t base, size_t len) {
  (void)len;
  ++counts.issues;
  c->wide = (char *)b->alias + (base - b->base);
}
void *wm_chunk_bytes(struct wm_chunk_auth *c, size_t at, size_t n) {
  (void)n;
  return (char *)c->wide + (at - wm_addr(c->wide));
}
void wm_chunk_retire(struct wm_chunk_auth *c) {
  (void)c;
  ++counts.retires;
}
void wm_chunk_forget(struct wm_chunk_auth *c) { c->wide = NULL; }
void wm_block_forget(struct wm_block_auth *b) { b->alias = NULL; }
void wm_chunk_report(void *report_page) { (void)report_page; }
