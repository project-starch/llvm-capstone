#include "regions.h"
#include <string.h>
struct buffer {
  struct wg_region region;
  struct ggml_context *ctx;
  unsigned owned, seen, allocated;
};
struct descriptor {
  struct wg_region region;
  unsigned live;
};
static struct buffer buffers[WG_BUFFERS];
static struct descriptor descriptors[WG_CONTEXTS];
static unsigned selected, temporal;
static uint64_t rebinds;
void wg_init_backing(void *metadata, void *payload, unsigned mode) {
  (void)metadata;
  if (mode > 1)
    wg_fail(203);
  temporal = mode;
  wg_regions_init(payload);
}
void wg_select_buffer(unsigned id) {
  if (id >= WG_BUFFERS)
    wg_fail(204);
  selected = id;
}
static void *prepare(unsigned id, size_t n, unsigned owned) {
  if (!n || n > WG_PAYLOAD_BYTES || n > SIZE_MAX - 15)
    wg_fail(205);
  struct buffer *b = &buffers[id];
  if (b->ctx || (owned && b->allocated && b->owned))
    wg_fail(206);
  size_t rounded = (n + 15) & ~(size_t)15;
  if (b->region.size && rounded > b->region.size) {
    wg_region_renew(&b->region, temporal);
    memset(&b->region, 0, sizeof b->region);
  }
  if (!b->region.size)
    wg_region_create(&b->region, rounded);
  b->owned = owned;
  b->allocated = 1;
  return b->region.alias;
}
void *wg_borrow_buffer(unsigned id, size_t n) {
  wg_select_buffer(id);
  return prepare(id, n, 0);
}
void *wg_aligned_alloc(size_t n) { return prepare(selected, n, 1); }
static struct buffer *find_buffer(void *p) {
  for (unsigned i = 0; i < WG_BUFFERS; ++i)
    if (buffers[i].allocated && wg_same_authority(p, buffers[i].region.alias))
      return &buffers[i];
  wg_fail(207);
}
void wg_aligned_free(void *p, size_t n) {
  struct buffer *b = find_buffer(p);
  if (!b->owned || b->ctx || n > b->region.size)
    wg_fail(208);
  wg_region_renew(&b->region, temporal);
  b->allocated = 0;
}
void *wg_bind(struct ggml_context *ctx, void *p, size_t n, int owned) {
  struct buffer *b = find_buffer(p);
  if (b->ctx || n > b->region.size || b->owned != (unsigned)owned)
    wg_fail(209);
  if (b->seen) {
    wg_region_renew(&b->region, temporal);
    ++rebinds;
  }
  b->seen = 1;
  b->ctx = ctx;
  return b->region.alias;
}
void *wg_reset_buffer(struct ggml_context *ctx, void *p) {
  struct buffer *b = find_buffer(p);
  if (b->ctx != ctx)
    wg_fail(210);
  wg_region_renew(&b->region, temporal);
  return b->region.alias;
}
void wg_unbind(struct ggml_context *ctx, void *p, int owned) {
  struct buffer *b = find_buffer(p);
  if (b->ctx != ctx || b->owned != (unsigned)owned)
    wg_fail(211);
  /* Borrowed graph storage remains live after its descriptor is destroyed. */
  b->ctx = NULL;
}
void *wg_meta_alloc(size_t n) {
  if (n > 128)
    wg_fail(212);
  for (unsigned i = 0; i < WG_CONTEXTS; ++i) {
    struct descriptor *d = &descriptors[i];
    if (!d->live) {
      if (!d->region.size)
        wg_region_create(&d->region, 128);
      d->live = 1;
      return d->region.alias;
    }
  }
  wg_fail(213);
}
void wg_meta_free(void *p) {
  for (unsigned i = 0; i < WG_CONTEXTS; ++i) {
    struct descriptor *d = &descriptors[i];
    if (d->live && wg_same_authority(p, d->region.alias)) {
      d->live = 0;
      wg_region_renew(&d->region, temporal);
      return;
    }
  }
  wg_fail(214);
}
void wg_backing_stats(struct wg_header *out) { out->rebinds = rebinds; }
