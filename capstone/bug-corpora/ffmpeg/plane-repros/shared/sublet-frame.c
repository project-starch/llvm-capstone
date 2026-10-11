/* THE SUBLET PORT OF THE FRAME CARVE, FFP_SUBLET_CARVE (added 2026-10-09; Capstone, Sublet heap).
 *
 * av_frame_get_buffer cuts ONE buffer into a frame's planes (libavutil/frame.c, get_video_buffer).
 * This is that carve ported to Sublet, linked with -Wl,--wrap=av_frame_get_buffer so the case calls
 * it unchanged: the buffer is lent LINEAR by the Sublet heap and split into one Sublet region per
 * plane, and each plane pointer is an alias bounded to its plane. Freeing the frame is the heap's
 * single revoke of the block.
 *
 * WHERE A PLANE ENDS is upstream's, not ours: the layout -- each plane's offset in the buffer, its
 * linesize, the buffer's size -- is learned by calling the real av_frame_get_buffer on a scratch
 * frame of the same format and size, and a plane's region is its allocated extent,
 * av_image_fill_plane_sizes at FFALIGN(height, 32), exactly what get_video_buffer reserves for it.
 * A port that narrowed to `linesize * height` instead would fault legitimate code that works over
 * padded_height (SIMD tails, av_image_copy_to_buffer); see the corpus README. */
#ifndef FFP_SUBLET_CARVE
#error "sublet-frame.c is the plane corpus's sublet-carve arm only"
#endif
#include "corpus.h"
#include <sublet/sublet.h>

unsigned long __capstone_sublet_malloc_linear(size_t n, capstone_cap_slot *out);
void __capstone_sublet_free_linear(unsigned long base);
int __real_av_frame_get_buffer(AVFrame *frame, int align);

/* One frame at a time, which is what every case holds. */
static struct {
  int live;
  unsigned long base;
  capstone_cap_slot rest, plane[4], gap[4];
} sf;

static void sublet_frame_free(void *opaque, uint8_t *data) {
  (void)opaque;
  (void)data;
  CHECK(sf.live, 951);
  __capstone_sublet_free_linear(sf.base); /* one revoke: every plane's alias dies */
  sf.live = 0;
}

int __wrap_av_frame_get_buffer(AVFrame *frame, int align) {
  CHECK(!sf.live && frame->width > 0 && frame->height > 0 && !frame->linesize[0], 952);
  /* Upstream's layout, from upstream: the same request on a scratch frame. */
  AVFrame *probe = av_frame_alloc();
  CHECK(probe, 953);
  probe->format = frame->format;
  probe->width = frame->width;
  probe->height = frame->height;
  int ret = __real_av_frame_get_buffer(probe, align);
  if (ret < 0) {
    av_frame_free(&probe);
    return ret;
  }
  CHECK(probe->buf[0] && !probe->buf[1], 954);
  size_t total = probe->buf[0]->size, sizes[4];
  ptrdiff_t linesizes[4];
  unsigned long off[4];
  int present[4];
  for (int i = 0; i < 4; i++) {
    linesizes[i] = probe->linesize[i];
    present[i] = probe->data[i] != NULL;
    off[i] = present[i] ? (unsigned long)(probe->data[i] - probe->buf[0]->data) : 0;
  }
  CHECK(av_image_fill_plane_sizes(sizes, frame->format, FFALIGN(frame->height, 32), linesizes) >= 0,
        955);
  av_frame_free(&probe);

  /* The same layout, lent LINEAR and split: [gap][plane 0][gap][plane 1]... on 16-byte bounds. */
  sf.base = __capstone_sublet_malloc_linear(total, &sf.rest);
  CHECK(sf.base, 956);
  sf.live = 1;
  unsigned long cur = sf.base;
  for (int i = 0; i < 4; i++) {
    if (!present[i])
      continue;
    unsigned long s = sf.base + off[i], e = s + sizes[i];
    CHECK(s >= cur && !(s & 15) && !(e & 15) && e <= sf.base + total, 957);
    if (s > cur)
      sublet_carve(&sf.rest, s, &sf.gap[i]);
    sublet_carve(&sf.rest, e, &sf.plane[i]);
    unsigned char *a = sublet_take(&sf.plane[i]);
    a += s - __builtin_capstone_cap_get_cursor(a);
    frame->data[i] = __builtin_capstone_cap_shrink(a, s, e);
    frame->linesize[i] = (int)linesizes[i];
    cur = e;
    printf("sublet-frame plane %d region=[%lu,%lu) linesize=%d\n", i, off[i], off[i] + sizes[i],
           frame->linesize[i]);
  }
  frame->buf[0] = av_buffer_create(frame->data[0], total, sublet_frame_free, NULL, 0);
  CHECK(frame->buf[0], 958);
  frame->extended_data = frame->data;
  return 0;
}
