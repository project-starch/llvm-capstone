/* THE FRAME CARVE ON MANAGED LIFETIMES, FFP_VIRTUAL_CARVE (virtual Capstone, CDERIVE/CREVOKE).
 *
 * av_frame_get_buffer cuts ONE buffer into a frame's planes (libavutil/frame.c, get_video_buffer).
 * This is that carve on the virtual profile's lifetime instructions, linked with
 * -Wl,--wrap=av_frame_get_buffer so the case calls it unchanged: the buffer comes from the process's
 * malloc (virtual mallocng), one child lifetime of it spans the frame (CDERIVE), and each plane
 * pointer is a child of that, bounded to its plane. Freeing the frame frees the buffer, which ends
 * the frame's lifetime and every plane's. It replaces the physical port (sublet-frame.c, a block
 * lent LINEAR by the Sublet heap and split), which went with that heap.
 *
 * WHERE A PLANE ENDS is upstream's, not ours, exactly as in the physical port: the layout -- each
 * plane's offset in the buffer, its linesize, the buffer's size -- is learned by calling the real
 * av_frame_get_buffer on a scratch frame of the same format and size, and a plane's region is its
 * allocated extent, av_image_fill_plane_sizes at FFALIGN(height, 32), what get_video_buffer
 * reserves for it. A port that narrowed to `linesize * height` would fault legitimate code that
 * works over padded_height (SIMD tails, av_image_copy_to_buffer); see the corpus README. */
#ifndef FFP_VIRTUAL_CARVE
#error "virtual-frame.c is the plane corpus's virtual carve arm only"
#endif
#include "corpus.h"
#include <capstone/capability.h>

int __real_av_frame_get_buffer(AVFrame *frame, int align);

/* One frame at a time, which is what every case and control holds. */
static struct {
  int live;
  unsigned char *block; /* the malloc'd buffer: freeing it ends every lifetime below */
  void *life;           /* the frame's child lifetime of the block */
} vf;

static void virtual_frame_free(void *opaque, uint8_t *data) {
  (void)opaque;
  (void)data;
  CHECK(vf.live, 951);
  free(vf.block);
  vf.block = NULL;
  vf.life = NULL;
  vf.live = 0;
}

int __wrap_av_frame_get_buffer(AVFrame *frame, int align) {
  CHECK(!vf.live && frame->width > 0 && frame->height > 0 && !frame->linesize[0], 952);
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
  size_t off[4];
  int present[4];
  for (int i = 0; i < 4; i++) {
    linesizes[i] = probe->linesize[i];
    present[i] = probe->data[i] != NULL;
    off[i] = present[i] ? (size_t)(probe->data[i] - probe->buf[0]->data) : 0;
  }
  CHECK(av_image_fill_plane_sizes(sizes, frame->format, FFALIGN(frame->height, 32), linesizes) >= 0,
        955);
  av_frame_free(&probe);

  /* The same layout in one malloc'd buffer: the frame a child of it, each plane a child of the frame. */
  vf.block = malloc(total);
  CHECK(vf.block, 956);
  vf.life = capstone_cap_derive(vf.block, 0, total);
  vf.live = 1;
  for (int i = 0; i < 4; i++) {
    if (!present[i])
      continue;
    CHECK(off[i] + sizes[i] <= total, 957);
    frame->data[i] = capstone_cap_derive(vf.life, off[i], sizes[i]);
    frame->linesize[i] = (int)linesizes[i];
    printf("virtual-frame plane %d region=[%zu,%zu) linesize=%d\n", i, off[i], off[i] + sizes[i],
           frame->linesize[i]);
  }
  frame->buf[0] = av_buffer_create(vf.block, total, virtual_frame_free, NULL, 0);
  CHECK(frame->buf[0], 958);
  frame->extended_data = frame->data;
  return 0;
}
