/* af_join dedup-bound defect, reduced to its allocator call sequence.
 *
 * Upstream fix 461fb22053 ("avfilter/af_join: fix wrong loop bound in buffer
 * dedup (use-after-free)"): try_push_frame() decides whether an input buffer is
 * already tracked by testing `j == i`, the channel index, instead of
 * `j == nb_buffers`. Once an earlier channel shared a buffer, nb_buffers falls
 * behind i, and a genuinely new buffer is never referenced -- so it is released
 * while the output frame's extended_data still points into it.
 *
 * Real here: libavutil/buffer.c, the AVBufferPool itself.
 * Reduced: the JoinContext dedup loop and the part of AVFrame it touches.
 * Reaching the original needs a filter graph and two audio inputs, which does
 * not change what the allocator is asked to do.
 */
#include "libavutil/buffer.h"
#include "trace.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCH 3
#define POOL_BYTES 64
#define PLANE_BYTES 32

_Noreturn void ff2_fail(unsigned code) {
  fprintf(stderr, "CONTROL-FAILED %u\n", code);
  exit(75); /* the corpus convention: a broken control is never a verdict */
}
void ff2_sink(const struct ff2_event *event) { (void)event; }
void ff2_lock(void) {}
void ff2_unlock(int *guard) { (void)guard; }

/* The part of AVFrame this defect touches. */
struct frame {
  unsigned char *extended_data[NCH];
  AVBufferRef *buf[NCH];
  int nb_buf;
};
/* af_join's ChannelMap. */
struct chanmap {
  int input, in_channel_idx;
};

/* av_frame_get_plane_buffer: the buffer whose storage covers this plane. */
static AVBufferRef *plane_buffer(struct frame *f, int idx) {
  for (int i = 0; i < f->nb_buf; i++) {
    unsigned char *d = f->buf[i]->data;
    if (f->extended_data[idx] >= d && f->extended_data[idx] < d + f->buf[i]->size)
      return f->buf[i];
  }
  return NULL;
}

/* try_push_frame's copy-and-dedup loop. `fixed` selects the upstream fix. */
static int join(struct frame *out, struct frame *in, const struct chanmap *map,
                int nb_channels, int fixed) {
  AVBufferRef *buffers[NCH];
  int nb_buffers = 0, i, j;
  for (i = 0; i < nb_channels; i++) {
    const struct chanmap *ch = &map[i];
    struct frame *cur = &in[ch->input];
    out->extended_data[i] = cur->extended_data[ch->in_channel_idx];
    AVBufferRef *buf = plane_buffer(cur, ch->in_channel_idx);
    if (!buf)
      ff2_fail(601);
    for (j = 0; j < nb_buffers; j++)
      if (buffers[j]->buffer == buf->buffer)
        break;
    if (j == (fixed ? nb_buffers : i))
      buffers[nb_buffers++] = buf;
  }
  /* create references to the buffers we copied to output */
  for (i = 0; i < nb_buffers; i++)
    if (!(out->buf[out->nb_buf++] = av_buffer_ref(buffers[i])))
      ff2_fail(602);
  return nb_buffers;
}

static struct frame make_input(AVBufferPool *pool, int planes, unsigned char fill) {
  struct frame f = {0};
  AVBufferRef *b = av_buffer_pool_get(pool);
  if (!b)
    ff2_fail(603);
  memset(b->data, fill, POOL_BYTES);
  f.buf[f.nb_buf++] = b;
  for (int i = 0; i < planes; i++)
    f.extended_data[i] = b->data + i * PLANE_BYTES;
  return f;
}

/* One binary, case and arm chosen at run time, as the other corpora do. */
int main(int argc, char **argv) {
  const char *want = argc > 1 ? argv[1] : "461fb22053";
  int fixed = argc > 2 && !strcmp(argv[2], "fixed");
  if (strcmp(want, "461fb22053")) {
    fprintf(stderr, "unknown case: %s\n", want);
    return 2;
  }
  void *metadata = aligned_alloc(64, FF2_META_BYTES);
  void *payload = aligned_alloc(64, FF2_PAYLOAD_BYTES);
  if (!metadata || !payload)
    ff2_fail(604);
  ff2_memory_init(metadata, FF2_META_BYTES);
  ff2_payload_init(payload, FF2_PAYLOAD_BYTES);
  ff2_set_mode(0); /* spatial only: this arm must show the defect, not block it */
  ff2_reset();

  AVBufferPool *pool = av_buffer_pool_init(POOL_BYTES, NULL);
  if (!pool)
    ff2_fail(605);

  /* Input 0 carries two channels in one buffer, input 1 carries one channel.
   * That is the minimum that makes nb_buffers fall behind i. */
  struct frame in[2];
  in[0] = make_input(pool, 2, 0xA0);
  in[1] = make_input(pool, 1, 0xB0);
  const struct chanmap map[NCH] = {{0, 0}, {0, 1}, {1, 0}};

  struct frame out = {0};
  int tracked = join(&out, in, map, NCH, fixed);

  unsigned char *victim = out.extended_data[2]; /* points into input 1's buffer */
  unsigned char *shared = in[1].buf[0]->data;
  if (victim != shared)
    ff2_fail(606);
  printf("arm=%s tracked_buffers=%d output_refs=%d\n",
         fixed ? "fixed" : "buggy", tracked, out.nb_buf);

  /* Live control: every channel must be readable before anything is released. */
  if (out.extended_data[0][0] != 0xA0 || out.extended_data[1][0] != 0xA0 ||
      out.extended_data[2][0] != 0xB0)
    ff2_fail(607);

  /* ff_filter_frame has taken the output; the inputs are freed. */
  for (int i = 0; i < 2; i++)
    av_buffer_unref(&in[i].buf[0]);

  /* Did input 1's storage go back to the pool while the output still names it? */
  AVBufferRef *reused = av_buffer_pool_get(pool);
  if (!reused)
    ff2_fail(608);
  int same = reused->data == shared;
  memset(reused->data, 0xCC, POOL_BYTES);

  printf("reuse_same_address=%d stale_read=0x%02X new_owner=0x%02X\n", same,
         victim[0], reused->data[0]);
  printf("VERDICT %s\n",
         !fixed && same && victim[0] == 0xCC
             ? "DEFECT-REPRODUCED stale pointer reads the new owner's payload"
         : fixed && !same ? "FIXED output holds a reference, storage not reissued"
                          : "INCONCLUSIVE");

  av_buffer_unref(&reused);
  for (int i = 0; i < out.nb_buf; i++)
    av_buffer_unref(&out.buf[i]);
  av_buffer_pool_uninit(&pool);
  free(metadata);
  free(payload);
  return 0;
}
