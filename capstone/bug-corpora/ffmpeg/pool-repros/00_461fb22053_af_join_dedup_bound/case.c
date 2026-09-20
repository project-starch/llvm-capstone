/* Case 0: 461fb22053 -- af_join tracks an input buffer by the wrong bound
 *
 * Shape: reference never taken / reuse / stale read
 * Consumer: libavfilter/af_join.c, try_push_frame()
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 */
#include "../shared/corpus.h"

#define NCH 3

/* The part of AVFrame this defect touches. */
struct frame {
  unsigned char *extended_data[NCH];
  AVBufferRef *buf[NCH];
  int nb_buf;
};
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

static struct frame make_input(int planes, unsigned char fill) {
  struct frame f = {0};
  AVBufferRef *b = av_buffer_pool_get(g_pool);
  CHECK(b, 603);
  memset(b->data, fill, POOL_BYTES);
  f.buf[f.nb_buf++] = b;
  for (int i = 0; i < planes; i++)
    f.extended_data[i] = b->data + i * PLANE_BYTES;
  return f;
}

FF2_CASE(0) {
  /* Input 0 carries two channels in one buffer, input 1 carries one channel:
   * the minimum that makes nb_buffers fall behind the channel index. */
  struct frame in[2];
  in[0] = make_input(2, 0xA0);
  in[1] = make_input(1, 0xB0);
  const struct chanmap map[NCH] = {{0, 0}, {0, 1}, {1, 0}};
  struct frame out = {0};

  AVBufferRef *buffers[NCH];
  int nb_buffers = 0, i, j;
  for (i = 0; i < NCH; i++) {
    const struct chanmap *ch = &map[i];
    struct frame *cur = &in[ch->input];
    out.extended_data[i] = cur->extended_data[ch->in_channel_idx];
    AVBufferRef *buf = plane_buffer(cur, ch->in_channel_idx);
    CHECK(buf, 601);
    for (j = 0; j < nb_buffers; j++)
      if (buffers[j]->buffer == buf->buffer)
        break;
    if (j == (fixed ? nb_buffers : i)) /* upstream 461fb22053 */
      buffers[nb_buffers++] = buf;
  }
  for (i = 0; i < nb_buffers; i++)
    CHECK((out.buf[out.nb_buf++] = av_buffer_ref(buffers[i])), 602);

  unsigned char *victim = out.extended_data[2];
  unsigned char *shared = in[1].buf[0]->data;
  CHECK(victim == shared, 606);
  /* Live control: every channel is readable before anything is released. */
  CHECK(out.extended_data[0][0] == 0xA0 && out.extended_data[1][0] == 0xA0 &&
            out.extended_data[2][0] == 0xB0, 607);

  for (i = 0; i < 2; i++)
    av_buffer_unref(&in[i].buf[0]); /* ff_filter_frame took the output */

  AVBufferRef *reused = av_buffer_pool_get(g_pool);
  CHECK(reused, 608);
  int same = reused->data == shared;
  memset(reused->data, 0xCC, POOL_BYTES);

  printf("tracked_buffers=%d output_refs=%d reuse_same_address=%d "
         "stale_read=0x%02X new_owner=0x%02X\n",
         nb_buffers, out.nb_buf, same, victim[0], reused->data[0]);
  FF2_VERDICT(!fixed && same && victim[0] == 0xCC, fixed && !same,
              "stale pointer reads the new owner's payload",
              "output holds a reference, storage not reissued");
  int bad = !fixed ? !(same && victim[0] == 0xCC) : !!same;
  av_buffer_unref(&reused);
  for (i = 0; i < out.nb_buf; i++)
    av_buffer_unref(&out.buf[i]);
  return bad;
}
