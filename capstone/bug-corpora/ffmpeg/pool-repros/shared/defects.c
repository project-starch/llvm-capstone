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

static AVBufferPool *g_pool;

/* ---- 1886c3269d: avcodec/h264_refs clears ref_list only up to ref_count ----
 * H264Ref carries data[3] into a picture's planes. The reset memsets entries
 * [len, ref_count) instead of [len, 32), so every record past ref_count keeps
 * its pointers into pictures that have since been returned. */
#define H264_SLOTS 8
struct h264_ref {
  unsigned char *data;
  int valid;
};
static int case_h264_refs(int fixed) {
  struct h264_ref ref_list[H264_SLOTS] = {0};
  AVBufferRef *pic[H264_SLOTS];
  int ref_count = 6, len = 2; /* the list shrank: only 2 entries stay valid */
  for (int i = 0; i < H264_SLOTS; i++) {
    if (!(pic[i] = av_buffer_pool_get(g_pool)))
      ff2_fail(620);
    pic[i]->data[0] = (unsigned char)(0x40 + i);
    ref_list[i].data = pic[i]->data;
    ref_list[i].valid = 1;
  }
  /* the reset, with upstream's bound and with the fix */
  int upto = fixed ? H264_SLOTS : ref_count;
  for (int i = len; i < upto; i++)
    memset(&ref_list[i], 0, sizeof(ref_list[i]));
  /* every picture past the valid length is returned to the pool */
  for (int i = len; i < H264_SLOTS; i++)
    av_buffer_unref(&pic[i]);
  int survivors = 0;
  for (int i = len; i < H264_SLOTS; i++)
    survivors += ref_list[i].valid;
  AVBufferRef *reused = av_buffer_pool_get(g_pool);
  if (!reused)
    ff2_fail(621);
  memset(reused->data, 0xCC, POOL_BYTES);
  /* slot 7 is past ref_count, so upstream's bound never cleared it */
  unsigned char *stale = ref_list[H264_SLOTS - 1].data;
  int same = stale == reused->data;
  printf("arm=%s survivors_past_len=%d reuse_same_address=%d stale_read=0x%02X\n",
         fixed ? "fixed" : "buggy", survivors, same,
         stale ? stale[0] : 0);
  printf("VERDICT %s\n",
         !fixed && survivors && same && stale[0] == 0xCC
             ? "DEFECT-REPRODUCED a record past ref_count still names reissued storage"
         : fixed && !survivors ? "FIXED the reset covers the whole list"
                               : "INCONCLUSIVE");
  av_buffer_unref(&reused);
  for (int i = 0; i < len; i++)
    av_buffer_unref(&pic[i]);
  return !fixed ? !(survivors && same && stale[0] == 0xCC) : !!survivors;
}

/* ---- 316531e61c: avfilter/vidstabtransform parks a raw plane pointer -------
 * The separate-buffer path stores a shallow copy of the source frame in the
 * library's state without allocating. A later in-place frame sees that state as
 * non-null, skips the allocation, and writes through the previous frame's
 * pointer -- into storage the caller no longer owns. */
struct vs_state {
  unsigned char *src; /* shallow copy, no ownership */
};
static int case_vidstab(int fixed) {
  struct vs_state td = {0};
  AVBufferRef *in = av_buffer_pool_get(g_pool);
  if (!in)
    ff2_fail(630);
  memset(in->data, 0x11, POOL_BYTES);
  td.src = in->data; /* vsTransformPrepare's separate-buffer path */
  av_buffer_unref(&in); /* the filter returns the source frame */
  if (fixed)
    td.src = NULL; /* always take the in-place path: nothing is carried over */

  AVBufferRef *next = av_buffer_pool_get(g_pool);
  if (!next)
    ff2_fail(631);
  memset(next->data, 0x22, POOL_BYTES);
  /* the next frame: vsFrameIsNull(&td.src) is false, so allocation is skipped
   * and vsFrameCopy writes through whatever td.src still names */
  int wrote_through_stale = 0;
  if (td.src) {
    td.src[0] = 0x99;
    wrote_through_stale = 1;
  }
  printf("arm=%s wrote_through_stale=%d new_owner_byte=0x%02X\n",
         fixed ? "fixed" : "buggy", wrote_through_stale, next->data[0]);
  printf("VERDICT %s\n",
         !fixed && wrote_through_stale && next->data[0] == 0x99
             ? "DEFECT-REPRODUCED the parked pointer corrupts the new owner"
         : fixed && next->data[0] == 0x22 ? "FIXED nothing is carried across frames"
                                          : "INCONCLUSIVE");
  int bad = !fixed ? !(wrote_through_stale && next->data[0] == 0x99)
                   : next->data[0] != 0x22;
  av_buffer_unref(&next);
  return bad;
}

/* One binary, case and arm chosen at run time, as the other corpora do. */
int main(int argc, char **argv) {
  const char *want = argc > 1 ? argv[1] : "461fb22053";
  int fixed = argc > 2 && !strcmp(argv[2], "fixed");
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
  g_pool = pool;
  if (!strcmp(want, "1886c3269d") || !strcmp(want, "316531e61c")) {
    int rc = !strcmp(want, "1886c3269d") ? case_h264_refs(fixed)
                                         : case_vidstab(fixed);
    av_buffer_pool_uninit(&pool);
    free(metadata);
    free(payload);
    return rc;
  }
  if (strcmp(want, "461fb22053")) {
    fprintf(stderr, "unknown case: %s\n", want);
    return 2;
  }

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
