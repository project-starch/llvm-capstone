/* Case 3: 5c66a3ab51 -- a non-reference VVC frame that is output gets fully
 * released, returning its pooled side tables while the decoder still holds them
 *
 * Shape: premature return to the pool / reuse / stale read
 * Consumer: libavcodec/vvc/refs.c, ff_vvc_set_new_ref() and ff_vvc_output_frame()
 * Allocator: AVRefStructPool for the side tables, AVBufferPool for the planes
 * Claims and provenance: case.json and PROVENANCE.md beside this file.
 *
 * This is the corpus's first AVRefStructPool case. The pool is a recycling free
 * list -- av_refstruct_unref pushes the entry onto pool->available_entries
 * (libavutil/refstruct.c:230-231) and the next av_refstruct_pool_get pops the
 * same one back (:258-261) -- so a premature release never reaches the system
 * allocator and the reuse is invisible to a free-keyed tool.
 */
#include "../shared/corpus.h"

/* libavcodec/vvc/refs.h:28-32 at the pin. */
#define VVC_FRAME_FLAG_OUTPUT (1 << 0)
#define VVC_FRAME_FLAG_SHORT_REF (1 << 1)
#define VVC_FRAME_FLAG_CORRUPT (1 << 4)

/* The part of VVCFrame this defect touches. */
struct vvc_frame {
  AVBufferRef *plane;          /* the payload, from the AVBufferPool */
  unsigned char *tab_dmvr_mvf; /* a side table, from the AVRefStructPool */
  unsigned char *rpl_tab;      /* ditto */
  int flags;
};

/* ff_vvc_unref_frame, transcribed from refs.c:44-73 at the pin: clearing the
 * last flag releases everything the frame owns, the pooled side tables included. */
static void unref_frame(struct vvc_frame *f, int flags) {
  if (!f->plane)
    return;
  f->flags &= ~flags;
  if (!(f->flags & ~VVC_FRAME_FLAG_CORRUPT))
    f->flags = 0;
  if (!f->flags) {
    av_buffer_unref(&f->plane);
    av_refstruct_unref(&f->tab_dmvr_mvf);
    av_refstruct_unref(&f->rpl_tab);
  }
}

FF2_CASE(3) {
  struct vvc_frame frame = {0};

  /* ff_vvc_set_new_ref, refs.c:150-158: the planes come from the buffer pool and
   * the side tables from their own refstruct pools. */
  frame.plane = av_buffer_pool_get(g_pool);
  CHECK(frame.plane, 603);
  frame.tab_dmvr_mvf = av_refstruct_pool_get(g_refpool);
  CHECK(frame.tab_dmvr_mvf, 604);
  frame.rpl_tab = av_refstruct_pool_get(g_refpool);
  CHECK(frame.rpl_tab, 605);
  CHECK(frame.tab_dmvr_mvf != frame.rpl_tab, 606);
  memset(frame.tab_dmvr_mvf, 0xA0, TAB_BYTES);

  /* The flags a picture gets when ph_pic_output_flag is set and
   * ph_non_ref_pic_flag is set. Upstream 5c66a3ab51 adds the SHORT_REF term at
   * refs.c:254; without it OUTPUT is the only flag, and the `if
   * (!ph_non_ref_pic_flag)` at :256-257 does NOT put SHORT_REF back, because
   * this picture is not a reference picture. That is the whole defect. */
  frame.flags = fixed ? (VVC_FRAME_FLAG_OUTPUT | VVC_FRAME_FLAG_SHORT_REF)
                      : VVC_FRAME_FLAG_OUTPUT;

  /* Live control: the table is readable while the frame is held. */
  CHECK(frame.tab_dmvr_mvf[0] == 0xA0, 607);
  unsigned char *held = frame.tab_dmvr_mvf; /* what the decoder goes on using */

  /* ff_vvc_output_frame can select the frame that has not finished decoding and
   * clears its OUTPUT flag. */
  unref_frame(&frame, VVC_FRAME_FLAG_OUTPUT);
  int released = frame.tab_dmvr_mvf == NULL;

  /* The next picture's side table comes from the same pool. */
  unsigned char *reissued = av_refstruct_pool_get(g_refpool);
  CHECK(reissued, 608);
  int same = reissued == held;
  memset(reissued, 0xCC, TAB_BYTES);

  printf("flags_after_output=%d tables_released=%d reuse_same_address=%d "
         "stale_read=0x%02X new_owner=0x%02X\n",
         frame.flags, released, same, held[0], reissued[0]);
  FF2_VERDICT(!fixed && released && same && held[0] == 0xCC, fixed && !released,
              "the decoder's side table went back to the pool and was reissued",
              "SHORT_REF keeps the frame alive, so the tables stay with it");
  int bad = !fixed ? !(released && same && held[0] == 0xCC) : !!released;
  av_refstruct_unref(&reissued);
  av_refstruct_unref(&frame.tab_dmvr_mvf);
  av_refstruct_unref(&frame.rpl_tab);
  av_buffer_unref(&frame.plane);
  return bad;
}
