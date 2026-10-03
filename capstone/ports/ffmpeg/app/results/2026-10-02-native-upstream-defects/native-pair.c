/* Native reproduction of the two upstream FFmpeg defects fixtures 24 and 25 transcribe.
 *
 * Purpose: the matched fix-differential pair, and the host-ASan arm. These defects are plain
 * malloc/free, so ASan MUST report them -- that is what makes them the control half rather than
 * a claim. An ASan arm that stayed silent here would mean the reproduction is wrong, not that
 * the defect is subtle.
 *
 *   case 24  libavcodec/vvc/thread.c ff_vvc_frame_thread_free: av_freep(&ft) nulls the local
 *            alias, leaving fc->ft naming freed storage.   fix: av_freep(&fc->ft)
 *   case 25  libswscale/ops_dispatch.c compile_single: comp = &p->comp is an interior pointer
 *            into p; p is freed; comp->backend is then read.   fix: use the copy c taken first
 *
 * usage: ff-native <24|25> <buggy|fixed>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* av_freep's semantics, verbatim: free what the pointer names, then null THAT pointer. */
static void av_freep_(void *arg) {
  void **p = (void **)arg;
  free(*p);
  *p = NULL;
}

struct frame_context {
  unsigned char *ft; /* VVCFrameContext::ft */
};

static int case24(int fixed) {
  struct frame_context fc_obj, *fc = &fc_obj;
  fc->ft = malloc(64);
  if (!fc->ft)
    return 75;
  memset(fc->ft, 0xA0, 64);

  if (fixed) {
    av_freep_(&fc->ft); /* the fix: null the FIELD */
  } else {
    unsigned char *ft = fc->ft; /* VVCFrameThread *ft = fc->ft; */
    av_freep_(&ft);             /* av_freep(&ft): nulls the local only */
  }
  unsigned char *q = malloc(64); /* the storage goes to a new owner */
  if (!q)
    return 75;
  memset(q, 0x5B, 64);

  int reused = (q == fc_obj.ft);
  if (fixed) {
    printf("VERDICT FIXED field-nulled=%d\n", fc_obj.ft == NULL);
    free(q);
    return 0;
  }
  unsigned char got = fc_obj.ft[0]; /* a later frame_thread_* reads fc->ft */
  printf("VERDICT DEFECT-REPRODUCED same-address=%d read=%02x (0x5b = the new owner's byte)\n",
         reused, got);
  free(q);
  return 0;
}

struct compiled_op {
  unsigned char tag[32];
  unsigned char backend[32]; /* the field read after the free */
};

static int case25(int fixed) {
  struct compiled_op *p = malloc(sizeof *p);
  if (!p)
    return 75;
  memset(p, 0xA0, sizeof *p);

  const unsigned char *comp = p->backend; /* comp = &p->comp: INTERIOR into p */
  struct compiled_op c = *p;              /* SwsCompiledOp c = *comp; taken before the free */
  free(p);                                /* av_free(p) */

  struct compiled_op *q = malloc(sizeof *q); /* the storage goes to a new owner */
  if (!q)
    return 75;
  memset(q, 0x5B, sizeof *q);

  int reused = ((const unsigned char *)q->backend == comp);
  if (fixed) {
    printf("VERDICT FIXED read-from-copy=%02x\n", c.backend[0]);
    free(q);
    return 0;
  }
  unsigned char got = comp[0]; /* (*output)->backend = comp->backend->flags; */
  printf("VERDICT DEFECT-REPRODUCED same-address=%d read=%02x (0x5b = the new owner's byte)\n",
         reused, got);
  free(q);
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <24|25> <buggy|fixed>\n", argv[0]);
    return 75;
  }
  int fixed = !strcmp(argv[2], "fixed");
  if (!fixed && strcmp(argv[2], "buggy"))
    return 75;
  if (!strcmp(argv[1], "24"))
    return case24(fixed);
  if (!strcmp(argv[1], "25"))
    return case25(fixed);
  return 75;
}
