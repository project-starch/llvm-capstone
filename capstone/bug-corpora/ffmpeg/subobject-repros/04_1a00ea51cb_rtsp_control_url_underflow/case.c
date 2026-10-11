#include "corpus.h"
#include <stdint.h>

/* RTSPStream's first members exactly as rtsp.h declares them at the pin, up to
 * and including control_url. The struct is ONE allocation --
 * av_mallocz(sizeof(RTSPStream)) at rtsp.c:286 and :525 -- so the bound between
 * control_url and the member before it is not a bound any allocator knows about.
 *
 * MAX_URL_SIZE is 4096 (internal.h:30). The members before control_url are two
 * pointers and three ints, so control_url does not start at offset 0; that is
 * the fact the whole case turns on and it is asserted below rather than assumed. */
#define MAX_URL_SIZE 4096

struct rtsp_stream {
  void *rtp_handle;      /* URLContext *  */
  void *transport_priv;  /* void *        */
  int stream_index;
  int interleaved_min, interleaved_max;
  char control_url[MAX_URL_SIZE];
  /* members after control_url exist upstream and cannot be reached by this
   * defect, so they are left out. */
};

/* The crossing, labelled so a fault or a sanitiser report can be required to
 * land HERE rather than merely somewhere in the program. */
__attribute__((noinline, used)) static int
read_probe(const volatile char *p) {
  return *p;
}

FF2_CASE(4) {
  /* Case 4 -- RTSP SDP parser, fix 1a00ea51cb. SUB-OBJECT: a one-byte read that
   * underflows out of the start of one struct member into the one before it,
   * inside a single allocation.
   *
   * rtsp.c:614 at the pin reads, for a relative control URL:
   *
   *     if (rtsp_st->control_url[strlen(rtsp_st->control_url)-1]!='/')
   *
   * An a=control: line that is empty, or a stream whose control_url has not
   * been set, leaves control_url an empty string. strlen is then 0 and
   * `strlen(...) - 1` is (size_t)-1, because strlen returns size_t. The index
   * is therefore SIZE_MAX, and `control_url + SIZE_MAX` is, in the modular
   * arithmetic a byte pointer obeys, control_url - 1.
   *
   * THE PREDICTION, AND ITS REASON, because this one is easy to get wrong: the
   * access COMPLETES on a capability machine. control_url does not start at
   * offset 0 -- it sits after two pointers and three ints -- so control_url - 1
   * is the last byte of interleaved_max and is comfortably INSIDE the
   * allocation. The address is representable and in bounds. Nothing faults, and
   * the index wrapping to SIZE_MAX is not what makes it reachable; the member's
   * non-zero offset is. The fix adds `len == 0 ||` and never forms the address.
   *
   * NOTHING WE HAVE CAN CATCH THIS, which is the row's purpose. */
  struct rtsp_stream *st = av_refstruct_allocz(sizeof *st);
  CHECK(st, 621);
  /* The claim, asserted: control_url is NOT first, so index -1 stays inside. */
  CHECK((char *)st->control_url > (char *)st, 622);
  CHECK((char *)&st->control_url[0] - 1 == (char *)&st->interleaved_max
            + sizeof st->interleaved_max - 1,
        623);

  /* A sentinel in the member the underflow lands on, so the read is attributed
   * to that member and not merely to "some byte". */
  st->interleaved_max = 0x5A5A5A5A;
  st->control_url[0] = '\0'; /* the empty relative control URL */

  size_t len = strlen(st->control_url);
  int saw;
  if (fixed) {
    /* The fix: len == 0 short-circuits and the address is never formed. */
    saw = (len == 0) ? -1 : read_probe(&st->control_url[len - 1]);
  } else {
    saw = read_probe(&st->control_url[len - 1]); /* index (size_t)-1 */
  }

  int underflowed = saw == 0x5A;        /* the low byte of interleaved_max */
  int never_formed = saw == -1;
  unsigned long offset = (unsigned long)((char *)st->control_url - (char *)st);

  printf("cap=%zu member_offset=%lu strlen=%zu saw=0x%02x\n", sizeof *st, offset,
         len, (unsigned)(saw & 0xFF));

  FF2_VERDICT(!fixed && underflowed && offset > 0,
              fixed && never_formed,
              "strlen(control_url)-1 wrapped to SIZE_MAX and the read landed one "
              "byte before control_url, on interleaved_max, inside one allocation",
              "the fix's len == 0 check short-circuits and the address is never formed");
  av_refstruct_unref(&st);
  return !fixed ? !underflowed : !never_formed;
}
