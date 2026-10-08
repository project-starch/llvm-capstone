#include "corpus.h"

/* cbs_av1's ITU-T T.35 metadata payload, fix 8553e6ef57. The payload is ONE
 * direct allocation: av_buffer_alloc at
 * libavcodec/cbs_av1_syntax_template.c, metadata_itut_t35. Here it is the
 * platform's calloc, because what the row turns on is the SIZE. */

FFH_CASE(5) {
  /* Case 5 -- cbs_av1 metadata_itut_t35, fix 8553e6ef57. PLAIN HEAP: a reader
   * that is allowed to run into FFmpeg's standard input padding runs off a
   * buffer allocated to the exact payload size.
   *
   * At the fix's parent:
   *
   *     current->payload_ref = av_buffer_alloc(current->payload_size);
   *
   * and the fix:
   *
   *     current->payload_ref = av_buffer_alloc(current->payload_size +
   *                                            AV_INPUT_BUFFER_PADDING_SIZE);
   *     memset(current->payload + current->payload_size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
   *
   * AV_INPUT_BUFFER_PADDING_SIZE is 64 in FFmpeg. The contract it encodes is
   * that bitstream readers may read whole words past the logical end of a
   * buffer; allocating the exact size breaks that contract, so the overrun is
   * in the ALLOCATION, not in the reader.
   *
   * THIS ONE LEAVES THE ALLOCATION. */
  const long payload_size = 24;
  const long padding = 64;            /* AV_INPUT_BUFFER_PADDING_SIZE */

  const long cap = fixed ? payload_size + padding : payload_size;
  unsigned char *payload = calloc((size_t)cap, 1);
  CHECK(payload, 751);
  for (long i = 0; i < payload_size; i++)
    payload[i] = (unsigned char)(0x40 + i);

  /* The claim, asserted: unpadded, the reader's last word starts past the end. */
  CHECK(payload_size < payload_size + padding, 752);

  /* A reader taking a 4-byte word at the payload's end, as a bitstream reader
   * may. The first byte past the logical payload is the crossing. */
  const long touched = payload_size + 3;
  unsigned v = 0;
  if (touched >= cap)
    v = read_probe_u8(&payload[touched]);     /* the labelled crossing */
  else
    v = payload[touched];

  o->cap = (unsigned long)cap;
  o->touched = touched;
  o->crossed = touched >= cap;
  o->extent = padding;
  o->damage = o->crossed && v == 0;   /* the value read is outside the payload */

  o->defect_text = "the T.35 payload was allocated at exactly payload_size, so a reader entitled "
                   "to FFmpeg's 64-byte input padding read past av_buffer_alloc";
  o->fixed_text = "the fix allocates payload_size + AV_INPUT_BUFFER_PADDING_SIZE and zeroes the "
                  "padding, so the same read stays inside";
  free(payload);
}
