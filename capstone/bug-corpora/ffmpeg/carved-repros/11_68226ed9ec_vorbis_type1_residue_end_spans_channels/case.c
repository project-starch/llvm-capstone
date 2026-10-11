#include "corpus.h"

/* vorbisdec's residue vectors, fix 68226ed9ec ("Fix decoder bug."). ONE
 * allocation holds every channel's residue (libavcodec/vorbisdec.c of the
 * fix's parent):
 *
 *     vc->channel_residues = av_malloc((vc->blocksize[1] / 2) * vc->audio_channels
 *                                      * sizeof(*vc->channel_residues));      // :952
 *
 * carved per submap as the packet is decoded (:1481, :1560-1562):
 *
 *     float *ch_res_ptr = vc->channel_residues;
 *     ...
 *     vorbis_residue_decode(vc, residue, ch, do_not_decode, ch_res_ptr, blocksize/2);
 *     ch_res_ptr += ch * blocksize / 2;
 *
 * A residue's end was validated against ALL channels' worth of samples for
 * every residue type (:681-683):
 *
 *     res_setup->end > vc->avccontext->channels * vc->blocksize[1] / 2 ||
 *
 * but types 0 and 1 decode one channel at a time, writing vec[voffset + ...]
 * for voffset up to end - 1 (:1359-1364, `vec[voffs]+=codebook.codevectors[...]`).
 * A one-channel submap whose residue ends past blocksize/2 adds its codevectors
 * into the NEXT channel's residue. The fix validates types 0 and 1 against one
 * channel's worth: `(res_setup->type == 2 ? channels : 1) * blocksize[1] / 2`.
 *
 * Two one-channel submaps: channel 0's residue ends at 2 * vlen and channel
 * 1's at vlen, so only the first crossing exists and it stays in the buffer.
 * The residues are cleared once before the submap loop (:1508), so the spill
 * survives into channel 1's decode and its output. */
FFC_CASE(11) {
  const size_t channels = 2, blocksize1 = 256, vlen = blocksize1 / 2;
  const size_t bytes = vlen * channels * sizeof(float);       /* :952 */
  const unsigned type = 1, end0 = 2 * vlen, end1 = vlen;
  const size_t limit = (fixed ? (type == 2 ? channels : 1) : channels) * blocksize1 / 2;

  float *residues = malloc(bytes);
  CHECK(residues, 811);
  float *ch[2];
  ch[0] = ffc_carve(residues, 0, vlen * sizeof(float), "channel_residues[0]");
  ch[1] = ffc_carve(residues, vlen * sizeof(float), vlen * sizeof(float), "channel_residues[1]");
#ifdef FFC_SUBLET_CARVE
  /* The Sublet carve's block is reachable only through its regions, which cover it exactly. */
  memset(ch[0], 0, vlen * sizeof(float));
  memset(ch[1], 0, vlen * sizeof(float));
#else
  memset(residues, 0, bytes);                                  /* :1508 */
#endif
  CHECK(end0 <= channels * vlen, 812);                         /* the nested window */

  if (end0 > limit) {
    /* The fix: the setup header is refused with AVERROR_INVALIDDATA, so no
     * packet is decoded with this residue. */
  } else {
    float *vec = ch[0];
    size_t voffs = 0;
    for (; voffs < end0 && voffs < vlen; voffs++)
      vec[voffs] += 0.25f;
    if (voffs < end0) {
      ffc_note(o, residues, bytes, ch[0], vlen * sizeof(float), &vec[voffs],
               (end0 - voffs) * sizeof(float));
      uint32_t bits = read_probe_u32((const volatile uint32_t *)&vec[voffs]);
      float old;
      memcpy(&old, &bits, sizeof old);
      vec[voffs] = old + 0.25f;
      for (voffs++; voffs < end0; voffs++)
        vec[voffs] += 0.25f;
    } else {
      ffc_note(o, residues, bytes, ch[0], vlen * sizeof(float), &vec[end0 - 1], sizeof(float));
    }
    /* Channel 1's own residue, onto whatever is there. */
    for (size_t k = 0; k < end1; k++)
      ch[1][k] += 0.5f;
    o->damage = ch[1][0] != 0.5f; /* channel 0's codevectors are in channel 1's output */
  }

  o->defect_text = "a type-1 residue ending at 2 * vlen added channel 0's codevectors into "
                   "channel 1's slice of channel_residues";
  o->fixed_text = "the fix refuses a type-0/1 residue that ends past one channel's blocksize/2";
  free(residues);
}
