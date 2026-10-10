#include "corpus.h"

/* alsdec's multi-channel-correlation data, fix cd09284924 ("Fix wrong buffer
 * allocation for MCC in ALS"). ONE allocation carved ONE SLOT per channel
 * (libavcodec/alsdec.c:1565-1579 of the fix's parent):
 *
 *     ctx->chan_data_buffer = av_malloc(sizeof(*ctx->chan_data_buffer) * num_buffers);
 *     ...
 *     ctx->chan_data[c] = ctx->chan_data_buffer + c;
 *
 * but read_channel_data() treats chan_data[c] as a LIST of up to `channels`
 * entries, ending in an entry whose stop_flag is set (:1119-1126):
 *
 *     ALSChannelData *current = cd;
 *     while (entries < channels && !(current->stop_flag = get_bits1(gb))) {
 *         ...
 *         current++;
 *         entries++;
 *
 * so a channel with one dependency writes its terminator into the NEXT
 * channel's slot. For channel 0 the list is at most `channels` slots long, so
 * it never leaves the allocation. When channel 1's own list is read into slot 1
 * afterwards, channel 0's terminator is gone and channel 0's list runs on into
 * channel 1's entries. The fix gives every channel num_buffers slots,
 * `chan_data_buffer + c * num_buffers`, in an allocation num_buffers^2 long.
 *
 * The struct is upstream's at the parent (:177-184). */
typedef struct {
  int stop_flag;
  int master_channel;
  int time_diff_flag;
  int time_diff_sign;
  int time_diff_index;
  int weighting[6];
} ALSChannelData;

/* One read_channel_data(): `deps` dependency entries, then the terminator. The
 * first channel's terminator is written through the probe when it is the
 * crossing access. */
static void read_channel_data(ALSChannelData *cd, unsigned deps, unsigned channels,
                              int c, int probe_terminator) {
  ALSChannelData *current = cd;
  unsigned entries = 0;
  while (entries < channels) {
    int stop = entries == deps;
    if (stop && probe_terminator)
      write_probe_u32((volatile uint32_t *)&current->stop_flag, 1);
    else
      current->stop_flag = stop;
    if (stop)
      break;
    current->master_channel = (c + 1) % (int)channels;
    current->time_diff_flag = 0;
    current->weighting[0] = current->weighting[1] = current->weighting[2] = 7;
    current++;
    entries++;
  }
}

FFC_CASE(4) {
  const unsigned channels = 3, num_buffers = channels; /* mc_coding: :1628 */
  const size_t slots = fixed ? num_buffers : 1;        /* the fix's c * num_buffers */
  const size_t bytes = sizeof(ALSChannelData) * num_buffers * slots;

  ALSChannelData *buffer = calloc(num_buffers * slots, sizeof(ALSChannelData));
  CHECK(buffer, 741);
  ALSChannelData *chan_data[3];
  static const char *names[3] = {"chan_data[0]", "chan_data[1]", "chan_data[2]"};
  for (unsigned c = 0; c < num_buffers; c++)
    chan_data[c] = ffc_carve(buffer, c * slots * sizeof(ALSChannelData),
                             slots * sizeof(ALSChannelData), names[c]);

  /* Channel 0 depends on channel 1: one entry, then its terminator at cd[1]. */
  ffc_note(o, buffer, bytes, chan_data[0], slots * sizeof(ALSChannelData),
           &chan_data[0][1].stop_flag, sizeof(int));
  read_channel_data(chan_data[0], 1, channels, 0, 1);
  /* Channel 1 depends on channel 2; channel 2 has no dependency. On the buggy
   * layout channel 1's entry overwrites channel 0's terminator. */
  read_channel_data(chan_data[1], 1, channels, 1, 0);
  read_channel_data(chan_data[2], 0, channels, 2, 0);

  /* Walk channel 0's list as revert_channel_correlation would. */
  unsigned n = 0;
  for (ALSChannelData *cur = chan_data[0]; n < channels && !cur->stop_flag; cur++)
    n++;
  o->damage = n != 1; /* channel 0 now carries channel 1's dependency too */

  o->defect_text = "channel 0's dependency list ran one slot past its one-slot carve, and "
                   "channel 1's entry then replaced its terminator";
  o->fixed_text = "the fix gives each channel num_buffers slots, so the terminator stays in "
                  "channel 0's own carve";
  free(buffer);
}
