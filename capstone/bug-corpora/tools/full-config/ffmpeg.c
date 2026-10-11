/* FULL CONFIGURATION, FFmpeg: the Sublet port of FFmpeg's own pools (FF_SUBLET_POOLS=1, the
 * application port's libavutil and ports/ffmpeg/app/src/capstone-domain/ffsublet.c) is linked into
 * a plain case and brought up before main(): a pool is made, one buffer is taken from it -- an
 * entry carved from a block the Sublet heap lends LINEAR -- written, and returned, which is the
 * port's one revoke. The pool stays alive for the whole case, so the case runs with the port live
 * and a lent block outstanding, as it would inside FFmpeg on the Sublet heap.
 *
 * The case's own objects never pass through the pool: they come straight from malloc. This image
 * therefore measures that the port changes nothing for a direct-allocation bug, and a reading
 * differing from the `sublet` arm's would be an interaction, not a catch by the port. */
#include <stdio.h>
#include <stdlib.h>

#include "libavutil/buffer.h"

void ff_sublet_counts(unsigned long out[3]);

static AVBufferPool *full_config_pool;

__attribute__((constructor)) static void full_config_ffmpeg(void) {
  unsigned long counts[3];
  full_config_pool = av_buffer_pool_init(256, NULL);
  AVBufferRef *ref = full_config_pool ? av_buffer_pool_get(full_config_pool) : NULL;
  if (!ref) {
    printf("FULLCONFIG-FAILED ffmpeg: the pool gave no buffer\n");
    fflush(stdout);
    exit(75);
  }
  ref->data[0] = 1;
  av_buffer_unref(&ref); /* back to the pool: the port's give, one revoke */
  ff_sublet_counts(counts);
  printf("FULLCONFIG ffmpeg pools=sublet-port live gives=%lu ends=%lu\n", counts[1], counts[2]);
  fflush(stdout);
  if (counts[1] < 1) {
    printf("FULLCONFIG-FAILED ffmpeg: the unref did not reach the port\n");
    fflush(stdout);
    exit(75);
  }
}
