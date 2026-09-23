/* FFmpeg as an application: demux -> decode -> per-frame MD5, shared by the native build
 * and the Capstone domain so both run byte-for-byte the same decode path.
 *
 * The output lines follow the framemd5 muxer's frame-line layout, and the MD5 is computed
 * over exactly what framemd5 hashes for decoded video: the frame packed with
 * av_image_copy_to_buffer(align = 1), i.e. the rawvideo packet. So the hash column can be
 * compared with a native `ffmpeg -f framemd5` reference directly.
 *
 * STAGED, so every run returns (CLAUDE.md, "make every run return"): ffapp_run stops after
 * the requested milestone and says how far it got. A failure is always reported with the
 * stage it happened in, never as a hang that only says "somewhere after the last marker". */
#ifndef FFAPP_DECODE_H
#define FFAPP_DECODE_H

/* Milestones. A successful run returns the milestone it stopped at. */
#define FFAPP_M1_MAIN        1  /* entered the program                          */
#define FFAPP_M2_OPEN        2  /* avformat_open_input + find_stream_info        */
#define FFAPP_M3_PACKET      3  /* first packet read                             */
#define FFAPP_M4_FRAME       4  /* first frame decoded and hashed                */
#define FFAPP_M5_ALL         5  /* every frame decoded, decoder drained          */

/* Failures: 10 * stage + detail, so the stage is readable from the number alone. */
#define FFAPP_E_OPEN        21  /* avformat_open_input failed                    */
#define FFAPP_E_INFO        22  /* avformat_find_stream_info failed              */
#define FFAPP_E_NOVIDEO     23  /* no video stream                               */
#define FFAPP_E_ALLOC       31  /* packet/frame/context allocation failed        */
#define FFAPP_E_NODECODER   32  /* no decoder for the stream's codec             */
#define FFAPP_E_CODEC       33  /* avcodec_open2 failed                          */
#define FFAPP_E_NOPACKET    34  /* the file yielded no packet at all             */
#define FFAPP_E_HASH        41  /* could not allocate the frame-packing buffer   */
#define FFAPP_E_NOFRAME     42  /* stream ended before any frame decoded         */

int ffapp_run(const char *path, int stop_at);

#endif
