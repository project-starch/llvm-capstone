/* The decode core. See ffapp_decode.h for the contract and the output format. */
#include <stdio.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libavutil/md5.h>
#include <libavutil/mem.h>

#include "ffapp_decode.h"

/* One framemd5-style line. Stream 0, timestamps are the frame index at tb 1/30, as in the
 * reference produced by record.sh; only the hash column is the oracle. */
static int emit_frame(const AVFrame *f, int n)
{
    uint8_t md5[16];
    int size = av_image_get_buffer_size(f->format, f->width, f->height, 1);
    uint8_t *buf = size > 0 ? av_malloc(size) : NULL;
    if (!buf)
        return FFAPP_E_HASH;
    av_image_copy_to_buffer(buf, size, (const uint8_t *const *)f->data, f->linesize,
                            f->format, f->width, f->height, 1);
    av_md5_sum(md5, buf, size);
    av_free(buf);
    printf("0, %8d, %8d, %8d, %8d, ", n, n, 1, size);
    for (int i = 0; i < 16; i++)
        printf("%02x", md5[i]);
    printf("\n");
    return 0;
}

int ffapp_run(const char *path, int stop_at)
{
    AVFormatContext *fmt = NULL;
    AVCodecContext *dec = NULL;
    AVPacket *pkt = NULL;
    AVFrame *frm = NULL;
    int ret, vs, frames = 0, packets = 0, status;

    printf("STAGE M1 main\n");
    if (stop_at <= FFAPP_M1_MAIN)
        return FFAPP_M1_MAIN;

    if (avformat_open_input(&fmt, path, NULL, NULL) < 0)
        return FFAPP_E_OPEN;
    if (avformat_find_stream_info(fmt, NULL) < 0) {
        status = FFAPP_E_INFO;
        goto out;
    }
    vs = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if (vs < 0) {
        status = FFAPP_E_NOVIDEO;
        goto out;
    }
    printf("STAGE M2 open streams=%u video=%d %dx%d\n", fmt->nb_streams, vs,
           fmt->streams[vs]->codecpar->width, fmt->streams[vs]->codecpar->height);
    if (stop_at <= FFAPP_M2_OPEN) {
        status = FFAPP_M2_OPEN;
        goto out;
    }

    const AVCodec *codec = avcodec_find_decoder(fmt->streams[vs]->codecpar->codec_id);
    if (!codec) {
        status = FFAPP_E_NODECODER;
        goto out;
    }
    dec = avcodec_alloc_context3(codec);
    pkt = av_packet_alloc();
    frm = av_frame_alloc();
    if (!dec || !pkt || !frm) {
        status = FFAPP_E_ALLOC;
        goto out;
    }
    avcodec_parameters_to_context(dec, fmt->streams[vs]->codecpar);
    dec->thread_count = 1;
    if (avcodec_open2(dec, codec, NULL) < 0) {
        status = FFAPP_E_CODEC;
        goto out;
    }

    while ((ret = av_read_frame(fmt, pkt)) >= 0) {
        if (packets++ == 0) {
            printf("STAGE M3 first packet size=%d\n", pkt->size);
            if (stop_at <= FFAPP_M3_PACKET) {
                av_packet_unref(pkt);
                status = FFAPP_M3_PACKET;
                goto out;
            }
        }
        if (pkt->stream_index == vs && avcodec_send_packet(dec, pkt) >= 0) {
            while (avcodec_receive_frame(dec, frm) >= 0) {
                if ((status = emit_frame(frm, frames++)) != 0) {
                    av_packet_unref(pkt);
                    goto out;
                }
                if (frames == 1) {
                    printf("STAGE M4 first frame\n");
                    if (stop_at <= FFAPP_M4_FRAME) {
                        av_packet_unref(pkt);
                        status = FFAPP_M4_FRAME;
                        goto out;
                    }
                }
            }
        }
        av_packet_unref(pkt);
    }
    if (packets == 0) {
        status = FFAPP_E_NOPACKET;
        goto out;
    }
    avcodec_send_packet(dec, NULL);           /* drain */
    while (avcodec_receive_frame(dec, frm) >= 0)
        if ((status = emit_frame(frm, frames++)) != 0)
            goto out;
    if (frames == 0) {
        status = FFAPP_E_NOFRAME;
        goto out;
    }
    printf("STAGE M5 frames=%d packets=%d\n", frames, packets);
    status = FFAPP_M5_ALL;

out:
    av_frame_free(&frm);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    return status;
}
