/* bstest: x264's own bit writer (common/bs.h, osdep.h's WORD_SIZE and endian_fix, unchanged) writing a known sequence:
 * the SPS header bits 0x67 0x64 0x00 0x1f, then ue/u values, flushed. Built as the Capstone arm (sizeof(void*) = 16),
 * the Plain arm (8), and, with -DFORCE_WORD8, the Capstone arm with WORD_SIZE redefined to 8 after osdep.h. */
#include <stdio.h>
#include <string.h>
#include "common/osdep.h"
#ifdef FORCE_WORD8
#undef WORD_SIZE
#define WORD_SIZE 8
#endif
#define MAY_ALIAS __attribute__((may_alias))
typedef union { uint32_t i; uint16_t b[2]; uint8_t c[4]; } MAY_ALIAS x264_union32_t;
#define M32(src) (((x264_union32_t*)(src))->i)
#include "common/bs.h"
int main(void) {
    static uint8_t buf[64] __attribute__((aligned(16)));
    bs_t s; memset(buf, 0, sizeof buf);
    bs_init(&s, buf, sizeof buf);
    bs_write(&s, 8, 0x67); bs_write(&s, 8, 0x64); bs_write(&s, 8, 0x00); bs_write(&s, 8, 0x1f);
    bs_write(&s, 13, 0x1234); bs_write(&s, 3, 5); bs_write(&s, 16, 0xbeef); bs_write(&s, 32, 0xdeadbeef);
    bs_flush(&s);
    printf("BS WORD_SIZE=%d bytes=%d :", (int)WORD_SIZE, (int)(s.p - s.p_start));
    for (int i = 0; i < 16; i++) printf(" %02x", buf[i]);
    printf("\n"); return 0;
}
