#ifndef CAPSTONE_SLT_MD5_H
#define CAPSTONE_SLT_MD5_H

/* MD5 (RFC 1321), because 91% of the SQLLogicTest corpus needs it.
 *
 * WHY THIS IS VENDORED AND NOT SKIPPED. The obvious way to avoid a hash implementation is
 * to compare only the records that spell their expected values out and report the rest as
 * SKIPPED. Measured on the corpus subset before writing any of this: of 12,516 query
 * records in select1-5 plus the aggfunc evidence file, 7,393 are of the form
 * "<n> values hashing to <md5>" -- between 80% and 92% per file. A runner without MD5
 * would therefore report a pass rate over the 9% remainder and license nothing, which is
 * the "no data rendered as a result" failure this project keeps paying for.
 *
 * WRITTEN FOR THE DOMAIN, so: no libc beyond memcpy, no unaligned word loads (bytes are
 * decoded explicitly, which also makes it endian-independent), no allocation, no statics.
 * The state is a plain struct the caller owns.
 *
 * slt_u32 must be exactly 32 bits. Everything is masked back to 32 bits after each add
 * and rotate rather than relying on the type's width, so a target where `unsigned` is
 * wider still computes the right digest.
 */

typedef unsigned int slt_u32;
#define SLT_M32(x) ((slt_u32)((x) & 0xffffffffu))

struct slt_md5 {
  slt_u32 h[4];
  unsigned long len;          /* total bytes fed, for the length trailer */
  unsigned char buf[64];
  unsigned buflen;
};

static const unsigned char slt_md5_shift[64] = {
  7,12,17,22, 7,12,17,22, 7,12,17,22, 7,12,17,22,
  5, 9,14,20, 5, 9,14,20, 5, 9,14,20, 5, 9,14,20,
  4,11,16,23, 4,11,16,23, 4,11,16,23, 4,11,16,23,
  6,10,15,21, 6,10,15,21, 6,10,15,21, 6,10,15,21
};

/* T[i] = floor(2^32 * abs(sin(i+1))). Tabulated rather than computed: the domain has no
   libm, and a table is one read-only global -- which under -capstone-gp-captable costs
   exactly one capability carve against the 1021-node pool. */
static const slt_u32 slt_md5_t[64] = {
  0xd76aa478u,0xe8c7b756u,0x242070dbu,0xc1bdceeeu,0xf57c0fafu,0x4787c62au,
  0xa8304613u,0xfd469501u,0x698098d8u,0x8b44f7afu,0xffff5bb1u,0x895cd7beu,
  0x6b901122u,0xfd987193u,0xa679438eu,0x49b40821u,0xf61e2562u,0xc040b340u,
  0x265e5a51u,0xe9b6c7aau,0xd62f105du,0x02441453u,0xd8a1e681u,0xe7d3fbc8u,
  0x21e1cde6u,0xc33707d6u,0xf4d50d87u,0x455a14edu,0xa9e3e905u,0xfcefa3f8u,
  0x676f02d9u,0x8d2a4c8au,0xfffa3942u,0x8771f681u,0x6d9d6122u,0xfde5380cu,
  0xa4beea44u,0x4bdecfa9u,0xf6bb4b60u,0xbebfbc70u,0x289b7ec6u,0xeaa127fau,
  0xd4ef3085u,0x04881d05u,0xd9d4d039u,0xe6db99e5u,0x1fa27cf8u,0xc4ac5665u,
  0xf4292244u,0x432aff97u,0xab9423a7u,0xfc93a039u,0x655b59c3u,0x8f0ccc92u,
  0xffeff47du,0x85845dd1u,0x6fa87e4fu,0xfe2ce6e0u,0xa3014314u,0x4e0811a1u,
  0xf7537e82u,0xbd3af235u,0x2ad7d2bbu,0xeb86d391u
};

static void slt_md5_init(struct slt_md5 *s) {
  s->h[0] = 0x67452301u; s->h[1] = 0xefcdab89u;
  s->h[2] = 0x98badcfeu; s->h[3] = 0x10325476u;
  s->len = 0; s->buflen = 0;
}

static void slt_md5_block(struct slt_md5 *s, const unsigned char *p) {
  slt_u32 m[16], a = s->h[0], b = s->h[1], c = s->h[2], d = s->h[3];
  unsigned i;
  for (i = 0; i < 16; i++)                    /* explicit LE decode: no unaligned loads */
    m[i] = (slt_u32)p[i * 4] | ((slt_u32)p[i * 4 + 1] << 8) |
           ((slt_u32)p[i * 4 + 2] << 16) | ((slt_u32)p[i * 4 + 3] << 24);
  for (i = 0; i < 64; i++) {
    slt_u32 f, tmp;
    unsigned g, sh = slt_md5_shift[i];
    if (i < 16)      { f = (b & c) | (~b & d);        g = i; }
    else if (i < 32) { f = (d & b) | (~d & c);        g = (5 * i + 1) & 15; }
    else if (i < 48) { f = b ^ c ^ d;                 g = (3 * i + 5) & 15; }
    else             { f = c ^ (b | SLT_M32(~d));     g = (7 * i) & 15; }
    f = SLT_M32(f + a + slt_md5_t[i] + m[g]);
    a = d; d = c; c = b;
    tmp = SLT_M32((f << sh) | (SLT_M32(f) >> (32 - sh)));
    b = SLT_M32(b + tmp);
  }
  s->h[0] = SLT_M32(s->h[0] + a); s->h[1] = SLT_M32(s->h[1] + b);
  s->h[2] = SLT_M32(s->h[2] + c); s->h[3] = SLT_M32(s->h[3] + d);
}

static void slt_md5_update(struct slt_md5 *s, const void *data, unsigned long n) {
  const unsigned char *p = (const unsigned char *)data;
  s->len += n;
  while (n) {
    unsigned room = 64u - s->buflen;
    unsigned take = (n < (unsigned long)room) ? (unsigned)n : room;
    unsigned k;
    for (k = 0; k < take; k++)
      s->buf[s->buflen + k] = p[k];
    s->buflen += take; p += take; n -= take;
    if (s->buflen == 64u) { slt_md5_block(s, s->buf); s->buflen = 0; }
  }
}

/* Writes 32 lowercase hex characters plus a NUL into `hex`. */
static void slt_md5_final(struct slt_md5 *s, char *hex) {
  static const char hexd[] = "0123456789abcdef";
  unsigned long bits = s->len * 8UL;
  unsigned char tail[8];
  unsigned char pad = 0x80u;
  unsigned i;
  slt_md5_update(s, &pad, 1);
  pad = 0x00u;
  while (s->buflen != 56u)
    slt_md5_update(s, &pad, 1);
  for (i = 0; i < 8; i++)
    tail[i] = (unsigned char)((bits >> (8 * i)) & 0xffu);
  slt_md5_update(s, tail, 8);
  for (i = 0; i < 16; i++) {
    unsigned byte = (unsigned)((s->h[i / 4] >> (8 * (i % 4))) & 0xffu);
    hex[i * 2]     = hexd[byte >> 4];
    hex[i * 2 + 1] = hexd[byte & 0xfu];
  }
  hex[32] = '\0';
}

#endif
