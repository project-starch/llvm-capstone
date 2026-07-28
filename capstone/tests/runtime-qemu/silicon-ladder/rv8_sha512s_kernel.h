/* SMALL sha512 variant: 16-entry K table (128 B), so it fits the DEFAULT 4 KiB
 * window with the DEFAULT unrolled-immediate path -- no DOMAIN_WINDOW=32k, no
 * LADDER_NO_RO_COPY. Its only purpose is to separate two explanations for R-7:
 * if this PASSES on silicon while the full rung hangs, the 32 KiB window or the
 * ~8 KB unrolled prologue is implicated; if it HANGS too, R-1 owns it and the
 * C-5 workaround is exonerated. Same compression loop either way. */
#ifndef RV8_SHA512S_KERNEL_H
#define RV8_SHA512S_KERNEL_H
/* Silicon-ladder rung: RV8 `sha512` -- the crypto/bitwise profile the ladder lacks.
 *
 * Source: rv8-bench `sha512.c` (derived from SUPERCOP, MIT). The compression
 * function is verbatim: the same round macros, the same 80-round schedule, the
 * same shift-register update. Only the harness differs.
 *
 * WHY THIS ONE, out of the RV8 set. The others are blocked by known issues rather
 * than by anything about this measurement:
 *   - `aes`     ~8 KB of Te/Td tables -> C-4 (large read-only data cannot be
 *               delivered into a domain) and C-5 (4 KiB code window).
 *   - `dhrystone` 684 lines -> C-5.
 *   - `qsort`   sorts in place, i.e. a register-indexed load with an intervening
 *               store to the same array -> R-1, the hardware hazard.
 *   - `miniz`   C-2 (i128 or/xor with mixed extends) at -O1/-O2.
 * sha512's only large constant is `sha512s_k[80]` at 640 B, which is inside the
 * large-RO delivery path already QEMU-validated at 16 KiB, and its working set is
 * a 128-byte block plus 8 chaining words.
 *
 * WHAT IT ADDS. Every existing rung is integer arithmetic, array traversal, or
 * calls. This is 64-bit rotate/xor/and dominated with a long dependency chain
 * through the shift register, and it reads a large constant table by index --
 * a genuinely different mix, which is the point of adding it.
 *
 * SHAPE PREDICTION under R-1: PASS. `sha512s_k[i]` is a read-only indexed load with
 * nothing ever stored to that table, the same shape as `beebs_bs` which passes.
 * `w[i&15]` IS both read and written, so if R-1 is broader than characterised this
 * rung will find it -- worth knowing either way.
 *
 * ADAPTATION, and it is the only one: upstream hashes 1,000,000 * 64 bytes, far too
 * long for a bracketed board measurement. This hashes a deterministic 64-byte buffer
 * SHA_REPS times. The oracle is the FNV fold of the final digest, computed natively
 * from this same header, so the rung is self-checking exactly like the others. */

typedef unsigned long long sha_u64;
typedef unsigned char      sha_u8;

#define SHA_REPS 64

/* NOT `static`: the gp cap-table glue lives in a separate translation unit and
   references each delivered global by name, so a file-local symbol fails to link
   ("undefined symbol: sha512s_k"). That is issue C-4's large-RO delivery path --
   it needs a linkable, non-.L symbol. External linkage is the fix. */
const sha_u64 sha512s_init_state[8] = {
  0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
  0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
  0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull
};

const sha_u64 sha512s_k[16] = {
  0x428a2f98d728ae22ull,0x7137449123ef65cdull,0xb5c0fbcfec4d3b2full,0xe9b5dba58189dbbcull,
  0x3956c25bf348b538ull,0x59f111f1b605d019ull,0x923f82a4af194f9bull,0xab1c5ed5da6d8118ull,
  0xd807aa98a3030242ull,0x12835b0145706fbeull,0x243185be4ee4b28cull,0x550c7dc3d5ffb4e2ull,
  0x72be5d74f27b896full,0x80deb1fe3b1696b1ull,0x9bdc06a725c71235ull,0xc19bf174cf692694ull,
};

static sha_u64 sha_s_chain[8];
static sha_u64 sha_s_w[16];

static sha_u64 sha_rotr(sha_u64 x, int d) { return (x >> d) | (x << (64 - d)); }
static sha_u64 sha_S0(sha_u64 h) { return sha_rotr(h,28) ^ sha_rotr(h,34) ^ sha_rotr(h,39); }
static sha_u64 sha_S1(sha_u64 h) { return sha_rotr(h,14) ^ sha_rotr(h,18) ^ sha_rotr(h,41); }
static sha_u64 sha_s0(sha_u64 a) { return sha_rotr(a,1)  ^ sha_rotr(a,8)  ^ (a >> 7); }
static sha_u64 sha_s1(sha_u64 b) { return sha_rotr(b,19) ^ sha_rotr(b,61) ^ (b >> 6); }
static sha_u64 sha_ch (sha_u64 e, sha_u64 f, sha_u64 g) { return g ^ (e & (g ^ f)); }
static sha_u64 sha_maj(sha_u64 a, sha_u64 b, sha_u64 c) { return (a & b) ^ (c & (a ^ b)); }

/* Verbatim compression function over the 16 words already in sha_s_w. */
static void sha512s_process_block(void) {
  sha_u64 i, tmp, a, b;
  sha_u64 h0 = sha_s_chain[0], h1 = sha_s_chain[1], h2 = sha_s_chain[2], h3 = sha_s_chain[3],
          h4 = sha_s_chain[4], h5 = sha_s_chain[5], h6 = sha_s_chain[6], h7 = sha_s_chain[7];

  for (i = 0; i < 16; i++) {
    tmp = sha_s_w[i] + h7 + sha_S1(h4) + sha_ch(h4,h5,h6) + sha512s_k[i & 15];
    h7 = h6; h6 = h5; h5 = h4;
    h4 = h3 + tmp;
    h3 = h2; h2 = h1; h1 = h0;
    h0 = tmp + sha_maj(h1,h2,h3) + sha_S0(h1);
  }
  for (; i < 16; i++) {
    a = sha_s_w[(i+1)  & 15];
    b = sha_s_w[(i+14) & 15];
    tmp = sha_s_w[i & 15] = sha_s0(a) + sha_s1(b) + sha_s_w[i & 15] + sha_s_w[(i+9) & 15];
    tmp = tmp + h7 + sha_S1(h4) + sha_ch(h4,h5,h6) + sha512s_k[i & 15];
    h7 = h6; h6 = h5; h5 = h4;
    h4 = h3 + tmp;
    h3 = h2; h2 = h1; h1 = h0;
    h0 = tmp + sha_maj(h1,h2,h3) + sha_S0(h1);
  }
  sha_s_chain[0] += h0; sha_s_chain[1] += h1; sha_s_chain[2] += h2; sha_s_chain[3] += h3;
  sha_s_chain[4] += h4; sha_s_chain[5] += h5; sha_s_chain[6] += h6; sha_s_chain[7] += h7;
}

static unsigned sha512s_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < SHA_REPS; rep++) {
    for (int i = 0; i < 8; i++) sha_s_chain[i] = sha512s_init_state[i];
    /* Deterministic block, varied per rep so no repetition can be folded away. */
    for (int i = 0; i < 16; i++)
      sha_s_w[i] = 0x0123456789abcdefull * (sha_u64)(i + 1) + (sha_u64)rep;
    sha512s_process_block();
    for (int i = 0; i < 8; i++) {
      h ^= (unsigned)(sha_s_chain[i] & 0xffffffffu); h *= 16777619u;
      h ^= (unsigned)(sha_s_chain[i] >> 32);         h *= 16777619u;
    }
  }
  return h;
}
#endif
