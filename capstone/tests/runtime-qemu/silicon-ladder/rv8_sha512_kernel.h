#ifndef RV8_SHA512_KERNEL_H
#define RV8_SHA512_KERNEL_H
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
 * sha512's only large constant is `sha512_k[80]` at 640 B, which is inside the
 * large-RO delivery path already QEMU-validated at 16 KiB, and its working set is
 * a 128-byte block plus 8 chaining words.
 *
 * WHAT IT ADDS. Every existing rung is integer arithmetic, array traversal, or
 * calls. This is 64-bit rotate/xor/and dominated with a long dependency chain
 * through the shift register, and it reads a large constant table by index --
 * a genuinely different mix, which is the point of adding it.
 *
 * SHAPE PREDICTION under R-1: PASS. `sha512_k[i]` is a read-only indexed load with
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
   ("undefined symbol: sha512_k"). That is issue C-4's large-RO delivery path --
   it needs a linkable, non-.L symbol. External linkage is the fix. */
const sha_u64 sha512_init_state[8] = {
  0x6a09e667f3bcc908ull, 0xbb67ae8584caa73bull, 0x3c6ef372fe94f82bull,
  0xa54ff53a5f1d36f1ull, 0x510e527fade682d1ull, 0x9b05688c2b3e6c1full,
  0x1f83d9abfb41bd6bull, 0x5be0cd19137e2179ull
};

const sha_u64 sha512_k[80] = {
  0x428a2f98d728ae22ull,0x7137449123ef65cdull,0xb5c0fbcfec4d3b2full,0xe9b5dba58189dbbcull,
  0x3956c25bf348b538ull,0x59f111f1b605d019ull,0x923f82a4af194f9bull,0xab1c5ed5da6d8118ull,
  0xd807aa98a3030242ull,0x12835b0145706fbeull,0x243185be4ee4b28cull,0x550c7dc3d5ffb4e2ull,
  0x72be5d74f27b896full,0x80deb1fe3b1696b1ull,0x9bdc06a725c71235ull,0xc19bf174cf692694ull,
  0xe49b69c19ef14ad2ull,0xefbe4786384f25e3ull,0x0fc19dc68b8cd5b5ull,0x240ca1cc77ac9c65ull,
  0x2de92c6f592b0275ull,0x4a7484aa6ea6e483ull,0x5cb0a9dcbd41fbd4ull,0x76f988da831153b5ull,
  0x983e5152ee66dfabull,0xa831c66d2db43210ull,0xb00327c898fb213full,0xbf597fc7beef0ee4ull,
  0xc6e00bf33da88fc2ull,0xd5a79147930aa725ull,0x06ca6351e003826full,0x142929670a0e6e70ull,
  0x27b70a8546d22ffcull,0x2e1b21385c26c926ull,0x4d2c6dfc5ac42aedull,0x53380d139d95b3dfull,
  0x650a73548baf63deull,0x766a0abb3c77b2a8ull,0x81c2c92e47edaee6ull,0x92722c851482353bull,
  0xa2bfe8a14cf10364ull,0xa81a664bbc423001ull,0xc24b8b70d0f89791ull,0xc76c51a30654be30ull,
  0xd192e819d6ef5218ull,0xd69906245565a910ull,0xf40e35855771202aull,0x106aa07032bbd1b8ull,
  0x19a4c116b8d2d0c8ull,0x1e376c085141ab53ull,0x2748774cdf8eeb99ull,0x34b0bcb5e19b48a8ull,
  0x391c0cb3c5c95a63ull,0x4ed8aa4ae3418acbull,0x5b9cca4f7763e373ull,0x682e6ff3d6b2b8a3ull,
  0x748f82ee5defb2fcull,0x78a5636f43172f60ull,0x84c87814a1f0ab72ull,0x8cc702081a6439ecull,
  0x90befffa23631e28ull,0xa4506cebde82bde9ull,0xbef9a3f7b2c67915ull,0xc67178f2e372532bull,
  0xca273eceea26619cull,0xd186b8c721c0c207ull,0xeada7dd6cde0eb1eull,0xf57d4f7fee6ed178ull,
  0x06f067aa72176fbaull,0x0a637dc5a2c898a6ull,0x113f9804bef90daeull,0x1b710b35131c471bull,
  0x28db77f523047d84ull,0x32caab7b40c72493ull,0x3c9ebe0a15c9bebcull,0x431d67c49c100d4cull,
  0x4cc5d4becb3e42b6ull,0x597f299cfc657e2aull,0x5fcb6fab3ad6faecull,0x6c44198c4a475817ull
};

static sha_u64 sha_chain[8];
static sha_u64 sha_w[16];

static sha_u64 sha_rotr(sha_u64 x, int d) { return (x >> d) | (x << (64 - d)); }
static sha_u64 sha_S0(sha_u64 h) { return sha_rotr(h,28) ^ sha_rotr(h,34) ^ sha_rotr(h,39); }
static sha_u64 sha_S1(sha_u64 h) { return sha_rotr(h,14) ^ sha_rotr(h,18) ^ sha_rotr(h,41); }
static sha_u64 sha_s0(sha_u64 a) { return sha_rotr(a,1)  ^ sha_rotr(a,8)  ^ (a >> 7); }
static sha_u64 sha_s1(sha_u64 b) { return sha_rotr(b,19) ^ sha_rotr(b,61) ^ (b >> 6); }
static sha_u64 sha_ch (sha_u64 e, sha_u64 f, sha_u64 g) { return g ^ (e & (g ^ f)); }
static sha_u64 sha_maj(sha_u64 a, sha_u64 b, sha_u64 c) { return (a & b) ^ (c & (a ^ b)); }

/* Verbatim compression function over the 16 words already in sha_w. */
static void sha512_process_block(void) {
  sha_u64 i, tmp, a, b;
  sha_u64 h0 = sha_chain[0], h1 = sha_chain[1], h2 = sha_chain[2], h3 = sha_chain[3],
          h4 = sha_chain[4], h5 = sha_chain[5], h6 = sha_chain[6], h7 = sha_chain[7];

  for (i = 0; i < 16; i++) {
    tmp = sha_w[i] + h7 + sha_S1(h4) + sha_ch(h4,h5,h6) + sha512_k[i];
    h7 = h6; h6 = h5; h5 = h4;
    h4 = h3 + tmp;
    h3 = h2; h2 = h1; h1 = h0;
    h0 = tmp + sha_maj(h1,h2,h3) + sha_S0(h1);
  }
  for (; i < 80; i++) {
    a = sha_w[(i+1)  & 15];
    b = sha_w[(i+14) & 15];
    tmp = sha_w[i & 15] = sha_s0(a) + sha_s1(b) + sha_w[i & 15] + sha_w[(i+9) & 15];
    tmp = tmp + h7 + sha_S1(h4) + sha_ch(h4,h5,h6) + sha512_k[i];
    h7 = h6; h6 = h5; h5 = h4;
    h4 = h3 + tmp;
    h3 = h2; h2 = h1; h1 = h0;
    h0 = tmp + sha_maj(h1,h2,h3) + sha_S0(h1);
  }
  sha_chain[0] += h0; sha_chain[1] += h1; sha_chain[2] += h2; sha_chain[3] += h3;
  sha_chain[4] += h4; sha_chain[5] += h5; sha_chain[6] += h6; sha_chain[7] += h7;
}

static unsigned sha512_compute(void) {
  unsigned h = 2166136261u;
  for (int rep = 0; rep < SHA_REPS; rep++) {
    for (int i = 0; i < 8; i++) sha_chain[i] = sha512_init_state[i];
    /* Deterministic block, varied per rep so no repetition can be folded away. */
    for (int i = 0; i < 16; i++)
      sha_w[i] = 0x0123456789abcdefull * (sha_u64)(i + 1) + (sha_u64)rep;
    sha512_process_block();
    for (int i = 0; i < 8; i++) {
      h ^= (unsigned)(sha_chain[i] & 0xffffffffu); h *= 16777619u;
      h ^= (unsigned)(sha_chain[i] >> 32);         h *= 16777619u;
    }
  }
  return h;
}
#endif
