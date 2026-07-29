#ifndef GPSTRESS_KERNEL_H
#define GPSTRESS_KERNEL_H
/* DESCRIPTOR-STRESS rung: the smallest domain that exercises every path the
 * descriptor-driven entry glue takes for SQLite.
 *
 * WHY IT EXISTS. The C-13 bisection ran on `beebs_prime`, which at -O1 has exactly
 * ONE global: zero-init, `blob_off = -1`. So a green bisection proves the glue works
 * for `count == 1, zero-fill` and says nothing about the record loop, the copy path,
 * the byte tail, or private symbols -- i.e. nothing about what SQLite (1,059 globals,
 * ~910 private, mixed sizes) actually needs. Jumping from one to 1,059 is the kind of
 * leap that has repeatedly produced unattributable board failures here.
 *
 * Each global below is chosen to hit a DIFFERENT branch of the glue, so a failure is
 * attributable rather than just "SQLite hangs":
 *
 *   g_bss     zero-init          -> blob_off == -1, the pure zero-fill path
 *   g_init    256 B, size%8==0   -> the 8-byte bulk COPY loop
 *   g_odd     13 B, size%8!=0    -> the BYTE TAIL (`lb`/`sb`), unreachable in prime
 *   g_big     2400 B > 2040      -> past the generated glue's per-global store limit
 *   st_priv   static const       -> a PRIVATE (.L) symbol: the SQLite-dominant shape,
 *                                   which the generated glue cannot even LINK
 *   g_scalar  8 B                -> a plain scalar, and makes count == 6 so the
 *                                   record loop iterates rather than running once
 *
 * The kernel also STORES into g_bss, so the carved storage is proven writable and not
 * merely readable. Plain C on purpose: the same header compiles for the domain
 * (freestanding capstone clang) and for the native oracle host.
 */

static unsigned      g_bss[64];                     /* zero-init  -> blob_off -1     */
static unsigned char g_odd[13] = {                  /* size%8 != 0 -> byte tail       */
  3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9 };
unsigned             g_scalar = 0x9E3779B9u;        /* file-scope scalar              */

/* File-scope (non-.L) initialized array, size%8 == 0: the copy path the generated
   glue CAN take, kept so a divergence between the two glues stays visible. */
unsigned g_init[64] = {
#define I8(i) (i)*11u+1u,(i)*11u+2u,(i)*11u+3u,(i)*11u+4u,(i)*11u+5u,(i)*11u+6u,(i)*11u+7u,(i)*11u+8u
  I8(0), I8(1), I8(2), I8(3), I8(4), I8(5), I8(6), I8(7)
};

/* > 2040 B: beyond the generated glue's 12-bit store-offset limit for one global. */
unsigned g_big[600] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

/* PRIVATE symbol. `static const` => a .L symbol the generated glue must `lla` from a
   separate TU, which does not link. Only the compiler can name it, which is the whole
   reason the descriptor exists. */
static const unsigned st_priv[128] = {
#define P8(i) (i)*13u+1u,(i)*13u+2u,(i)*13u+3u,(i)*13u+4u,(i)*13u+5u,(i)*13u+6u,(i)*13u+7u,(i)*13u+8u
#define P64(i) P8(i),P8(i+1),P8(i+2),P8(i+3),P8(i+4),P8(i+5),P8(i+6),P8(i+7)
  P64(0), P64(8)
};

static unsigned gpstress_compute(void) {
  unsigned h = 2166136261u;
  /* Prove the zero-init storage is WRITABLE, then fold it in. */
  for (int i = 0; i < 64; i++) g_bss[i] = (unsigned)i * 3u + g_scalar;
  for (int r = 0; r < 8; r++) {
    for (int i = 0; i < 64; i++)  { h ^= g_bss[i];   h *= 16777619u; }
    for (int i = 0; i < 64; i++)  { h ^= g_init[i];  h *= 16777619u; }
    for (int i = 0; i < 13; i++)  { h ^= g_odd[i];   h *= 16777619u; }
    for (int i = 0; i < 600; i++) { h ^= g_big[i];   h *= 16777619u; }
    for (int i = 0; i < 128; i++) { h ^= st_priv[i]; h *= 16777619u; }
  }
  h ^= g_scalar; h *= 16777619u;
  return h;
}
#endif
