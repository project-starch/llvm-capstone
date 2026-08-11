#ifndef S06SCALE_KERNEL_H
#define S06SCALE_KERNEL_H
/* DOES THE S-06 FIXUP'S STORE PATTERN WEDGE AT SCALE, WITH SQLITE REMOVED?
 *
 * This is the experiment three previous attempts failed to run, and the reason they failed is
 * worth stating because it dictates the whole design here.
 *
 * The fixup (plain-store both halves, then lay `ldc`/`stc` on top) is correct in simulation and
 * on a small rung, yet WEDGES SQLite with mcause 25. Every attempt to localise that used SQLite
 * arms, and every one of them was CONFOUNDED, because repairing the data also makes SQLite
 * execute more code -- depth and data-correctness are coupled by construction, so no SQLite pair
 * can separate "the store pattern is toxic" from "correct data reaches a second fault".
 *
 * The only way out is to delete SQLite from the experiment. Here there is NO data-dependent
 * control flow at all: both arms touch the same bytes, in the same order, for the same number of
 * iterations, and differ ONLY in the per-chunk store pattern.
 *
 *   s06scale_fix()   ld, ld, ldc, sd, sd, stc     the fixup's pattern
 *   s06scale_base()  ldc, stc                     the baseline's pattern
 *
 * Read the pair, not either alone:
 *   fix wedges, base returns  -> the store pattern is intrinsically toxic at scale. Every fix
 *                                using it is dead, including the shipped workaround.
 *   both return               -> the pattern is exonerated at scale, and SQLite's wedge is a
 *                                second fault reachable only once the data is correct.
 *   both wedge                -> something in this rung is wrong, not the pattern. Say so; do
 *                                not report a verdict.
 *
 * WORKING SET. The wedge is documented as appearing "only at scale" and never on a hot 10 KB
 * rung, so the buffers must exceed the D-cache: it is 32 KB, 8-way, 256 sets of 16 B
 * (capstone_cv64a6_imafdc_sv39_config_pkg.sv:48-50). 32 KB source + 32 KB destination gives a
 * 64 KB working set, so a single linear pass evicts every line it touched. Iterating a SMALL
 * buffer many times would NOT do this -- it would stay resident and reproduce the hot-line
 * conditions under which the fixup already passes.
 *
 * CAPABILITIES ARE INTERSPERSED, one per 256 bytes, because a buffer of pure plain data would
 * never exercise the `stc`-writes-both-banks path and the pattern would not be the pattern under
 * test. Their presence is also what makes the baseline arm meaningful: it must preserve them.
 *
 * VERDICT: the number of 16-byte chunks that came back correct, capped into a small distinctive
 * range. A wrong COUNT is bisectable where a hang is not, which is why this returns a number
 * rather than a pass flag.
 */

#define S06SCALE_BYTES   (32u * 1024u)
#define S06SCALE_CHUNKS  (S06SCALE_BYTES / 16u)
#define S06SCALE_CAPEVERY 16u            /* one capability per 16 chunks = per 256 bytes */

__attribute__((aligned(16))) static unsigned char s06scale_src[S06SCALE_BYTES];
__attribute__((aligned(16))) static unsigned char s06scale_dst[S06SCALE_BYTES];

/* ldc rd, 0(rs) / stc rs2, 0(rs1) as raw encodings: the assembler has no mnemonic for these. */
#define S06_LDC(out, addr) \
  __asm__ volatile(".insn i 0x5b, 0x3, %0, 0(%1)" : "=r"(out) : "r"(addr))
#define S06_STC(val, addr) \
  __asm__ volatile(".insn s 0x5b, 0x4, %0, 0(%1)" :: "r"(val), "r"(addr) : "memory")

static void s06scale_fill(void)
{
  volatile unsigned long *s = (volatile unsigned long *)s06scale_src;
  unsigned i;
  for (i = 0; i < S06SCALE_CHUNKS; i++) {
    s[2*i]     = 0x0123456789abcdefUL ^ (unsigned long)i;
    s[2*i + 1] = 0xfedcba9876543210UL ^ (unsigned long)i;
  }
  /* Sprinkle real capabilities. Written with stc so they are genuinely tagged; a plain store
     would leave untagged bytes that merely look like a pointer. */
  for (i = 0; i < S06SCALE_CHUNKS; i += S06SCALE_CAPEVERY) {
    void *cap = (void *)&s06scale_dst[0];
    S06_STC(cap, &s06scale_src[i * 16u]);
  }
}

/* Count chunks whose LOW half survived. The low half is the half S-06 never loses, so a wrong
   count here means something worse than S-06 happened -- which is exactly what this is watching
   for. Capability chunks are skipped: their low word is a cursor, not the fill pattern. */
static unsigned s06scale_check(void)
{
  volatile unsigned long *d = (volatile unsigned long *)s06scale_dst;
  unsigned i, ok = 0;
  for (i = 0; i < S06SCALE_CHUNKS; i++) {
    if ((i % S06SCALE_CAPEVERY) == 0) { ok++; continue; }
    if (d[2*i] == (0x0123456789abcdefUL ^ (unsigned long)i)) ok++;
  }
  return ok;
}

/* THE FIXUP PATTERN: read both halves plainly, then ldc/stc on top. */
static unsigned s06scale_fix(void)
{
  unsigned i;
  s06scale_fill();
  for (i = 0; i < S06SCALE_CHUNKS; i++) {
    unsigned char *sp = &s06scale_src[i * 16u];
    unsigned char *dp = &s06scale_dst[i * 16u];
    unsigned long lo = *(volatile unsigned long *)sp;
    unsigned long hi = *(volatile unsigned long *)(sp + 8);
    void *cap;
    S06_LDC(cap, sp);
    *(volatile unsigned long *)dp       = lo;
    *(volatile unsigned long *)(dp + 8) = hi;
    S06_STC(cap, dp);                    /* capability store LAST -- the pattern under test */
  }
  return s06scale_check();
}

/* THE BASELINE PATTERN: ldc/stc only. Same bytes, same order, same iteration count. */
static unsigned s06scale_base(void)
{
  unsigned i;
  s06scale_fill();
  for (i = 0; i < S06SCALE_CHUNKS; i++) {
    unsigned char *sp = &s06scale_src[i * 16u];
    unsigned char *dp = &s06scale_dst[i * 16u];
    void *cap;
    S06_LDC(cap, sp);
    S06_STC(cap, dp);
  }
  return s06scale_check();
}
#endif /* S06SCALE_KERNEL_H */
