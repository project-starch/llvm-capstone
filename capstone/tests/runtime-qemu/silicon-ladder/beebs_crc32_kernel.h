#ifndef BEEBS_CRC32_KERNEL_H
#define BEEBS_CRC32_KERNEL_H
/* Silicon-ladder rung 3: BEEBS crc32 (a *found* benchmark).
 *
 * Faithful to the BEEBS crc32 CRC computation, with ONE deliberate adaptation:
 * the 256-entry crc_32_tab is COMPUTED at runtime (the standard 0xedb88320
 * generator loop -- byte-identical to the upstream const table) instead of being
 * a file-scope `const` initializer. Why: on this silicon-gp model the generator
 * materializes initialized globals as `li/sd` instruction immediates, and a 1 KiB
 * const table expands to ~2 KiB of .text, which (a) collides with the fixed
 * globals offset in link-gpfree.ld and (b) overflows the monitor's PCC code
 * window (all code must fit in [base, base+0x1000) for the silicon image SPLIT).
 * Large initialized *read-only* tables therefore need a different delivery
 * mechanism on silicon -- a tracked open item (SQLite will hit it). Runtime table
 * generation keeps the benchmark's real work intact and makes the table a 1 KiB
 * .bss array, so this rung instead validates: two globals (the table + a
 * function-local `static seed`), a 1 KiB .bss array reached via `ldc gp[i]`,
 * nested loops, and a real crc32pseudo->rand_beebs->UPDC32 call graph indexing
 * the table 1024x. The crc value is itself the oracle; a native host folds the
 * identical value (both compute the same table, so they agree by construction). */
typedef unsigned char  BYTE;
typedef unsigned long  DWORD;
typedef unsigned short WORD;
typedef DWORD UNS_32_BITS;

static UNS_32_BITS crc_32_tab[256];

/* Generate the standard reflected CRC-32 table (polynomial 0xedb88320). This
   reproduces the exact values of the upstream BEEBS static crc_32_tab[]. */
static void crc32_init_table(void) {
  unsigned n;
  int k;
  /* The polynomial is deliberately made OPAQUE to the optimizer.
   *
   * At -O0 this loop runs as written. At -O1+ LLVM constant-folds the whole
   * nest and re-materializes the result as a 2048 B *private* constant
   * (`.L.crctable`), which defeats the entire point of generating the table at
   * runtime (see the header comment): a large initialized read-only global then
   * has to be delivered into the domain, and the gp cap-table glue cannot do it
   * -- 2048 B overflows the unrolled 12-bit store-offset path, and the large-RO
   * copy path needs a *linkable* (non-`.L`) symbol, which a private constant is
   * not. Result: the rung failed to build at -O1 and above.
   *
   * One opaque register breaks the constant-folding without changing a single
   * runtime operation, so the table stays a 1 KiB .bss array at every -O level
   * and the emitted arithmetic is identical. */
  UNS_32_BITS poly = 0xedb88320UL;
  __asm__("" : "+r"(poly));
  for (n = 0; n < 256; n++) {
    UNS_32_BITS c = (UNS_32_BITS)n;
    for (k = 0; k < 8; k++)
      c = (c & 1u) ? (poly ^ (c >> 1)) : (c >> 1);
    crc_32_tab[n] = c & 0xffffffffUL;
  }
}

#define UPDC32(octet, crc) (crc_32_tab[((crc) ^ ((BYTE)octet)) & 0xff] ^ ((crc) >> 8))

static int rand_beebs(void) {
  static long int seed = 0;
  seed = (seed * 1103515245L + 12345) & ((1UL << 31) - 1);
  return (int)(seed >> 16);
}

static DWORD crc32pseudo(void) {
  int i;
  DWORD oldcrc32 = 0xFFFFFFFF;
  for (i = 0; i < 1024; ++i)
    oldcrc32 = UPDC32(rand_beebs(), oldcrc32);
  return ~oldcrc32;
}

static unsigned crc_compute(void) {
  crc32_init_table();
  return (unsigned)(int)crc32pseudo();
}
#endif
