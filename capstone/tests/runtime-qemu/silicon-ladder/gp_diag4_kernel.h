#ifndef GP_DIAG4_KERNEL_H
#define GP_DIAG4_KERNEL_H
/* Silicon-ladder DIAGNOSTIC rung v4: dump RAW PER-ELEMENT READBACKS.
 *
 * WHY. v3 asked "is iterated shared-region access the fault?" and answered NO,
 * twice over: a loop over a purely LOCAL STACK array (no shared region, no
 * globals) came back wrong, and a STRAIGHT-LINE read of the shared region (no
 * loop) also came back wrong. Neither the region nor a loop is necessary. See
 *   history/25-07-2026_19-xx_gp-diag-v3-...md
 *
 * But every v3 probe reported a SUM over 8 reads, so a wrong probe conflates
 * eight unknowns. Decomposing those sums suggests the bad reads return values
 * like 0x08000000 and 0x819B7F90 -- a 128 MiB length and a DRAM address, i.e.
 * CAPABILITY METADATA rather than the stored data. That is only an inference
 * from sums, and inference-from-a-fold is precisely the mistake v1 made (see the
 * checksum-inversion dead end). So v4 reports the ACTUAL WORDS.
 *
 * DESIGN RULE: there is NOT ONE LOOP in this kernel. Every seed and every
 * readback is straight-line, because v3 showed straight-line code already
 * reproduces the fault, and a loop would only re-introduce a construct that is
 * itself under suspicion.
 *
 * Slot map (controller prints res[3..] as dbg0..; LADDER_DBG_SLOTS must be >=36):
 *   dbg0 ..dbg7   res[W1+k] read back straight-line, FIRST pass   expect 256<<k
 *   dbg8 ..dbg15  larr[k]   (LOCAL STACK array)                   expect 256<<k
 *   dbg16..dbg23  garr[k]   (GLOBAL via the gp cap-table)         expect 256<<k
 *   dbg24..dbg31  res[W1+k] read back again, SECOND pass          expect 256<<k
 *   dbg32         canary                                          expect 0xC0FFEE
 *
 * HOW TO READ IT:
 *   - first pass wrong  -> the straight-line STORE never landed, or the load is
 *     broken; either way it is not about lifetime or clobbering.
 *   - first pass right, second wrong -> something CLOBBERS the window in between.
 *   - larr wrong but garr right -> the fault is on the STACK capability, not the
 *     gp cap-table (v3's dbg0-vs-dbg4 split said exactly this).
 *   - a wrong word that is a DRAM ADDRESS or a power-of-two LENGTH is capability
 *     metadata leaking into a data load -- report the exact value, do not fold it.
 *
 * Everything lives in domain_main: v3 put the work in a noinline helper taking
 * the region capability as an ARGUMENT, which v2 never did. That is an
 * uncontrolled difference between the clean v2 run and the dirty v3 one, so v4
 * removes it rather than leaving it in the experiment. */

#define GPD4_CANARY   0xC0FFEEUL
#define GPD4_N        8
#define GPD4_W1       40          /* res[40..48): data window, clear of the 36 dbg slots */
#define GPD4_NSLOT    33          /* dbg0..dbg32 */
#define GPD4_SEED(i)  (256UL << (i))
#define GPD4_SUM      65280UL

static unsigned long gpd4_garr[GPD4_N];

/* Seed a window with straight-line stores. */
#define GPD4_SEED8(dst, base)                                                  \
  do {                                                                         \
    (dst)[(base) + 0] = GPD4_SEED(0);  (dst)[(base) + 1] = GPD4_SEED(1);       \
    (dst)[(base) + 2] = GPD4_SEED(2);  (dst)[(base) + 3] = GPD4_SEED(3);       \
    (dst)[(base) + 4] = GPD4_SEED(4);  (dst)[(base) + 5] = GPD4_SEED(5);       \
    (dst)[(base) + 6] = GPD4_SEED(6);  (dst)[(base) + 7] = GPD4_SEED(7);       \
  } while (0)

/* Copy 8 words into consecutive debug slots, straight-line. */
#define GPD4_DUMP8(slot, src, base)                                            \
  do {                                                                         \
    res[(slot) + 0] = (src)[(base) + 0];  res[(slot) + 1] = (src)[(base) + 1]; \
    res[(slot) + 2] = (src)[(base) + 2];  res[(slot) + 3] = (src)[(base) + 3]; \
    res[(slot) + 4] = (src)[(base) + 4];  res[(slot) + 5] = (src)[(base) + 5]; \
    res[(slot) + 6] = (src)[(base) + 6];  res[(slot) + 7] = (src)[(base) + 7]; \
  } while (0)

#define GPD4_SUM8(src, base)                                                   \
  ((src)[(base) + 0] + (src)[(base) + 1] + (src)[(base) + 2] +                 \
   (src)[(base) + 3] + (src)[(base) + 4] + (src)[(base) + 5] +                 \
   (src)[(base) + 6] + (src)[(base) + 7])

/* Fills res[3 .. 3+GPD4_NSLOT) and returns the straight-line sum of the
   second-pass window read (the harness's oracle gate).
   always_inline (honoured even at -O0) so the body really does end up in
   domain_main: a plain `static` callee would still be a real call taking the
   region CAPABILITY as an argument, which is the v3 confound this rung removes. */
__attribute__((always_inline)) static inline unsigned long
gpd4_run(unsigned long *res) {
  unsigned long larr[GPD4_N];

  GPD4_SEED8(res, GPD4_W1);
  GPD4_SEED8(larr, 0);
  GPD4_SEED8(gpd4_garr, 0);

  GPD4_DUMP8(3 + 0,  res,       GPD4_W1);   /* dbg0..7   shared region, pass 1 */
  GPD4_DUMP8(3 + 8,  larr,      0);         /* dbg8..15  local stack array     */
  GPD4_DUMP8(3 + 16, gpd4_garr, 0);         /* dbg16..23 global via gp table   */
  GPD4_DUMP8(3 + 24, res,       GPD4_W1);   /* dbg24..31 shared region, pass 2 */
  res[3 + 32] = GPD4_CANARY;                /* dbg32                           */

  return GPD4_SUM8(res, GPD4_W1);
}
#endif
