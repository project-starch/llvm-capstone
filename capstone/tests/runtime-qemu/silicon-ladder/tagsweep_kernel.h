#ifndef TAGSWEEP_H
#define TAGSWEEP_H
/* TAGSWEEP -- does a capability stored in memory and reloaded come back UNTAGGED?
 *
 * WHY THIS EXISTS. Every S-07 instance reduces to one sentence: a capability held in memory and
 * reloaded later comes back untagged. Provenance is not the mechanism -- a monitor-granted
 * shared-region cap (output_text+0xdc), a global's address stored into a heap struct
 * (sqlite3OsRead's pMethods) and an ordinary heap cap (sqlite3DbMallocRawNN+0xd8) all fail. The
 * smallest thing that reproduces it is still a 1.5 MB SQLite image, which is a terrible
 * reproducer to hand anyone.
 *
 * WHAT IS ACTUALLY NEW HERE, because four synthetic rungs already came back clean and repeating
 * them would waste a boot. s06spill/s06bnds/s06wr/s06pld (spill-reload round trips) PASS,
 * s07chase (20k dependent ldc hops) and s07indep both returned 0. The ONE arm that targeted the
 * memory path -- s07evict -- is recorded VOID, not negative: it assumed 64-byte cache lines.
 * Verified in the config we actually build, capstone_cv64a6_imafdc_sv39_config_pkg.sv:48-50,
 * the D-cache is 32768 B / 8-way / 128-bit lines, i.e. 16-byte lines, 4 KiB per way, 256 sets.
 * So the eviction it relied on never happened and that axis has never been tested.
 *
 * WHY NO EVICTION LOOP IS NEEDED AT ALL. The same config makes the D-cache WRITE-THROUGH,
 * NO-WRITE-ALLOCATE. A stc therefore does not allocate: after the store pass the slots are in
 * DRAM and NOT in the cache, so the first reload of each slot is necessarily a MISS REFILL --
 * which is exactly the path the measured wedge implicates (src=1, MISS REFILL). Sizing the
 * buffer past the cache gets the suspect path by construction, with no eviction loop to get
 * wrong. That is the whole design: it is the one thing s07evict was trying to do and failed at.
 *
 * WHY IT CANNOT WEDGE. The reload is checked with lcc field 1, the TOTAL type query, which
 * returns 7 for NOT_CAP without raising (capstone_dyn_unit.anvil:195). So a lost tag is COUNTED
 * rather than fatal, and the run always returns a number -- unlike every wedging repro to date,
 * which yields one bit and destroys its own reporting channel.
 *
 * THE SEEDED POSITIVE CONTROL IS NOT OPTIONAL. TAGSWEEP_SEED slots are deliberately clobbered
 * with a scalar store, which drops the granule's tag, so the counter MUST report at least
 * SEED*REPS. If it does not, the instrument is broken and the run is discarded rather than read
 * as "no tag loss". s07evict is precisely the case of an unproven instrument being read as a
 * clean result, and it cost a board session.
 *
 * SEED MUST BE 0 UNDER QEMU. op_helper.c:719 `assert(rs1_v->tag)` fires before any selector
 * check, so QEMU ABORTS on a type query of an untagged value where silicon returns 7. The pass
 * path validates under emulation; the control only runs on the board. That divergence is a real
 * QEMU gap, noted rather than worked around here.
 */

#ifndef TAGSWEEP_N
#define TAGSWEEP_N 4096u        /* x16 B = 64 KiB, past the 32 KiB cache -> every reload misses */
#endif
#ifndef TAGSWEEP_REPS
#define TAGSWEEP_REPS 1024u
#endif
#ifndef TAGSWEEP_SEED
#define TAGSWEEP_SEED 3u        /* 0 for the QEMU arm; see above */
#endif

#define TAGSWEEP_OK    0xA5000000u
#define TAGSWEEP_FAULT 0xEE000000u

/* volatile per element: the store and the reload must both survive to the artifact. A
 * repeat-the-load-N-times ladder on this project was once CSE'd into ONE ldc regardless of N,
 * and the whole set reported with total confidence having tested nothing. The disassembly is
 * checked before the boot as well -- this is belt and braces on purpose. */
static void *volatile tagsweep_slots[TAGSWEEP_N];
static unsigned tagsweep_anchor;

/* lcc rd, rs1, 1 -- rs2=x1 encodes the selector. Same encoding as cap_q_type in the SQLite
 * domain (sqlite_capstone_domain.c:498); deliberately not re-derived. */
static unsigned long tagsweep_type(const void *p) {
  unsigned long v = 0;
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(v) : "r"(p));
  return v;
}

static unsigned tagsweep_compute(void)
{
  unsigned long lost = 0, seeded_lost = 0;
  unsigned i, r;
  /* A real capability with the same provenance as the dominant failing site: the address of a
     global, which under the gp-captable ABI arrives as `ldc gp[i]` from the cap table. */
  void *base = (void *)&tagsweep_anchor;

  for (r = 0; r < TAGSWEEP_REPS; r++) {
    for (i = 0; i < TAGSWEEP_N; i++)
      tagsweep_slots[i] = base;                       /* stc: write-through, no allocate */

    for (i = 0; i < TAGSWEEP_SEED; i++)               /* scalar store drops the granule tag */
      *(volatile unsigned long *)(void *)&tagsweep_slots[i] = 0x5EEDul + i;

    for (i = 0; i < TAGSWEEP_N; i++) {
      void *p = tagsweep_slots[i];                    /* ldc: must refill from DRAM */
      if (tagsweep_type(p) == 7ul) {                  /* 7 == NOT_CAP */
        lost++;
        if (i < TAGSWEEP_SEED) seeded_lost++;
      }
    }
  }

  /* The control decides whether the number below means anything at all. */
  if (seeded_lost != (unsigned long)TAGSWEEP_SEED * (unsigned long)TAGSWEEP_REPS)
    return TAGSWEEP_FAULT;
  return TAGSWEEP_OK | (unsigned)((lost - seeded_lost) & 0x00FFFFFFul);
}
#endif
