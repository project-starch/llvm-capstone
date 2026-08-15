#ifndef S07CHASE_KERNEL_H
#define S07CHASE_KERNEL_H
/* BUILD RECIPE -- without this the independent arm cannot be rebuilt by anyone but us:
 *
 *   RUNG=s07chase  bash tests/runtime-qemu/silicon-ladder/verify-and-stage-rung.sh s07chase
 *   RUNG=s07indep  DOMAIN_EXTRA_CFLAGS=-DS07CHASE_INDEPENDENT \
 *                  bash tests/runtime-qemu/silicon-ladder/verify-and-stage-rung.sh s07chase
 *   selftest arms: add -DS07CHASE_SELFTEST (must return >= 0x100)
 *
 * BOTH ARMS ARE VOID AS TESTS OF A-1, and the reason is in this file's history rather than the
 * hardware: at -O0 (build-ladder-domain.sh:73) every loaded capability is spilled immediately, so
 * 18 of s07indep's 43 `ldc`s have their result consumed ONE instruction later. Neither arm ever
 * puts two capability loads in flight. Their zeros are real measurements of nothing. Anyone
 * rebuilding this must use inline asm and verify the emitted burst by disassembly BEFORE spending
 * a boot.
 */
/* S-07 MINIMAL REPRODUCER ATTEMPT: back-to-back DEPENDENT capability loads.
 *
 * WHERE THIS SHAPE COMES FROM -- a measured fault, not a guess. On 2026-08-15, on the
 * S-06-fixed bitstream `caplifive_s06fixs08fix.bit`, with NO software workarounds and NO
 * instrumentation, the full SQLite workload wedged on its FIRST execution at
 * `sqlite3OsRead+0x4c` with mcause 25:
 *
 *     3a834:  ldc  a0, 0x0(a0)
 *     3a838:  ldc  a4, 0x0(a0)      <- a4 is produced BY an ldc
 *     3a83c:  ldc  a4, 0x20(a4)     <== mcause 25: a4 is NOT_CAP
 *
 * `ldc`'s guard is rs1-only (capstone_dyn_unit.anvil:327-330), so unlike the earlier
 * `cincoffset` sites this is UNAMBIGUOUS: the capability an `ldc` produced arrived untagged,
 * and the immediately dependent `ldc` raised on it.
 *
 * WHY THIS SHAPE AND NOT ANOTHER. The leading RTL hypothesis (A-1 in
 * fpga-repros/S07-.../rtl/MECHANISMS-AND-PATCH-PROPOSAL.md) is that the capability load syncer
 * tracks exactly ONE outstanding request -- one `cap_trans_id` register plus one `req_set` bit,
 * overwritten unconditionally by any later arming -- so a second capability load issued while
 * one is in flight DISPLACES the first, and the displaced response is bypassed to a writeback
 * port that carries no capability. A dependent pair of `ldc`s is the minimal construct that puts
 * two capability loads in flight back to back. That is exactly what the measured site does.
 *
 * The four existing rungs (s06spill/bnds/wr/pld) all spill ONE capability and reload it. None of
 * them issues a second capability load while the first is outstanding. That is the untested gap
 * this rung is aimed at, and it is why they all return 0xFFFF while SQLite wedges.
 *
 * METHOD. A chain of capabilities in a static array: each cell holds a pointer to the next cell.
 * Chasing it issues exactly the measured pattern -- `ldc` of the next link, then an immediately
 * dependent `ldc` through it. After each hop the loaded value's TYPE is queried with LCC field 1,
 * which is TOTAL on this silicon, so a lost tag is REPORTED rather than faulted on and the rung
 * returns a number instead of wedging.
 *
 * retval:
 *   0            no untagged link seen in S07CHASE_HOPS hops -- this shape does NOT reproduce it
 *   n > 0        the number of hops whose loaded capability came back NOT_CAP. THE REPRODUCER.
 *
 * A zero here is informative but NOT exoneration: SQLite wedges with a warm cache, a 256 KiB
 * heap and far more capability traffic than a tight loop creates. Report it as "this shape alone
 * is not sufficient", never as "the mechanism is refuted".
 */

#ifndef S07CHASE_LINKS
#define S07CHASE_LINKS 64
#endif
#ifndef S07CHASE_HOPS
#define S07CHASE_HOPS 20000
#endif
/* Links chased back-to-back with NO intervening query. This is the whole point of the rung:
   the A-1 mechanism needs a second capability load issued while the first is still outstanding,
   and any instruction that consumes the loaded value serialises them. */
#ifndef S07CHASE_BURST
#define S07CHASE_BURST 8
#endif

/* 16-byte aligned so every cell is a capability granule. */
__attribute__((aligned(16))) static void *s07chase_ring[S07CHASE_LINKS];

#define S07CHASE_LCC_TYPE(out_, cap_) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out_) : "r"(cap_))

#define S07CHASE_BARRIER() __asm__ volatile("" ::: "memory")

static unsigned s07chase_compute(void)
{
  unsigned bad = 0;
  unsigned i;

  /* Build the ring: cell i points at cell i+1, last points back at cell 0. Every stored value is
     a real capability, so every load below MUST come back tagged. */
  for (i = 0; i < S07CHASE_LINKS; i++)
    s07chase_ring[i] = (void *)&s07chase_ring[(i + 1u) % S07CHASE_LINKS];

  S07CHASE_BARRIER();

  {
    void **p = (void **)&s07chase_ring[0];
    unsigned h;
    unsigned b_seed = 0;
    (void)b_seed;
    for (h = 0; h < S07CHASE_HOPS; h += S07CHASE_BURST) {
      unsigned long ty_;
      unsigned b;

      /* THE QUERY MUST NOT SIT BETWEEN THE LOADS. An `lcc` consumes the loaded capability and
         forces the load to complete, which serialises exactly the overlap this rung exists to
         create -- the first version did that and would have measured nothing while looking
         clean. So chase S07CHASE_BURST links back to back with NOTHING in between, which is the
         measured pattern (ldc -> immediately dependent ldc), and query only once per burst. */
#ifdef S07CHASE_INDEPENDENT
      /* INDEPENDENT loads, which is what A-1 actually needs and what the dependent chase
         CANNOT provide. A dependent `ldc` needs the previous one's result as its base address,
         so it physically cannot issue while that one is outstanding -- a chase can never put two
         capability loads in flight at once. Loading from EIGHT UNRELATED slots does, because no
         address depends on any result. The first version of this rung missed that and returned a
         clean 0 over 20000 hops while being structurally incapable of creating the condition. */
      {
        void *v0 = s07chase_ring[(b_seed + 0u) % S07CHASE_LINKS];
        void *v1 = s07chase_ring[(b_seed + 1u) % S07CHASE_LINKS];
        void *v2 = s07chase_ring[(b_seed + 2u) % S07CHASE_LINKS];
        void *v3 = s07chase_ring[(b_seed + 3u) % S07CHASE_LINKS];
        void *v4 = s07chase_ring[(b_seed + 4u) % S07CHASE_LINKS];
        void *v5 = s07chase_ring[(b_seed + 5u) % S07CHASE_LINKS];
        void *v6 = s07chase_ring[(b_seed + 6u) % S07CHASE_LINKS];
        void *v7 = s07chase_ring[(b_seed + 7u) % S07CHASE_LINKS];
        unsigned long t0,t1,t2,t3,t4,t5,t6,t7;
        S07CHASE_LCC_TYPE(t0, v0);  S07CHASE_LCC_TYPE(t1, v1);
        S07CHASE_LCC_TYPE(t2, v2);  S07CHASE_LCC_TYPE(t3, v3);
        S07CHASE_LCC_TYPE(t4, v4);  S07CHASE_LCC_TYPE(t5, v5);
        S07CHASE_LCC_TYPE(t6, v6);  S07CHASE_LCC_TYPE(t7, v7);
        if (t0==7UL||t1==7UL||t2==7UL||t3==7UL||t4==7UL||t5==7UL||t6==7UL||t7==7UL)
          if (bad < 0xFFFFu) bad++;
        b_seed += 3u;
        p = (void **)&s07chase_ring[0];
      }
#else
      for (b = 0; b < S07CHASE_BURST; b++)
        p = (void **)*(void *volatile *)p;
#endif

      S07CHASE_LCC_TYPE(ty_, p);
      if (ty_ == 7UL) {           /* 7 = NOT_CAP */
        if (bad < 0xFFFFu) bad++;
        /* Re-seat on a known-good link so one bad hop cannot cascade into a wild chase. */
        p = (void **)&s07chase_ring[0];
      }
      S07CHASE_BARRIER();
    }
  }

#ifdef S07CHASE_SELFTEST
  /* POSITIVE CONTROL, and it is not optional: a 0 from a query that cannot report 7 is exactly
     the failure mode this project keeps paying for. Query a plain integer and require the count
     to move. */
  {
    unsigned long junk_ = 0x5A5Aul, tj_;
    S07CHASE_LCC_TYPE(tj_, junk_);
    if (tj_ == 7UL) bad += 0x100u;
  }
#endif

  return bad;
}
#endif /* S07CHASE_KERNEL_H */
