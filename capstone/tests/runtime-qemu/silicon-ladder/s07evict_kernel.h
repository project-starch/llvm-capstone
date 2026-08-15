#ifndef S07EVICT_KERNEL_H
#define S07EVICT_KERNEL_H
/* S-07: does a spilled capability survive being EVICTED from L1 before it is reloaded?
 *
 * WHY THIS AND NOT ANOTHER SHAPE. On the current bitstream the board localized the fault to a
 * capability that was verified TAGGED by an LCC query, spilled by the compiler to a stack slot,
 * and reloaded NOT_CAP three instructions later (`sqlite3OsRead+0x144`). But the bare round trip
 * is SOUND: `s06spill` returns 0xFFFF across three independent redraws on this same silicon,
 * 48 round trips, every one tagged. So the sequence is necessary and NOT sufficient.
 *
 * What the failing site still has that `s06spill` lacks is working set. SQLite runs with a heap,
 * page cache and btree structures; `s06spill` spills sixteen slots in a tight loop that never
 * leave L1. This rung adds exactly that one variable and nothing else: the SAME sixteen spilled
 * capabilities, with a buffer walk between the store and the reload that is large enough to
 * evict them.
 *
 * TWO EARLIER RUNGS WERE VOID AND IT IS WORTH KNOWING WHY. `s07chase` chased a DEPENDENT pointer
 * chain, and a dependent load cannot issue while its predecessor is outstanding, so it could never
 * create the overlap it was built for. `s07indep` was meant to fix that with independent loads,
 * but at -O0 every result is spilled one instruction later, so it did not either. Both returned 0
 * with FIRING positive controls -- which proved the detector worked and said nothing about whether
 * the trigger existed. The lesson applied here: the eviction walk must be verified in the
 * DISASSEMBLY to sit between the store and the reload, not merely written that way in C.
 *
 * METHOD. Sixteen pointer locals, each spilled to its own stack slot exactly as in `s06spill`.
 * Then a full pass over S07EVICT_BYTES of scratch, touching one word per cache line, which both
 * evicts the stack slots and refills the cache with unrelated lines. Then each pointer is reloaded
 * and its type queried with LCC field 1, which is TOTAL, so a lost tag is REPORTED rather than
 * faulted on and the rung returns a number instead of wedging.
 *
 * retval = a 16-bit mask, bit i SET if slot i came back as a capability.
 *   0xFFFF (65535)  eviction does not cost the tag either -- this shape is also insufficient
 *   anything else   THE MINIMAL REPRODUCER, and the zero bits name which slots lost it
 *
 * A zero-bit result here would mean the ingredient is cache pressure, which is a far smaller and
 * more simulable statement than "run SQLite".
 */

/* Bigger than the L1 data cache so a full pass evicts everything, but small enough to fit the
   domain's data region. Walked at 64-byte stride = one touch per line. */
#ifndef S07EVICT_BYTES
#define S07EVICT_BYTES (48u * 1024u)
#endif
#ifndef S07EVICT_PASSES
#define S07EVICT_PASSES 2u
#endif

__attribute__((aligned(16))) static unsigned long s07evict_target[2];
__attribute__((aligned(64))) static volatile unsigned char s07evict_scratch[S07EVICT_BYTES];

#define S07EVICT_LCC_TYPE(out_, cap_) \
  __asm__ volatile(".insn r 0x5b, 0x1, 0x4, %0, %1, x1" : "=r"(out_) : "r"(cap_))

#define S07EVICT_BARRIER() __asm__ volatile("" ::: "memory")

#define S07EVICT_DECL(i) void *p##i = (void *)s07evict_target;
#define S07EVICT_CHECK(i)                          \
  do {                                             \
    unsigned long ty_;                             \
    S07EVICT_LCC_TYPE(ty_, p##i);                  \
    if (ty_ != 7UL) r |= (1u << (i));              \
  } while (0)

static unsigned s07evict_compute(void)
{
  unsigned r = 0;
  unsigned long sink = 0;
  unsigned pass, off;

  s07evict_target[0] = 0x5A5A5A5A5A5A5A5AUL;

  S07EVICT_DECL(0)  S07EVICT_DECL(1)  S07EVICT_DECL(2)  S07EVICT_DECL(3)
  S07EVICT_DECL(4)  S07EVICT_DECL(5)  S07EVICT_DECL(6)  S07EVICT_DECL(7)
  S07EVICT_DECL(8)  S07EVICT_DECL(9)  S07EVICT_DECL(10) S07EVICT_DECL(11)
  S07EVICT_DECL(12) S07EVICT_DECL(13) S07EVICT_DECL(14) S07EVICT_DECL(15)

  S07EVICT_BARRIER();

  /* THE ONE ADDED VARIABLE: evict the spilled slots before reloading them. `volatile` on the
     scratch keeps the walk from being optimised away -- without it this rung silently becomes
     s06spill and reports a clean 0xFFFF having tested nothing new. */
  for (pass = 0; pass < S07EVICT_PASSES; pass++)
    for (off = 0; off < S07EVICT_BYTES; off += 64u)
      sink += s07evict_scratch[off];

  S07EVICT_BARRIER();

  S07EVICT_CHECK(0);  S07EVICT_CHECK(1);  S07EVICT_CHECK(2);  S07EVICT_CHECK(3);
  S07EVICT_CHECK(4);  S07EVICT_CHECK(5);  S07EVICT_CHECK(6);  S07EVICT_CHECK(7);
  S07EVICT_CHECK(8);  S07EVICT_CHECK(9);  S07EVICT_CHECK(10); S07EVICT_CHECK(11);
  S07EVICT_CHECK(12); S07EVICT_CHECK(13); S07EVICT_CHECK(14); S07EVICT_CHECK(15);

#ifdef S07EVICT_SELFTEST
  /* POSITIVE CONTROL. 0xFFFF from a query that cannot return anything else is not a measurement.
     Note this only validates the DETECTOR; the eviction walk itself must be checked in the
     disassembly, which is the mistake that made two earlier rungs void. */
  {
    unsigned long junk_ = 0x5A5Aul, tj_;
    S07EVICT_LCC_TYPE(tj_, junk_);
    if (tj_ == 7UL) r = 0u;
  }
#endif

  if (sink == 0x1234UL) r ^= 0x8000u;   /* consume the walk so it cannot be elided */
  return r;
}
#endif /* S07EVICT_KERNEL_H */
