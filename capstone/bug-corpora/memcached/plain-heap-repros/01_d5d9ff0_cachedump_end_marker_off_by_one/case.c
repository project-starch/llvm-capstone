#include "corpus.h"

/* item_cachedump's buffer is ONE direct allocation -- `buffer = malloc(memlimit)`
 * at items.c:222 -- and its headroom guard reserves five bytes for "END\r\n"
 * while the writes that follow need six.
 *
 * Here it is the platform's malloc, which is what this corpus is for: the
 * defect crosses the malloc bound itself, with no slab or cache layer in
 * between, so it is the NOT-NESTED row. */

#define END_MARKER "END\r\n"

MCH_CASE(1) {
  /* Case 1 -- item_cachedump, fix d5d9ff0. PLAIN HEAP: a one-byte WRITE past a
   * direct malloc, because a reserved-headroom constant counts the bytes of a
   * marker and forgets its terminator.
   *
   * The loop at the pin (items.c:226-240) is:
   *
   *     buffer = malloc(memlimit);
   *     ...
   *         if (bufcurr + len + 5 > memlimit)   // 5 is END\r\n
   *             break;
   *         strcpy(buffer + bufcurr, temp);
   *         bufcurr += len;
   *     ...
   *     memcpy(buffer + bufcurr, END_MARKER, 5);   // then a NUL terminator
   *
   * "END\r\n" is five characters, so the guard's `+ 5` looks right. But the
   * marker is written with a terminating NUL after it -- the buffer is handed
   * back as a C string -- so six bytes are needed, not five. The fix changes
   * the guard to `+ 6` and says so in its own comment: `6 is END\r\n\0`.
   *
   * So an entry whose length brings `bufcurr` to exactly `memlimit - 5` passes
   * the guard, and the terminator lands at offset `memlimit`: one byte past the
   * allocation.
   *
   * THIS LEAVES THE ALLOCATION, which is what separates this corpus from the
   * sub-object and slab ones: a per-object bound is NOT in bounds for it. ASan
   * sees it, and CheriBSD sees it when the crossing leaves the allocator's
   * USABLE size -- a condition this corpus has already had refuted once, so the
   * request size is chosen to land on a size class and is recorded.
   *
   * The upstream fix is cited by hash only, deliberately: its subject line
   * names an outside contributor, and a person's name must not enter a
   * committed file in this tree. */
  const unsigned long memlimit = 64; /* a size class, so there is no slack */
  const unsigned long headroom = fixed ? 6 : 5; /* the one term the fix changes */

  char *buffer = malloc(memlimit);
  CHECK(buffer, 801);
  memset(buffer, 'x', memlimit);

  /* One entry sized so that bufcurr lands exactly at memlimit - 5: the guard
   * admits it at the pin and rejects it under the fix. */
  const unsigned long len = memlimit - 5;
  unsigned long bufcurr = 0;
  int admitted = (bufcurr + len + headroom) <= memlimit;

  o->cap = memlimit;
  o->touched = 0;
  o->crossed = 0;
  o->damage = 0;

  if (admitted) {
    bufcurr += len; /* the entry is written, as upstream's strcpy does */
    /* Now the END marker plus its terminator. The marker fits; the NUL is the
     * byte the guard did not reserve. */
    unsigned long nul_at = bufcurr + sizeof END_MARKER - 1; /* 5 chars, then NUL */
    o->touched = nul_at;
    o->crossed = nul_at >= memlimit;
    if (o->crossed) {
      /* Probe only the crossing byte. The labelled probe is what a fault or a
       * sanitiser report is required to land on. */
      write_probe((volatile unsigned char *)buffer + nul_at, '\0');
      o->damage = 1; /* the string the caller receives is unterminated in-bounds */
    } else {
      write_probe((volatile unsigned char *)buffer + nul_at, '\0');
    }
  }

  /* Asserted, not assumed: at the pin the entry IS admitted and the terminator
   * IS past the end; under the fix the entry is rejected and nothing is
   * written. A reduction that got the arithmetic wrong fails here rather than
   * reporting a verdict about nothing. */
  CHECK(fixed ? (!admitted) : (admitted && o->crossed), 802);

  o->defect_text = "the headroom guard reserved 5 bytes for \"END\\r\\n\" while the "
                   "marker is written with a terminating NUL, so the NUL landed one "
                   "byte past malloc(memlimit)";
  o->fixed_text = "the fix's `+ 6` headroom counts the terminator, so the entry is "
                  "rejected and nothing is written past the allocation";
  free(buffer);
}
