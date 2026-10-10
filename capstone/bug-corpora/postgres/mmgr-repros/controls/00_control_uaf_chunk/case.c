#include "corpus.h"

/* CONTROL, not a defect. The plainest stale read a memory context allows: one
 * chunk, pfree'd, then read through the alias that outlived it, with no reuse
 * in between. It says what the arm IS before any case is scored: the replay
 * arena hands chunks out as offsets inside one capability, so the read must
 * COMPLETE there; the Sublet context pools issue each chunk as its own
 * capability and revoke it on pfree, so the read must FAULT, at the labelled
 * probe. tools/arms.json records both expectations, and a run whose control
 * does anything else scores no case (NO-READING, control-failed). */
PG_CASE(0) {
  unsigned char *chunk = MemoryContextAlloc(pg_root, 64);
  chunk[0] = 23;
  pg_held = chunk;
  pfree(chunk);
  pg_mark();
  (void)pg_probe(pg_held);
}
