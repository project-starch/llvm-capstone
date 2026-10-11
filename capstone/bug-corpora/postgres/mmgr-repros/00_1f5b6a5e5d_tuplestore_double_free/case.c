#include "corpus.h"

/* NEGATIVE CONTROL, under -DPG_NEGATIVE_CONTROL.
 *
 * SCHEMA rule 5: every oracle needs one. This arm reports a fault, and a
 * fault is only this case's result if it depends on the defect. The control
 * keeps the allocation and free traffic BYTE FOR BYTE and changes only the
 * one access that is invalid, so what is tested is whether the pattern alone
 * makes the arm fault. It must complete.
 *
 * Replacing the access rather than deleting it is deliberate: an arm that
 * faults because the sequence ran at all would still be let off by a control
 * that simply does less work.
 */

PG_CASE(0) {
/* Row 1 -- tuplestore, bug #19438, fix 1f5b6a5e5d.
 * dumptuples() frees tuples as it writes them but clears memtupdeleted only
 * after the loop, so a WRITETUP that throws leaves memtuples[] holding freed
 * chunks; tuplestore_end then frees them a second time.
 *
 * The stale access is the second pfree itself, so there is no read probe: the
 * manager faults reading the revoked chunk's header before any bookkeeping
 * runs. This case's oracle therefore accepts a fault anywhere, unlike the
 * others. */
  unsigned char *memtuples[2];
  memtuples[0] = MemoryContextAlloc(pg_root, 64);
  memtuples[1] = MemoryContextAlloc(pg_root, 64);
  memtuples[0][0] = 11;
  pfree(memtuples[0]); /* WRITETUP freed it, then threw */
  pg_mark();
#ifdef PG_NEGATIVE_CONTROL
  pfree(memtuples[1]); /* a LIVE chunk: the same call, a valid argument */
#else
  pfree(memtuples[0]); /* the end walk, from index 0 -- the defect */
#endif
  pg_held = memtuples[1];
}
