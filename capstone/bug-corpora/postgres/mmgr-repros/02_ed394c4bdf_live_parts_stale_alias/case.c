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

PG_CASE(2) {
/* Row 4 -- expand_partitioned_rtentry, fix ed394c4bdf. One Bitmapset, two
 * aliases; bms_del_member frees it through the field when the last member
 * goes, and the loop keeps reading the local. */
  unsigned char *set = MemoryContextAlloc(pg_root, 64);
  set[0] = 31;
  unsigned char *field = set; /* relinfo->live_parts */
  pg_held = set;              /* the local live_parts */
  pfree(field);               /* bms_del_member emptied and freed it */
  unsigned char *fresh = MemoryContextAlloc(pg_root, 64);
  fresh[0] = 37;
  pg_mark();
#ifdef PG_NEGATIVE_CONTROL
  (void)pg_probe(fresh); /* the LIVE replacement, same read, valid pointer */
#else
  (void)pg_probe(pg_held); /* the stale alias -- the defect */
#endif
}
