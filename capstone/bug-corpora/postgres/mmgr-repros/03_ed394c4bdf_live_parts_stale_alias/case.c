#include "corpus.h"

PG_CASE(3) {
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
  (void)pg_probe(pg_held);
}
