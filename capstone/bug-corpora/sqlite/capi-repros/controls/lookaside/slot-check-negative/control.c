/* la-slot-check-negative -- the positive control for la_alloc's "is a lookaside slot" check.
 *
 * Not one of the pre-registered controls. The three controls accept a block as a lookaside slot
 * on la_alloc's check (la_common.h). This program shows that check can come out FAILED: on
 * the same connection it requests
 *   small     128 bytes, lookaside enabled              -> a slot: the check must say ok;
 *   oversize  2000 bytes (> lookaside.sz = 1200)        -> served by memsys5: must say FAILED;
 *   disabled  128 bytes with the lookaside disabled     -> served by memsys5: must say FAILED.
 * The lookaside is disabled the way SQLite itself does it around a schema parse
 * (db->lookaside.bDisable++). The runner script (slot-check-negative.sh) reads the three CHECK
 * lines; the program's own failure count is therefore 2 when the check works.
 */
#include "sqlite3-capstone.c"
#include "repro322_common.h"
#include "la_common.h"

static int run_case(void) {
  if (repro_init()) return 1;
  sqlite3 *db = la_open();
  if (!db) return FAILRC("open", SQLITE_ERROR);
  unsigned char *small = la_alloc(db, "small", 128);
  unsigned char *big = la_alloc(db, "oversize", 2000);
  db->lookaside.bDisable++;
  unsigned char *dis = la_alloc(db, "disabled", 128);
  db->lookaside.bDisable--;
  if (dis) sqlite3DbFree(db, dis);
  if (big) sqlite3DbFree(db, big);
  if (small) sqlite3DbFree(db, small);
  la_close(db);
  printf("CONTROL la-slot-check-negative failures=%d (2 expected)\n", ctl_failures);
  return 0;
}

REPRO322_MAIN("la-slot-check-negative")
