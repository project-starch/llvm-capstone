/* A1's probes in the speedtest1 domain: the grain of the protection, one read each.
 *
 * -DSPEEDTEST1_PROBE=n runs one of these instead of the benchmark, as a matched pair with the
 * unprotected build (run.sh probe <n> runs both, the unprotected one first, as the control).
 *
 *   1  a column name read after sqlite3_finalize freed it, a lookaside slot
 *   2  a memsys5 block read after sqlite3_free
 *   3  a memsys5 block read after its buddy was freed
 *   4  a lookaside slot read after the slot beside it was finalized
 *   5  the last byte of a 128-byte block read, a 100-byte request
 *   6  the byte after that block read
 *
 * Under the port 1, 2 and 6 halt the domain at the read and the log carries the fault line; 3,
 * 4 and 5 read on, which is the point: a revocation reaches one object and not the block around
 * it, and a block's alias carries the block's bounds. The unprotected build reads through all
 * six and prints the marker. The marker strings are the ones the recorded passes carry.
 *
 * Linked in beside speedtest1_domain.c (the port runner's SPEEDTEST1_PROBE_SRC); the domain's
 * payload writers are all it needs from there.
 */
#include "sqlite3.h"
#include "sqlite_hostcall.h"

void speedtest1_output_text(const char *text);
void speedtest1_output_uint(unsigned long value);

void speedtest1_probe(unsigned *res) {
  volatile char c = 0;
  char one[2] = {0, 0};
  if (sqlite3_initialize() != SQLITE_OK) {
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ initialize failed\n");
    *res = SQLITE_HC_ERR_INITIALIZE;
    return;
  }
  if (SPEEDTEST1_PROBE == 1) {
    sqlite3 *db = 0;
    sqlite3_stmt *st = 0;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK ||
        sqlite3_prepare_v2(db, "SELECT 1 AS colname", -1, &st, 0) != SQLITE_OK ||
        sqlite3_step(st) != SQLITE_ROW) {
      speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ setup failed\n");
      *res = 0x5117E105u;
      return;
    }
    const char *name = sqlite3_column_name(st, 0);
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ lookaside: live name=");
    speedtest1_output_text(name);
    speedtest1_output_text("\n");
    sqlite3_finalize(st); /* frees the name: a lookaside slot, revoked under the port */
    c = name[0];
    one[0] = c;
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF_NOTRAP__ lookaside name[0]=");
    speedtest1_output_text(one);
    speedtest1_output_text("\n");
    sqlite3_close(db);
  } else if (SPEEDTEST1_PROBE == 2) {
    char *p = sqlite3_malloc(4096);
    if (!p) {
      speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ malloc failed\n");
      *res = 0x5117E105u;
      return;
    }
    p[0] = 'm';
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ memsys5: live p[0]=m\n");
    sqlite3_free(p); /* the block's handle is revoked under the port */
    c = ((volatile char *)p)[0];
    one[0] = c;
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF_NOTRAP__ memsys5 p[0]=");
    speedtest1_output_text(one);
    speedtest1_output_text("\n");
  } else if (SPEEDTEST1_PROBE == 3) {
    /* The grain of a free: two blocks, buddies in memsys5, one freed, the other read.
       Returns with the marker when the neighbour survives its sibling's revocation. */
    char *a = sqlite3_malloc(4096);
    char *b = sqlite3_malloc(4096);
    if (!a || !b) {
      speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ malloc failed\n");
      *res = 0x5117E105u;
      return;
    }
    a[0] = 'a';
    b[0] = 'b';
    sqlite3_free(a);
    c = ((volatile char *)b)[0];
    one[0] = c;
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_SIBLING__ memsys5 b[0]=");
    speedtest1_output_text(one);
    speedtest1_output_text(" after a was freed\n");
    sqlite3_free(b);
  } else if (SPEEDTEST1_PROBE == 4) {
    /* The grain of a free in the pool: two statements' column names, two slots; one
       statement finalized, the other's name read. */
    sqlite3 *db = 0;
    sqlite3_stmt *s1 = 0, *s2 = 0;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK ||
        sqlite3_prepare_v2(db, "SELECT 1 AS one", -1, &s1, 0) != SQLITE_OK ||
        sqlite3_prepare_v2(db, "SELECT 2 AS two", -1, &s2, 0) != SQLITE_OK ||
        sqlite3_step(s1) != SQLITE_ROW || sqlite3_step(s2) != SQLITE_ROW) {
      speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ setup failed\n");
      *res = 0x5117E105u;
      return;
    }
    const char *n2 = sqlite3_column_name(s2, 0);
    sqlite3_finalize(s1); /* frees s1's name, a slot beside s2's */
    c = n2[0];
    one[0] = c;
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_SIBLING__ lookaside n2[0]=");
    speedtest1_output_text(one);
    speedtest1_output_text(" after s1 was finalized\n");
    sqlite3_finalize(s2);
    sqlite3_close(db);
  } else {
    /* The grain of the bounds: a 100-byte request is a 128-byte block. 5 reads the block's
       last byte and returns, 6 reads the byte after the block and must trap. */
    char *p = sqlite3_malloc(100);
    if (!p) {
      speedtest1_output_text("__CAPSTONE_SPEEDTEST1_UAF__ malloc failed\n");
      *res = 0x5117E105u;
      return;
    }
    speedtest1_output_text("__CAPSTONE_SPEEDTEST1_BOUNDS__ size=");
    speedtest1_output_uint((unsigned long)sqlite3_msize(p));
    speedtest1_output_text("\n");
    c = ((volatile char *)p)[SPEEDTEST1_PROBE == 5 ? 127 : 128];
    (void)c;
    speedtest1_output_text(SPEEDTEST1_PROBE == 5 ? "__CAPSTONE_SPEEDTEST1_BOUNDS__ p[127] read\n"
                                    : "__CAPSTONE_SPEEDTEST1_BOUNDS_NOTRAP__ p[128] read\n");
    sqlite3_free(p);
  }
  *res = SQLITE_HC_RET_DONE;
}
