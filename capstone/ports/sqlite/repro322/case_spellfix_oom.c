/* row24 / sqlite-d9305a65c876 -- spellfix editDist3Install() registers editdist3()
 * three times sharing ONE heap EditDist3Config, and only the arity-1 registration
 * carries the editDist3ConfigDelete destructor. The structure is:
 *
 *     pConfig = sqlite3_malloc64(sizeof(*pConfig));                       // (1)
 *     rc = create_function_v2("editdist3", 2, ..., pConfig, ..., 0);       // (2)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 3, ..., pConfig, ..., 0); }  // (3)
 *     if( rc==SQLITE_OK ){ rc = create_function_v2("...", 1, ..., pConfig, ...,
 *                                                  editDist3ConfigDelete); }          // (4)
 *     }else{ sqlite3_free(pConfig); }
 *
 * NOTE the else belongs to the THIRD if, so it fires when (2) or (3) failed:
 *   - (3) fails after (2) succeeded -> pConfig is FREED while editdist3/2 is still
 *     registered on it. Calling editdist3(a,b) then runs editDist3SqlFunc ->
 *     editDist3FindLang over the freed config. THIS is the use-after-free.
 *   - (2) fails -> nothing got registered, so no UAF.
 *   - (4) fails -> the else is NOT taken, so pConfig merely leaks (no destructor).
 * There is no refcount, hence the bug. Trunk-only fix. Ext: ext/misc/spellfix.c.
 *
 * SQLITE_UNTESTABLE removes sqlite3_test_control, so the only OOM lever is
 * exhausting the memsys5 arena. editDist3Install is the LAST step of
 * spellfix1Register, so everything before it can be allowed to succeed.
 *
 * Rather than guess the reserve size across many QEMU boots, this domain SWEEPS it
 * itself in one boot: for each k it drains the arena to ~k*64 free bytes, runs the
 * init, then releases the arena and asks whether editdist3/2 is still callable.
 * rc==SQLITE_NOMEM together with a callable editdist3/2 IS the bug state, and the
 * domain then calls it to perform the freed-config read.
 * NOTE: -DSQLITE_DQS=0 -> SQL string literals must be single-quoted. */
#include "repro322_common.h"

int sqlite3_spellfix_init(sqlite3*, char**, const void*);

/* Two-stage drain: fill the bulk with big blocks, then top up with 64-byte ones
 * (memsys5 minimum) for fine granularity. ~150 allocations per sweep step instead
 * of ~4000, which matters because the QEMU runner only waits a bounded time for
 * the shell prompt and a slow domain looks like a hang. */
#define BIG   4096
#define SML   64
#define MAXBIG 128
#define MAXSML 512

static void *g_big[MAXBIG];
static void *g_sml[MAXSML];

static int run_case(void){
  if (repro_init()) return 1;
  int k, hit = 0;

  for (k = 1; k <= 40 && !hit; k++){
    sqlite3 *db = 0;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK){ sqlite3_close(db); continue; }

    /* drain the arena: coarse then fine */
    int nb = 0, ns = 0, i; void *p;
    while (nb < MAXBIG && (p = sqlite3_malloc(BIG)) != 0) g_big[nb++] = p;
    while (ns < MAXSML && (p = sqlite3_malloc(SML)) != 0) g_sml[ns++] = p;
    /* give back the last k fine blocks (adjacent -> memsys5 coalesces them) */
    for (i = 0; i < k && ns > 0; i++) sqlite3_free(g_sml[--ns]);

    int rc = sqlite3_spellfix_init(db, 0, 0);

    /* release the arena BEFORE probing: prepare() needs memory, otherwise a
     * still-full arena would report "not callable" for the wrong reason. */
    while (ns > 0) sqlite3_free(g_sml[--ns]);
    while (nb > 0) sqlite3_free(g_big[--nb]);

    sqlite3_stmt *st = 0;
    int callable = (sqlite3_prepare_v2(db, "SELECT editdist3('kitten','sitting')", -1, &st, 0) == SQLITE_OK);

    out_text("k="); out_uint((unsigned)k);
    out_text(" rc="); out_uint((unsigned)(rc<0?-rc:rc));
    out_text(" callable="); out_uint((unsigned)callable);

    if (rc == SQLITE_NOMEM && callable){
      /* BUG STATE: (3) failed after (2) succeeded, so the else freed pConfig while
       * editdist3/2 stayed registered on it. prepare() does NOT invoke the function,
       * so reaching editDist3FindLang below can only happen via this branch. */
      out_text("  <== BUG STATE, calling editdist3 over freed config\n");
      int rows = 0; while (sqlite3_step(st) == SQLITE_ROW) rows++;
      out_text("spellfixoom editdist3 rows="); out_uint((unsigned)rows); out_text("\n");
      hit = 1;
    } else {
      out_text("\n");
    }
    if (st) sqlite3_finalize(st);
    sqlite3_close(db);
  }

  if (!hit) out_text("spellfixoom NO BUG STATE FOUND in sweep\n");
  out_text("spellfixoom NOTRAP done\n");
  return 0;
}
REPRO322_MAIN("spellfixoom")
