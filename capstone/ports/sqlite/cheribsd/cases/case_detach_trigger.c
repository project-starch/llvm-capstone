/* row4 / sqlite-6397a78b2b -- after DETACH the auxiliary schema is freed, but a
 * TEMP TRIGGER keeps pTabSchema pointing at it (attach.c detachFunc frees the aux
 * btree+schema; the temp trigger in aDb[1] survives). A later DROP TRIGGER calls
 * sqlite3DropTriggerPtr -> tableOfTrigger(pTrigger) ->
 *   sqlite3HashFind(&pTrigger->pTabSchema->tblHash, pTrigger->table)
 * which READS the freed aux schema (trigger.c). Fixed 3.30.0. Core, needs TEMPDB.
 * CONTROL: reproduce; on unprotected capstone the freed-schema read completes or
 * faults incidentally. Build in the coreT group (-USQLITE_OMIT_TEMPDB). */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  char *e=0;
  rc=sqlite3_exec(db,"ATTACH ':memory:' AS aux;",0,0,&e);
  if(rc){out_text("detachtrig attach rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;}
  rc=sqlite3_exec(db,
    "CREATE TABLE aux.t(x);"
    "CREATE TEMP TRIGGER tr AFTER INSERT ON aux.t BEGIN SELECT 1; END;",0,0,&e);
  if(rc){out_text("detachtrig setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;}
  out_text("detachtrig before detach\n");
  rc=sqlite3_exec(db,"DETACH aux;",0,0,&e);
  out_text("detachtrig detach rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;
  /* tr.pTabSchema now dangles at the freed aux schema; DROP TRIGGER derefs it via tableOfTrigger */
  out_text("detachtrig before drop\n");
  rc=sqlite3_exec(db,"DROP TRIGGER tr;",0,0,&e);
  out_text("detachtrig drop rc=");out_uint((unsigned)rc);out_text(" (");out_text(e?e:"");out_text(")\n");sqlite3_free(e);e=0;
  sqlite3_close(db);
  out_text("detachtrig NOTRAP done\n"); return 0;
}
REPRO322_MAIN("detachtrig")
