/* row10 / sqlite-fb8ca7de0c -- fts5StructureAddLevel() edits an Fts5Structure in
 * place; when the structure is shared (nRef>1) with a scanning fts5vocab cursor,
 * the in-place write reallocs/corrupts memory the cursor still reads
 * (fts5_index.c:925, nRef at 339). Fixed 3.37.0. Ext: FTS5.
 * CONTROL: open an fts5vocab scan cursor on ft (holds a ref to the structure),
 * then INSERT rows into ft mid-scan to drive segment merges / add-level on the
 * shared structure. On unprotected capstone completes -> NOTRAP.
 * NOTE: build uses -DSQLITE_DQS=0, so SQL string literals MUST be single-quoted.
 * Heap stays at the 256K default: raising it inflates the in-image sqlite_heap[]
 * and the domain then exceeds the buddy allocator limit (create_dom failed). */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,
    "CREATE VIRTUAL TABLE ft USING fts5(a);"
    "INSERT INTO ft VALUES('alpha beta');"
    "INSERT INTO ft VALUES('beta gamma');"
    "INSERT INTO ft VALUES('gamma delta');"
    "INSERT INTO ft VALUES('delta epsilon');"
    "CREATE VIRTUAL TABLE vv USING fts5vocab('ft','row');",0,0,0);
  if(rc){out_text("fts5inplace setup rc=");out_uint((unsigned)rc);out_text(" (");out_text(sqlite3_errmsg(db));out_text(")\n");}
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT term,doc,cnt FROM vv",-1,&st,0)==SQLITE_OK){
    int n=0, nins=0;
    while(sqlite3_step(st)==SQLITE_ROW){
      n++;
      if(n==1){
        /* structure now shared with the live vocab cursor (nRef>1); write to drive
         * merges so fts5StructureAddLevel edits the shared structure in place */
        int i;
        for(i=0;i<24;i++){
          if(sqlite3_exec(db,"INSERT INTO ft VALUES('lorem ipsum dolor sit amet');",0,0,0)==SQLITE_OK) nins++;
        }
        out_text("fts5inplace mid-scan inserts ok=");out_uint((unsigned)nins);out_text("\n");
      }
    }
    out_text("fts5inplace rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("fts5inplace prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts5inplace NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5inplace")
