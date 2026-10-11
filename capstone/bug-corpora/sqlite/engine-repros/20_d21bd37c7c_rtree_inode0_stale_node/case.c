/* new-16 / sqlite-d21bd37c7c -- an rtree node inserted into the node hash with
 * iNode==0 is freed but can never be unlinked.
 *
 * nodeWrite() in 3.22.0:
 *     sqlite3_step(p); rc = sqlite3_reset(p);
 *     if( pNode->iNode==0 && rc==SQLITE_OK ){
 *       pNode->iNode = sqlite3_last_insert_rowid(pRtree->db);
 *       nodeHashInsert(pRtree, pNode);          <-- no check for iNode==0
 *     }
 * If the %_node INSERT is suppressed (a BEFORE INSERT trigger raising IGNORE) and
 * no other rowid insert has happened on the connection, last_insert_rowid() is 0.
 * The node is then linked into aHash[0] still carrying iNode==0.  nodeHashDelete()
 * keys off iNode and treats 0 as "not in the hash", so nodeRelease() frees the block
 * and the hash keeps a dangling entry.  The next nodeAcquire() walks that bucket and
 * reads p->iNode out of freed memory.  Fixed on trunk 2026-09-24 (expected 3.54.0)
 * by returning SQLITE_CORRUPT_VTAB when last_insert_rowid() is 0.
 *
 * Host ASan oracle on 3.22.0, identical shape:
 *   READ of size 8, 8 bytes into a 1000-byte region   (RtreeNode.iNode)
 *     nodeHashLookup <- nodeAcquire <- rtreeFilter            (the use)
 *     nodeRelease <- SplitNode <- rtreeInsertCell             (the free)
 *     nodeNew <- SplitNode <- rtreeInsertCell                 (the alloc)
 *
 * Two requirements and how they are met in a freestanding :memory: domain:
 *  1) A connection whose node hash is EMPTY when the suppressed write happens.
 *     Upstream does `db close; sqlite3 db test.db`, which this domain cannot do --
 *     the VFS xOpen always returns SQLITE_CANTOPEN.  Instead this uses the row-12
 *     trick: shared cache plus a SECOND connection on file::memory:?cache=shared.
 *     Each connection builds its own Rtree object, so db2 starts with an empty
 *     aHash[].  (A :memory: database, shared-cache URI included, takes SQLite's
 *     memDb path and never calls xOpen.)  Needs -USQLITE_OMIT_SHARED_CACHE.
 *  2) A node filled exactly to capacity, so the next INSERT splits it.
 *     iNodeSize is page_size-64 and a 5-column cell is 24 bytes, so page_size=1024
 *     gives (1024-64-4)/24 = 39 cells.  Hence exactly 39 rows.  (At the default
 *     4096 the per-node cap RTREE_MAXCELLS=51 applies instead -- that is why
 *     upstream's test uses 51.)  An off-by-one row count does NOT trigger.
 *
 * `zero` is an empty INTEGER PRIMARY KEY table: it exists only so that nothing on
 * the connection has ever set a rowid, keeping last_insert_rowid() at 0.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
#include "repro322_common.h"

#define URI "file::memory:?cache=shared"

static void rcline(const char *what, int rc){
  out_text("rtreeinode0 "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int count1(sqlite3 *db, const char *sql, const char *label){
  sqlite3_stmt *st=0; int v=-1;
  if(sqlite3_prepare_v2(db,sql,-1,&st,0)==SQLITE_OK && sqlite3_step(st)==SQLITE_ROW){
    v = sqlite3_column_int(st,0);
  }
  sqlite3_finalize(st);
  out_text("rtreeinode0 "); out_text(label); out_text("=");
  out_uint((unsigned)(v<0?0:v)); out_text("\n");
  return v;
}

static int openshared(sqlite3 **pdb, const char *tag){
  int rc = sqlite3_open_v2(URI, pdb,
      SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_URI, 0);
  out_text("rtreeinode0 open "); out_text(tag); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc));
  if(rc!=SQLITE_OK && *pdb){ out_text(" ("); out_text(sqlite3_errmsg(*pdb)); out_text(")"); }
  out_text("\n");
  return rc;
}

static int run_case(void){
  if (repro_init()) return 1;
  char *e=0; int rc;

  rc = sqlite3_enable_shared_cache(1);
  rcline("shared_cache", rc);

  sqlite3 *db1=0, *db2=0;
  if(openshared(&db1,"db1")){ sqlite3_close(db1); return 1; }

  rc = sqlite3_exec(db1,
      "PRAGMA page_size=1024;"
      "CREATE TABLE zero(x INTEGER PRIMARY KEY);"
      "CREATE VIRTUAL TABLE t1 USING rtree(id,x1,x2,y1,y2);", 0,0,&e);
  rcline("setup", rc);
  if(e){ out_text("rtreeinode0 setup err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db1); return 1; }

  /* Exactly 39 rows: fills one node to capacity at page_size=1024. */
  sqlite3_stmt *ins=0;
  if(sqlite3_prepare_v2(db1,"INSERT INTO t1 VALUES(?,?,?,?,?)",-1,&ins,0)!=SQLITE_OK){
    out_text("rtreeinode0 insert prepare err ("); out_text(sqlite3_errmsg(db1)); out_text(")\n");
    sqlite3_close(db1); return 1;
  }
  int i, nins=0;
  for(i=1;i<=39;i++){
    sqlite3_bind_int(ins,1,i); sqlite3_bind_int(ins,2,i); sqlite3_bind_int(ins,3,i+1);
    sqlite3_bind_int(ins,4,i); sqlite3_bind_int(ins,5,i+1);
    if(sqlite3_step(ins)==SQLITE_DONE) nins++;
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  out_text("rtreeinode0 inserted="); out_uint((unsigned)nins); out_text("\n");
  count1(db1,"SELECT count(*) FROM t1","rows");
  count1(db1,"SELECT count(*) FROM t1_node","nodes before");

  /* Suppress every further %_node INSERT. */
  rc = sqlite3_exec(db1,
      "CREATE TRIGGER tr BEFORE INSERT ON t1_node BEGIN SELECT raise(IGNORE); END;",0,0,&e);
  rcline("trigger", rc);
  if(e){ out_text("rtreeinode0 trigger err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db1); return 1; }

  /* db2 is a fresh connection: its own Rtree with an empty aHash[]. */
  if(openshared(&db2,"db2")){ sqlite3_close(db1); sqlite3_close(db2); return 1; }
  int nb = count1(db2,"SELECT count(*) FROM t1_node","db2 nodes");

  out_text("rtreeinode0 before split-insert\n");
  rc = sqlite3_exec(db2,"INSERT INTO t1 VALUES(1000,0,1,0,1);",0,0,&e);
  /* rc=0 here == nodeWrite accepted iNode==0 == vulnerable build.
   * After the trunk fix this is SQLITE_CORRUPT (11). */
  rcline("split insert", rc);
  if(e){ out_text("rtreeinode0 ins err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  int insrc = rc;

  /* DECISIVE: the node count must be UNCHANGED -- the trigger really did swallow
   * the %_node write, so pNode->iNode stayed 0 and went into aHash[0] anyway. */
  int na = count1(db2,"SELECT count(*) FROM t1_node","db2 nodes after");

  out_text("rtreeinode0 before dangling-hash query\n");
  int got = count1(db2,"SELECT count(*) FROM t1 WHERE id=25","probe rows");

  if(na!=nb) out_text("rtreeinode0 WARNING %_node grew; trigger did not suppress the write\n");
  if(insrc!=SQLITE_OK) out_text("rtreeinode0 WARNING split insert was refused\n");
  (void)got;

  sqlite3_close(db2); sqlite3_close(db1);
  out_text("rtreeinode0 NOTRAP done\n"); return 0;
}
REPRO322_MAIN("rtreeinode0")
