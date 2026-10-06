/* new-3 (rtree arm) / sqlite-eab0e10304 -- rtree leaves a freed heap buffer bound
 * SQLITE_STATIC to a persistent cached statement.
 *
 * nodeNew() allocates the node and its payload as ONE block:
 *     pNode = sqlite3_malloc(sizeof(RtreeNode) + pRtree->iNodeSize);
 *     pNode->zData = (u8 *)&pNode[1];
 * and nodeWrite() binds that payload into the cached pWriteNode statement without
 * ever binding it back to NULL:
 *     sqlite3_bind_blob(p, 2, pNode->zData, pRtree->iNodeSize, SQLITE_STATIC);
 *     sqlite3_step(p);
 *     rc = sqlite3_reset(p);
 * sqlite3_reset() does not clear bindings, so when nodeRelease() does
 * sqlite3_free(pNode) the statement's parameter 2 dangles into the freed block.
 * Fixed 3.23.0 (16 days after 3.22.0 shipped) by adding sqlite3_bind_null(p, 2).
 *
 * This is the SECOND arm of the same defect; case_fts3_static_bind.c is the fts3 arm.
 * Both use the route the fix's own commit message names -- "obtain a pointer to the
 * persistent statement using sqlite3_next_stmt() and attempt to access the freed
 * buffer using sqlite3_expanded_sql() or similar" -- with the "or similar" variant,
 * re-STEPping the statement, because sqlite3_expanded_sql() is compiled out in this
 * port: -DSQLITE_OMIT_FLOATING_POINT=1 makes 3.22.0's sqliteInt.h also
 * `#define SQLITE_OMIT_TRACE 1`, so the function body is `return 0;`. See
 * case_fts3_static_bind.c for the full write-up of that.
 *
 * Host ASan oracle on 3.22.0 (page_size=1024, 120 rows), identical under
 * -DSQLITE_RTREE_INT_ONLY which this group uses:
 *   re-step  : READ of size 820, 40 bytes into a freed 864-byte region
 *                use   sqlite3VdbeSerialPut <- sqlite3VdbeExec <- sqlite3_step
 *   expanded : READ of size 1   (upstream's route, for comparison)
 *     free  nodeRelease <- rtreeUpdate
 *     alloc nodeAcquire
 *   Offset 40 is exactly sizeof(RtreeNode) on the host, i.e. &pNode[1] == zData,
 *   which confirms the single-block layout is what is being read.
 *
 * WHY THIS ARM IS WORTH HAVING ALONGSIDE THE fts3 ONE: its consequence is far
 * sharper. The re-step writes the freed block's current contents over the node's
 * row in %_node, and a later full scan then simply LOSES the rows that lived in
 * that node -- quantitative, observable data loss with no memory-safety tool:
 *     full_scan_before=120 -> full_scan_after=71      (49 rows gone)
 *     node data: same length 820, sum changed, first4 0x00000031 -> 0x00000000
 * Note PRAGMA integrity_check returns ok here: it does not validate rtree shadow
 * tables, so the row count is the signal, not integrity_check.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
/* CheriBSD DEVIATION from the Capstone trigger, and why.
 *
 * The bug is the dangling SQLITE_STATIC binding: rtree binds pNode->zData to the cached
 * write-node statement and then frees pNode, so re-stepping copies whatever now lives in
 * the freed block over the node row. Whether that is VISIBLE depends on the allocator
 * having recycled the block in between. On Capstone it had (32 of 120 rows went missing);
 * on CheriBSD the first run came back len_changed=0 sum_changed=0 rows_lost=0 -- the freed
 * block still held the original node image, so the copy was a no-op. Under purecap the
 * surrounding allocations are larger, so memsys5 hands out different blocks and the node
 * block simply was not reused by the next_stmt/sqlite3_sql work that follows.
 *
 * So before the re-step this version explicitly asks memsys5 for blocks of the node size
 * and fills them with a marker. If one of them IS the freed node block, the dangling
 * parameter now points at the marker and the consequence becomes deterministic instead of
 * incidental. The trigger, the binding and the re-step are untouched; the probe below
 * reports whether the freed block was actually recovered, so a miss stays visible.
 */
#include "repro322_common.h"

static void rcline(const char *what, int rc){
  out_text("rtreestatic "); out_text(what); out_text(" rc=");
  out_uint((unsigned)(rc<0?-rc:rc)); out_text("\n");
}

static int count1(sqlite3 *db, const char *sql){
  sqlite3_stmt *st=0; int v=-1;
  if(sqlite3_prepare_v2(db,sql,-1,&st,0)==SQLITE_OK && sqlite3_step(st)==SQLITE_ROW)
    v = sqlite3_column_int(st,0);
  sqlite3_finalize(st); return v;
}

/* FNV-1a over the highest-numbered %_node payload, plus length and first 4 bytes. */
static int nodehash(sqlite3 *db, unsigned *pLen, unsigned *pSum, unsigned *pFirst){
  sqlite3_stmt *st=0; int ok=0;
  *pLen=0; *pSum=0; *pFirst=0;
  if(sqlite3_prepare_v2(db,
       "SELECT data FROM rt_node ORDER BY nodeno DESC LIMIT 1",-1,&st,0)==SQLITE_OK
     && sqlite3_step(st)==SQLITE_ROW){
    const unsigned char *p = sqlite3_column_blob(st,0);
    int n = sqlite3_column_bytes(st,0), i;
    unsigned sum = 2166136261u;
    if(p){
      for(i=0;i<n;i++) sum = (sum ^ p[i]) * 16777619u;
      *pLen=(unsigned)n; *pSum=sum;
      if(n>=4) *pFirst = ((unsigned)p[0]<<24)|((unsigned)p[1]<<16)|((unsigned)p[2]<<8)|p[3];
      ok=1;
    }
  }
  sqlite3_finalize(st); return ok;
}

static int contains(const char *hay, const char *nee){
  if(!hay || !nee) return 0;
  for(; *hay; hay++){
    const char *h=hay, *n=nee;
    while(*n && *h==*n){ h++; n++; }
    if(!*n) return 1;
  }
  return 0;
}

static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; char *e=0; int rc, i;

  rc = sqlite3_open(":memory:", &db);
  rcline("open", rc);
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  rc = sqlite3_exec(db,"PRAGMA page_size=1024;",0,0,&e);
  rcline("pagesize", rc); if(e){ sqlite3_free(e); e=0; }

  rc = sqlite3_exec(db,"CREATE VIRTUAL TABLE rt USING rtree(id,x1,x2);",0,0,&e);
  rcline("create", rc);
  if(e){ out_text("rtreestatic create err ("); out_text(e); out_text(")\n"); sqlite3_free(e); e=0; }
  if(rc!=SQLITE_OK){ sqlite3_close(db); return 1; }

  /* enough rows for several nodes, so that losing one node is visible */
  sqlite3_stmt *ins=0;
  if(sqlite3_prepare_v2(db,"INSERT INTO rt VALUES(?,?,?)",-1,&ins,0)!=SQLITE_OK){
    out_text("rtreestatic insert prepare err ("); out_text(sqlite3_errmsg(db)); out_text(")\n");
    sqlite3_close(db); return 1;
  }
  int nins=0;
  for(i=1;i<=120;i++){
    sqlite3_bind_int(ins,1,i); sqlite3_bind_int(ins,2,i); sqlite3_bind_int(ins,3,i+1);
    if(sqlite3_step(ins)==SQLITE_DONE) nins++;
    sqlite3_reset(ins);
  }
  sqlite3_finalize(ins);
  out_text("rtreestatic inserted="); out_uint((unsigned)nins); out_text("\n");

  int nodes = count1(db,"SELECT count(*) FROM rt_node");
  int scan_before = count1(db,"SELECT count(*) FROM rt");
  out_text("rtreestatic node_rows="); out_uint((unsigned)(nodes<0?0:nodes));
  out_text(" scan_before="); out_uint((unsigned)(scan_before<0?0:scan_before)); out_text("\n");

  unsigned l1,s1,f1,l2,s2,f2;
  int have1 = nodehash(db,&l1,&s1,&f1);
  out_text("rtreestatic before present="); out_uint((unsigned)have1);
  out_text(" len="); out_uint(l1); out_text(" sum="); out_uint(s1);
  out_text(" first4="); out_uint(f1); out_text("\n");
  if(!have1 || nodes < 2){
    out_text("rtreestatic WARNING fewer than 2 nodes; no node was written then freed\n");
    sqlite3_close(db); out_text("rtreestatic NOTRAP done\n"); return 0;
  }

  /* reach the persistent cached write-node statement through the public API */
  sqlite3_stmt *target=0; int nstmt=0, nsql=0;
  { sqlite3_stmt *s;
    for(s=sqlite3_next_stmt(db,0); s; s=sqlite3_next_stmt(db,s)){
      const char *z = sqlite3_sql(s); nstmt++;
      if(z){ nsql++; if(contains(z,"_node") && contains(z,"VALUES")) target=s; }
    }
  }
  out_text("rtreestatic cached_stmts="); out_uint((unsigned)nstmt);
  out_text(" with_sql="); out_uint((unsigned)nsql);
  out_text(" target="); out_uint((unsigned)(target!=0)); out_text("\n");
  if(!target){
    out_text("rtreestatic WARNING write-node statement not reachable via next_stmt\n");
    sqlite3_close(db); out_text("rtreestatic NOTRAP done\n"); return 0;
  }
  out_text("rtreestatic target sql=["); out_text(sqlite3_sql(target)); out_text("]\n");

  /* CheriBSD addition (see header): make the freed block's contents observable. */
  unsigned marker_f4 = 0x5A5A5A5Au;
  {
    enum { NPROBE = 16 };
    static void *probe[NPROBE];
    int np = 0, recovered = -1, i;
    int nodesz = 1024 - 64;            /* pRtree->iNodeSize for page_size=1024 */
    for(i = 0; i < NPROBE; i++){
      void *q = sqlite3_malloc(nodesz);
      if(!q) break;
      probe[np++] = q;
      const unsigned char *b = (const unsigned char *)q;
      unsigned fq = ((unsigned)b[0]<<24)|((unsigned)b[1]<<16)|((unsigned)b[2]<<8)|b[3];
      if(fq == f1 && recovered < 0) recovered = i;
    }
    out_text("rtreestatic probe_allocs="); out_uint((unsigned)np);
    out_text(" recovered_freed_node=");
    if(recovered < 0) out_text("no\n");
    else { out_text("yes at index "); out_uint((unsigned)recovered); out_text("\n"); }
    /* keep them allocated so the marker stays put while the re-step reads through
     * the dangling SQLITE_STATIC pointer */
    for(i = 0; i < np; i++) memset(probe[i], 0x5A, (size_t)nodesz);
    out_text("rtreestatic probe blocks filled with 0x5A\n");
  }

  /* THE USE: parameter 2 still points into the freed RtreeNode block */
  out_text("rtreestatic before re-step\n");
  int srq = sqlite3_step(target);
  out_text("rtreestatic restep rc="); out_uint((unsigned)(srq<0?-srq:srq)); out_text("\n");
  sqlite3_reset(target);

  int have2 = nodehash(db,&l2,&s2,&f2);
  int scan_after = count1(db,"SELECT count(*) FROM rt");
  int nodes_after = count1(db,"SELECT count(*) FROM rt_node");
  out_text("rtreestatic after present="); out_uint((unsigned)have2);
  out_text(" len="); out_uint(l2); out_text(" sum="); out_uint(s2);
  out_text(" first4="); out_uint(f2); out_text("\n");
  out_text("rtreestatic node_rows_after="); out_uint((unsigned)(nodes_after<0?0:nodes_after));
  /* Report -1 as ERROR rather than letting the <0?0: guard print it as a plain 0:
   * a failed scan and a scan that returns zero rows are different outcomes, and on
   * CheriBSD this is the failed one. */
  out_text(" scan_after=");
  if(scan_after < 0) out_text("ERROR(query failed)"); else out_uint((unsigned)scan_after);
  out_text("\n");

  /* DECISIVE: the freed block's current contents were copied over the node's row,
   * so a full scan loses whatever lived in that node. */
  out_text("rtreestatic VERDICT len_changed="); out_uint((unsigned)(l1!=l2));
  out_text(" sum_changed="); out_uint((unsigned)(s1!=s2));
  out_text(" first4_changed="); out_uint((unsigned)(f1!=f2));
  out_text(" marker_in_row="); out_uint((unsigned)(f2==marker_f4));
  out_text(" rows_lost=");
  if(scan_after < 0){
    out_text("ALL(scan failed), was "); out_uint((unsigned)scan_before);
  } else {
    out_uint((unsigned)((scan_before>scan_after) ? scan_before-scan_after : 0));
  }
  out_text("\n");
  if(s1==s2) out_text("rtreestatic note node data unchanged; the read was silent and the"
                      " freed block had not been reused yet\n");

  /* integrity_check does NOT validate rtree shadow tables -- recorded for contrast */
  rc = sqlite3_exec(db,"PRAGMA integrity_check;",0,0,&e);
  rcline("integrity", rc);
  if(e){ sqlite3_free(e); e=0; }

  sqlite3_close(db);
  out_text("rtreestatic NOTRAP done\n"); return 0;
}
REPRO322_MAIN("rtreestatic")
