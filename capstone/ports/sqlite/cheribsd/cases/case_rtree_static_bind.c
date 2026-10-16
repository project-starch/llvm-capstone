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
 * row in %_node, and the table then loses the rows that lived in that node --
 * observable data loss with no memory-safety tool involved. Measured 2026-10-16
 * on CheriBSD purecap, with the reclaim step below in place:
 *     scan_before=120 -> scan_after=ERROR(query failed)   (all 120 unreadable)
 *     node data: same length 820, sum changed, first4 0x00000031 -> 0x5A5A5A5A
 *     PRAGMA integrity_check: ok
 * An earlier header recorded "120 -> 71, 49 rows gone"; that was a run in which
 * the freed block happened to hold a partially valid node image, so the scan
 * still parsed most of the tree. The failure mode is not quantitative -- what the
 * scan does depends on the bytes that happen to be in the block -- so the signal
 * to report is that the row changed and the scan no longer returns the table,
 * never a particular row count.
 * Note PRAGMA integrity_check returns ok either way: it does not validate rtree
 * shadow tables, so it is not the signal.
 * NOTE: -DSQLITE_DQS=0, so SQL string literals must be single-quoted. */
/* WHY THE TRIGGER RECLAIMS THE FREED BLOCK BEFORE THE RE-STEP.
 *
 * The defect is the dangling SQLITE_STATIC binding, and the re-step always reads through
 * it. Whether that read has a VISIBLE consequence depends on the allocator having handed
 * the block to someone else in between, and left to chance it usually has not: measured
 * 2026-10-16, both virtual Capstone arms came back len_changed=0 sum_changed=0 rows_lost=0
 * with the case's own note "the freed block had not been reused yet", and CheriBSD purecap
 * did the same before this step was added. The stale read still happened -- the Sublet arm
 * faults on it with cause 25, and probe site 56 witnesses it -- but the row it wrote was
 * byte-identical to the row that was already there, so the data loss this case is about
 * could not be demonstrated.
 *
 * So before the re-step the trigger asks memsys5 for blocks of the node's payload size and
 * fills them with 0x5A. If one of them is the freed node block -- and with memsys5's
 * power-of-two size classes it is -- the dangling parameter now points at the marker, and
 * the consequence becomes deterministic rather than incidental. The binding, the free and
 * the re-step are untouched; this only decides what the freed block CONTAINS when the
 * re-step reads it.
 *
 * Whether the block really was reclaimed is reported by marker_in_row in the VERDICT line,
 * which is the check that can fire: 0x5A5A5A5A in the %_node row can only have come from
 * these blocks. An earlier version of this step tried to recognise the block at allocation
 * time by comparing its first four bytes against the node's, which CANNOT work -- memsys5
 * writes Mem5Link{next,prev} over the first eight bytes of every freed block, so those
 * bytes are a freelist index, never the node image. It reported "no" on runs where the
 * marker did reach the row.
 *
 * This file is byte-for-byte the corpus case, 21_eab0e10304_rtree_static_bind_stale_text
 * under bug-corpora/sqlite/engine-repros.
 * Neither reason ports/sqlite/cheribsd/port.json gives for the two case sets differing
 * applies here: this case opens :memory: and needs no database file, and the block above is
 * part of the trigger, not probe instrumentation.
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

  /* See the header: make the freed block's contents observable, so the consequence is
   * deterministic instead of depending on whether something happened to reuse it. */
  unsigned marker_f4 = 0x5A5A5A5Au;
  {
    enum { NPROBE = 16 };
    static void *probe[NPROBE];
    int np = 0, i;
    /* l1 is the node payload length just read out of %_node, i.e. pRtree->iNodeSize.
     * The node block is sizeof(RtreeNode)+iNodeSize, and memsys5 rounds both to the
     * same power-of-two size class, so a request of l1 competes for that block.
     * Do NOT hardcode page_size-64: getNodeSize clamps iNodeSize so at most
     * RTREE_MAXCELLS cells fit, which at page_size=1024 gives 820, not 960. */
    int nodesz = (int)l1;
    for(i = 0; i < NPROBE; i++){
      void *q = sqlite3_malloc(nodesz);
      if(!q) break;
      probe[np++] = q;
    }
    /* Keep them allocated so the marker stays put while the re-step reads through the
     * dangling SQLITE_STATIC pointer. Whether the freed node block is among them is
     * answered by marker_in_row below, not here: memsys5 has already overwritten the
     * freed block's first eight bytes with Mem5Link{next,prev}, so there is nothing at
     * allocation time that identifies it as the node's. */
    for(i = 0; i < np; i++) memset(probe[i], 0x5A, (size_t)nodesz);
    out_text("rtreestatic probe_allocs="); out_uint((unsigned)np);
    out_text(" size="); out_uint((unsigned)nodesz);
    out_text(" filled with 0x5A; recovery is reported as marker_in_row below\n");
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
