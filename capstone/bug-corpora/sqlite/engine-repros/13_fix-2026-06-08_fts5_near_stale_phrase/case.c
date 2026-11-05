/* row22 / sqlite-2026-06-08 (b677c5afd4) -- fts5ExprNearIsMatch hands each phrase
 * poslist to a lookahead reader, zeroes pPoslist->n, and then writes the trimmed
 * poslist back over the SAME buffer while the reader is still reading it
 * (fts5_expr.c 560/596/627). The comment there argues this is safe because the
 * output is a subset of the input. It is not: every append goes through
 * sqlite3Fts5PoslistWriterAppend, which opens with fts5BufferGrow(pBuf, 5+5+5)
 * and so demands 15 spare bytes. The moment the output length passes nSpace-15
 * sqlite3Fts5BufferSize reallocs and the reader's pointer dangles. Fixed 3.53.3
 * by appending with sqlite3Fts5PoslistSafeAppend, which never grows.
 *
 * THE TRIGGER WAS TOO SHORT UNTIL 2026-10-29 AND THIS CASE MEASURED NOTHING.
 * It ran NEAR(quick brown, 3) over two short rows. The in-place rewrite did run
 * under live readers, but nSpace doubles from 64, so the realloc only happens
 * when the poslist length lands in (nSpace-15, nSpace]. Measured on the host,
 * poslists of 56-66, 120-130 and 252-256 bytes report and every other length is
 * silent; those two rows built poslists far below the first window.
 *
 * CONTROL arm: 'a b' sixty times in one row. 'a' lands on positions 0,2,4,... and
 * 'b' on 1,3,5,..., so each phrase poslist is three bytes for the first entry
 * plus one byte per delta = 62 bytes, inside the nSpace=64 window, and
 * NEAR(a b, 1) keeps every entry so the writer tracks the reader byte for byte.
 * Host ASan reports heap-use-after-free, a 1-byte read 52 bytes inside the freed
 * 64-byte region in sqlite3Fts5PoslistNext64 under fts5LookaheadReaderNext,
 * freed by sqlite3_realloc under sqlite3Fts5PoslistWriterAppend -- the read and
 * the free are both inside fts5ExprNearIsMatch.
 * NOTE: build in fts5 group. */
#include "repro322_common.h"
static int run_case(void){
  if (repro_init()) return 1;
  sqlite3 *db=0; int rc=sqlite3_open(":memory:",&db); if(rc)return FAILRC("open",rc);
  rc=sqlite3_exec(db,"CREATE VIRTUAL TABLE ft USING fts5(x);",0,0,0);
  if(rc)return FAILRC("fts5near create",rc);
  rc=sqlite3_exec(db,"INSERT INTO ft VALUES('"
    "a b a b a b a b a b a b a b a b a b a b "
    "a b a b a b a b a b a b a b a b a b a b "
    "a b a b a b a b a b a b a b a b a b a b "
    "a b a b a b a b a b a b a b a b a b a b "
    "a b a b a b a b a b a b a b a b a b a b "
    "a b a b a b a b a b a b a b a b a b a b"
    "');",0,0,0);
  if(rc)return FAILRC("fts5near insert",rc);
  sqlite3_stmt*st=0;
  if(sqlite3_prepare_v2(db,"SELECT rowid FROM ft WHERE ft MATCH 'NEAR(a b, 1)'",-1,&st,0)==SQLITE_OK){
    int n=0; while(sqlite3_step(st)==SQLITE_ROW) n++;
    out_text("fts5near rows=");out_uint((unsigned)n);out_text("\n");
    sqlite3_finalize(st);
  } else { out_text("fts5near prepare err (");out_text(sqlite3_errmsg(db));out_text(")\n"); }
  sqlite3_close(db);
  out_text("fts5near NOTRAP done\n"); return 0;
}
REPRO322_MAIN("fts5near")
