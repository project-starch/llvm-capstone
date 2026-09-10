/* row9 — sqlite3-ruby issue #49, BINDING-FAITHFUL repro (level 2).
 *
 * Models sqlite3-ruby's C glue (ext/sqlite3/{database,statement}.c):
 *   sqlite3Ruby     { sqlite3 *db; }                 // Database wrapper
 *   sqlite3StmtRuby { sqlite3_stmt *st; int done_p; } // Statement wrapper
 * A Statement keeps a Ruby ivar back-reference to its Database (@connection).
 * Statement#close does: REQUIRE_OPEN_STMT(ctx); sqlite3_finalize(ctx->st);
 * ctx->st = NULL;  — the guard only checks ctx->st != NULL.
 *
 * The bug (github.com/sparklemotion/sqlite3-ruby/issues/49): a Statement outlives
 * its Database and their GC finalizers run in an unlucky order. Database teardown
 * (clear_cache! / db close) finalizes the parent's tracked statements — freeing
 * the underlying sqlite3_stmt — but the child Statement wrapper still holds a
 * non-NULL, now-dangling `st`. When that Statement's own #close later runs,
 * REQUIRE_OPEN_STMT passes (st != NULL) and sqlite3_finalize(ctx->st) is called a
 * SECOND time on the freed statement -> use-after-free INSIDE sqlite3_finalize
 * (the observed backtrace: "sqlite3_finalize+0x23").
 *
 * This is the HIERARCHICAL-REVOKE shape: the Database owns its Statements; the
 * parent's teardown finalized the child's stmt, but nothing invalidated the
 * child's retained handle, so the child double-finalizes. Under Capstone, closing
 * the Database revokes the child's `st` capability, so the child's finalize faults
 * locally instead of touching freed memory.
 *
 * Faithful to: the two wrapper structs and field names, the parent->child
 * ownership, REQUIRE_OPEN_STMT checking only NULL, and the double sqlite3_finalize
 * that the observed crash lands in. Approximated: Ruby GC/ivars -> an explicit
 * child list + retained pointer. Self-contained: links real SQLite; no Ruby VM.
 * Oracle: heap-use-after-free (ASan) inside sqlite3_finalize.
 */
#include <stdlib.h>
#include "sqlite3.h"

typedef struct {
	sqlite3_stmt *st;
	int done_p;
} sqlite3StmtRuby;

/* A Database wrapper that tracks the statements prepared against it. */
typedef struct {
	sqlite3 *db;
	sqlite3StmtRuby **stmts;   /* tracked children (the @statements cache) */
	int nstmts;
} sqlite3Ruby;

#define REQUIRE_OPEN_STMT(ctx) \
	do { if (!(ctx)->st) return; } while (0)   /* raises "closed statement" in Ruby */

/* Statement#close: finalize and NULL — but see the double call below. */
static void stmt_rb_close(sqlite3StmtRuby *ctx)
{
	REQUIRE_OPEN_STMT(ctx);            /* only checks st != NULL */
	sqlite3_finalize(ctx->st);        /* 2nd call here -> UAF inside finalize */
	ctx->st = NULL;
}

/* Database teardown (clear_cache! / close): finalize each tracked statement's
 * underlying stmt, but do NOT reach into the child wrapper to NULL its `st`
 * (the parent only owns the sqlite3_stmt, not the child's Ruby field). */
static void db_teardown(sqlite3Ruby *db_obj)
{
	for (int i = 0; i < db_obj->nstmts; i++)
		sqlite3_finalize(db_obj->stmts[i]->st);  /* frees the sqlite3_stmt */
	sqlite3_close_v2(db_obj->db);
	db_obj->db = NULL;
}

int main(void)
{
	sqlite3Ruby *db_obj = calloc(1, sizeof(*db_obj));
	sqlite3_open(":memory:", &db_obj->db);

	sqlite3StmtRuby *stmt = calloc(1, sizeof(*stmt));
	sqlite3_prepare_v2(db_obj->db, "SELECT 1", -1, &stmt->st, 0);

	/* Database tracks the statement (the @statements cache). */
	db_obj->stmts = calloc(1, sizeof(*db_obj->stmts));
	db_obj->stmts[0] = stmt;
	db_obj->nstmts = 1;

	db_teardown(db_obj);    /* parent finalizes stmt->st, leaves child's st dangling */
	stmt_rb_close(stmt);    /* child #close: 2nd sqlite3_finalize on freed stmt -> UAF */
	return 0;
}
