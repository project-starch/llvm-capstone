/* row10 — sqlite3-ruby "Closed Statement" reuse, BINDING-FAITHFUL repro (level 2).
 *   thread: groups.google.com/g/sqlite3-ruby/c/SGRQE_2MZ8I
 *
 * Same sqlite3-ruby glue as row9:
 *   sqlite3Ruby     { sqlite3 *db; sqlite3StmtRuby **stmts; ... }
 *   sqlite3StmtRuby { sqlite3_stmt *st; int done_p; }
 * Statement#reset! (reset_bang) does: REQUIRE_OPEN_STMT(ctx); sqlite3_reset(ctx->st);
 * and the REQUIRE_OPEN_STMT guard checks ONLY `ctx->st != NULL`.
 *
 * The bug: when the Database is torn down it finalizes the statements prepared
 * against it (freeing each underlying sqlite3_stmt) but does NOT reach into the
 * child Statement wrapper to NULL its `st`. A later Statement#reset! (reusing the
 * "closed" statement) therefore passes REQUIRE_OPEN_STMT — the pointer is
 * non-NULL, merely dangling — and calls sqlite3_reset() on the freed statement
 * -> use-after-free INSIDE sqlite3_reset.
 *
 * This is the direct analog of row4 (PHP: reset() without an initialised guard):
 * a liveness check that only tests NULL, not revocation, so a freed-but-retained
 * handle slips through. HIERARCHICAL-REVOKE shape: the Database owns the
 * Statement; revoking the Database's authority on close should invalidate the
 * child's `st` capability so reset! faults locally instead of touching freed
 * memory.
 *
 * Faithful to: the wrapper structs + field names, the parent->child ownership,
 * REQUIRE_OPEN_STMT checking only NULL, reset_bang -> sqlite3_reset(ctx->st), and
 * the retained dangling `st`. Approximated: Ruby GC/ivars -> explicit list.
 * Self-contained: links real SQLite; no Ruby VM.
 * Oracle: heap-use-after-free (ASan) inside sqlite3_reset.
 */
#include <stdlib.h>
#include "sqlite3.h"

typedef struct {
	sqlite3_stmt *st;
	int done_p;
} sqlite3StmtRuby;

typedef struct {
	sqlite3 *db;
	sqlite3StmtRuby **stmts;
	int nstmts;
} sqlite3Ruby;

#define REQUIRE_OPEN_STMT(ctx) \
	do { if (!(ctx)->st) return; } while (0)   /* raises "closed statement" in Ruby */

/* Statement#reset! : pre-fix path — guard only checks st != NULL. */
static void stmt_reset_bang(sqlite3StmtRuby *ctx)
{
	REQUIRE_OPEN_STMT(ctx);           /* passes: st is non-NULL but dangling */
	sqlite3_reset(ctx->st);           /* UAF: ctx->st was finalized by db teardown */
	ctx->done_p = 0;
}

/* Database teardown: finalize the tracked statements (frees each sqlite3_stmt),
 * but leave the child wrappers' `st` fields dangling (non-NULL). */
static void db_teardown(sqlite3Ruby *db_obj)
{
	for (int i = 0; i < db_obj->nstmts; i++)
		sqlite3_finalize(db_obj->stmts[i]->st);   /* frees the sqlite3_stmt */
	sqlite3_close_v2(db_obj->db);
	db_obj->db = NULL;
}

int main(void)
{
	sqlite3Ruby *db_obj = calloc(1, sizeof(*db_obj));
	sqlite3_open(":memory:", &db_obj->db);

	sqlite3StmtRuby *stmt = calloc(1, sizeof(*stmt));
	sqlite3_prepare_v2(db_obj->db, "SELECT 1", -1, &stmt->st, 0);
	sqlite3_step(stmt->st);            /* first use, before close */

	db_obj->stmts = calloc(1, sizeof(*db_obj->stmts));
	db_obj->stmts[0] = stmt;
	db_obj->nstmts = 1;

	db_teardown(db_obj);     /* finalizes stmt->st, leaves it dangling (not NULLed) */
	stmt_reset_bang(stmt);   /* reuse the "closed" statement: reset on freed stmt -> UAF */
	return 0;
}
