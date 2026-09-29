/* row4 — PHP bug #66550, BINDING-FAITHFUL repro (level 2).
 *
 * Models PHP ext/sqlite3 (same object model as row5): a php_sqlite3_db_object
 * owns a `zend_llist free_list` of statements; each php_sqlite3_stmt_object
 * keeps a `db_obj` back-pointer, its sqlite3_stmt, and an `initialised` flag.
 *
 * The bug: SQLite3::close() runs zend_llist_clean(free_list) ->
 * php_sqlite3_free_list_dtor(), which sqlite3_finalize()s each statement
 * (freeing the underlying sqlite3_stmt) and sets the wrapper's initialised=0 —
 * but leaves the wrapper's now-dangling `stmt` pointer in place. A later
 * SQLite3Stmt::reset() then, in the pre-fix code (no SQLITE3_CHECK_INITIALIZED
 * guard), calls sqlite3_reset() on the freed statement -> use-after-free.
 *   Repro (PHP), bugs.php.net/bug.php?id=66550:
 *     $db = new SQLite3(':memory:');
 *     $db->exec('CREATE TABLE foo (id INTEGER, bar STRING)');
 *     $stmt = $db->prepare('SELECT bar FROM foo WHERE id=:id');
 *     $db->close();     // finalizes stmt, frees sqlite3_stmt, initialised=0
 *     $stmt->reset();   // dereferences the freed sqlite3_stmt
 *
 * Faithful to: the free_list teardown, php_sqlite3_free_list_dtor's
 * finalize-and-mark-stale behaviour, the retained dangling stmt pointer, and
 * the missing initialised guard on reset(). Approximated: zend_llist reduced to
 * a minimal list. Self-contained: links real SQLite; no Zend/PHP runtime.
 * Oracle: heap-use-after-free (ASan) inside sqlite3_reset on the finalized stmt.
 */
#include <stdlib.h>
#include "sqlite3.h"

struct llist_node {
	void *data;
	struct llist_node *next;
};
typedef struct {
	struct llist_node *head;
	size_t count;
} zend_llist;

typedef struct {
	int initialised;
	sqlite3 *db;
	zend_llist free_list;
} php_sqlite3_db_object;

typedef struct {
	sqlite3_stmt *stmt;
	php_sqlite3_db_object *db_obj;
	int initialised;
} php_sqlite3_stmt_object;

static void llist_add(zend_llist *l, void *data)
{
	struct llist_node *n = malloc(sizeof(*n));
	n->data = data;
	n->next = l->head;
	l->head = n;
	l->count++;
}

/* php_sqlite3_free_list_dtor(): finalize the underlying statement and mark the
 * wrapper uninitialised — but do NOT clear the dangling stmt pointer. */
static void php_sqlite3_free_list_dtor(php_sqlite3_stmt_object *stmt)
{
	if (stmt->stmt)
		sqlite3_finalize(stmt->stmt);   /* frees the sqlite3_stmt */
	stmt->initialised = 0;              /* wrapper marked stale */
}

/* SQLite3::close(): tear down tracked statements, then close the connection. */
static void php_sqlite3_close(php_sqlite3_db_object *db_obj)
{
	for (struct llist_node *n = db_obj->free_list.head; n; n = n->next)
		php_sqlite3_free_list_dtor(n->data);   /* zend_llist_clean */
	sqlite3_close_v2(db_obj->db);
	db_obj->initialised = 0;
}

/* SQLite3Stmt::reset(): pre-fix path — no SQLITE3_CHECK_INITIALIZED guard. */
static void php_sqlite3_stmt_reset(php_sqlite3_stmt_object *stmt)
{
	sqlite3_reset(stmt->stmt);   /* UAF: stmt->stmt was finalized by close() */
}

int main(void)
{
	php_sqlite3_db_object *db_obj = calloc(1, sizeof(*db_obj));
	sqlite3_open(":memory:", &db_obj->db);
	db_obj->initialised = 1;
	sqlite3_exec(db_obj->db, "CREATE TABLE foo (id INTEGER, bar STRING)", 0, 0, 0);

	php_sqlite3_stmt_object *stmt = calloc(1, sizeof(*stmt));
	sqlite3_prepare_v2(db_obj->db, "SELECT bar FROM foo WHERE id=:id", -1,
	                   &stmt->stmt, 0);
	stmt->db_obj = db_obj;
	stmt->initialised = 1;
	llist_add(&db_obj->free_list, stmt);

	php_sqlite3_close(db_obj);       /* $db->close(): finalizes stmt->stmt */
	php_sqlite3_stmt_reset(stmt);    /* $stmt->reset(): UAF on freed stmt */
	return 0;
}
