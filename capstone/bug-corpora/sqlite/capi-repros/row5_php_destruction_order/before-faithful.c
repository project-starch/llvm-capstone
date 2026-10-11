/* row5 — PHP bug #69971, BINDING-FAITHFUL repro (level 2).
 *
 * Models PHP ext/sqlite3's real object model (php_sqlite3_structs.h):
 *   - php_sqlite3_db_object owns a `zend_llist free_list` tracking its live
 *     statement objects;
 *   - php_sqlite3_stmt_object keeps a back-pointer `db_obj` to its parent
 *     connection.
 *
 * The bug: at request shutdown the Zend GC frees the DB object BEFORE the
 * statement object. The statement free handler then calls
 * php_sqlite3_compare_stmt_free(), which walks `intern->db_obj->free_list` to
 * unlink itself — but that memory was already freed with the db object.
 *   Dev note (bugs.php.net/bug.php?id=69971): "php_sqlite3_db_object is
 *   destroyed before php_sqlite3_stmt by shutdown handler, but stmt object
 *   tries to access intern->db_obj->free_list, which is in memory already
 *   freed."
 *
 * Faithful to: the two real glue structs and their field names, the
 * statement->connection back-pointer, the free_list tracking relationship, the
 * real crash function name, and the exact wrong destruction order. Approximated:
 * zend_llist is reduced to a minimal linked list (modeling the full Zend list
 * machinery is out of scope and irrelevant to the lifetime bug).
 *
 * Self-contained: links the real SQLite amalgamation; needs no Zend/PHP runtime.
 * Host oracle: heap-use-after-free (ASan), in php_sqlite3_compare_stmt_free.
 */
#include <stdlib.h>
#include "sqlite3.h"

/* --- minimal stand-in for Zend's zend_llist (db_object->free_list) --- */
struct llist_node {
	void *data;                 /* -> php_sqlite3_stmt_object */
	struct llist_node *next;
};
typedef struct {
	struct llist_node *head;
	size_t count;
} zend_llist;

/* --- faithful subset of ext/sqlite3/php_sqlite3_structs.h --- */
typedef struct {
	int initialised;
	sqlite3 *db;
	zend_llist free_list;       /* the connection tracks its live statements */
} php_sqlite3_db_object;

typedef struct {
	sqlite3_stmt *stmt;
	php_sqlite3_db_object *db_obj;   /* back-pointer to the parent connection */
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

/* php_sqlite3_compare_stmt_free(): the real crash site. The statement
 * destructor removes itself from the parent connection's free_list. */
static void php_sqlite3_compare_stmt_free(php_sqlite3_db_object *db_obj,
                                          php_sqlite3_stmt_object *stmt)
{
	struct llist_node **pp = &db_obj->free_list.head;   /* <-- UAF read */
	while (*pp) {
		if ((*pp)->data == stmt) {
			struct llist_node *dead = *pp;
			*pp = dead->next;
			free(dead);
			db_obj->free_list.count--;
			break;
		}
		pp = &(*pp)->next;
	}
}

/* Zend free_storage handler for the DB object — runs FIRST at shutdown. */
static void php_sqlite3_object_free_storage(php_sqlite3_db_object *db_obj)
{
	if (db_obj->initialised)
		sqlite3_close_v2(db_obj->db);   /* close variant is immaterial to the bug */
	struct llist_node *n = db_obj->free_list.head;
	while (n) {
		struct llist_node *next = n->next;
		free(n);
		n = next;
	}
	free(db_obj);
}

/* Zend free_storage handler for the STMT object — runs AFTER the db object. */
static void php_sqlite3_stmt_object_free_storage(php_sqlite3_stmt_object *stmt)
{
	if (stmt->initialised && stmt->stmt)
		sqlite3_finalize(stmt->stmt);
	/* the bug: reach back into the already-freed parent connection */
	php_sqlite3_compare_stmt_free(stmt->db_obj, stmt);
	free(stmt);
}

int main(void)
{
	/* $conn = new SQLite3(':memory:'); $stmt = $conn->prepare('SELECT 1'); */
	php_sqlite3_db_object *db_obj = calloc(1, sizeof(*db_obj));
	sqlite3_open(":memory:", &db_obj->db);
	db_obj->initialised = 1;

	php_sqlite3_stmt_object *stmt = calloc(1, sizeof(*stmt));
	sqlite3_prepare_v2(db_obj->db, "SELECT 1", -1, &stmt->stmt, 0);
	stmt->db_obj = db_obj;
	stmt->initialised = 1;
	llist_add(&db_obj->free_list, stmt);   /* connection tracks the statement */

	/* Request shutdown in the order that triggers #69971: the connection
	 * object is torn down before its statement object. */
	php_sqlite3_object_free_storage(db_obj);      /* frees db_obj + free_list */
	php_sqlite3_stmt_object_free_storage(stmt);   /* UAF: reads db_obj->free_list */
	return 0;
}
