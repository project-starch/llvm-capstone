/* row8 — CPython gh-85981 / bpo-41815, BINDING-FAITHFUL repro (level 2).
 *
 * Models CPython's _sqlite C glue (Modules/_sqlite/connection.c): a
 * pysqlite_Connection wrapper with an `initialized` flag and a `db` handle.
 * Connection.close() runs sqlite3_close_v2(self->db) and then sets
 * self->db = NULL (defensive nulling). Connection.backup(target) reads the
 * SOURCE handle from self->db and the DEST handle from ((Connection*)target)->db
 * and calls sqlite3_backup_init(dest, "main", source, "main").
 *
 * The bug (pre-fix): backup() had NO check_connection(self) guard, so calling
 * source.backup(target) AFTER source.close() passes self->db == NULL to
 * sqlite3_backup_init as pSrcDb. Real SQLite dereferences pSrcDb->mutex with no
 * NULL check (API armor off) -> segfault.
 *
 * PROVENANCE (real upstream artifacts, not paraphrase):
 *   Issue : bpo-41815 / https://github.com/python/cpython/issues/85981
 *   Fix   : GH-22322 (backports GH-22344 3.9, GH-22345 3.8)
 *   File  : Modules/_sqlite/connection.c (the backup method)
 *   Verbatim regression test added by the fix (Lib/sqlite3/test/backup.py):
 *     def test_bad_source_closed_connection(self):
 *         bck = sqlite.connect(':memory:')
 *         source = sqlite.connect(":memory:")
 *         source.close()
 *         with self.assertRaises(sqlite.ProgrammingError):
 *             source.backup(bck)        # pre-fix: NULL source handle -> SEGV
 *   Verbatim fix hunk added to the backup method:
 *     if (!pysqlite_check_thread(self) || !pysqlite_check_connection(self)) {
 *         return NULL;
 *     }
 *
 * FIDELITY NOTE: the real defect is a NULL/closed-handle dereference, NOT a
 * heap-use-after-free. pysqlite nulls self->db on close, so backup reads a NULL
 * source handle. The essence before.c fabricates a UAF (it frees the wrapper and
 * re-reads the field); the honest faithful crash class here is null-deref. See
 * NOTE.md. Oracle for THIS file: SEGV on NULL inside sqlite3_backup_init.
 *
 * Faithful to: the initialized/db wrapper fields, close()'s sqlite3_close_v2 +
 * self->db=NULL, backup()'s source=self->db / dest=target->db handle sourcing,
 * and the missing check_connection guard. Approximated: the CPython object/GC
 * machinery. Self-contained: links real SQLite; no CPython runtime.
 */
#include <stdlib.h>
#include "sqlite3.h"

/* Mirror of pysqlite_Connection (the fields the backup path touches). */
typedef struct {
	int initialized;
	sqlite3 *db;
} pysqlite_Connection;

/* pysqlite_check_connection(): the guard the fix adds to backup(). */
static int pysqlite_check_connection(pysqlite_Connection *con)
{
	if (!con->initialized)
		return 0;   /* "Base Connection.__init__ not called." */
	if (!con->db)
		return 0;   /* "Cannot operate on a closed database." */
	return 1;
}

/* Connection.close(): finalize/close and NULL the handle. */
static void pysqlite_connection_close(pysqlite_Connection *self)
{
	if (self->db) {
		sqlite3_close_v2(self->db);
		self->db = NULL;            /* defensive nulling on close */
	}
}

/* Connection.backup(target): PRE-FIX path — no check_connection(self) guard. */
static sqlite3_backup *pysqlite_connection_backup(pysqlite_Connection *self,
                                                  pysqlite_Connection *target)
{
	/* Fix (GH-22322) inserts here:
	 *   if (!pysqlite_check_thread(self) || !pysqlite_check_connection(self))
	 *       return NULL;                                                      */
	sqlite3 *bck_conn = target->db;               /* dest */
	/* source = self->db, which close() set to NULL -> NULL deref in init */
	return sqlite3_backup_init(bck_conn, "main", self->db, "main");
}

int main(void)
{
	pysqlite_Connection *bck = calloc(1, sizeof(*bck));
	sqlite3_open(":memory:", &bck->db);
	bck->initialized = 1;

	pysqlite_Connection *source = calloc(1, sizeof(*source));
	sqlite3_open(":memory:", &source->db);
	source->initialized = 1;

	pysqlite_connection_close(source);        /* source.close(): source->db = NULL */

	/* source.backup(bck): passes NULL source handle to sqlite3_backup_init. */
	sqlite3_backup *b = pysqlite_connection_backup(source, bck);
	if (b)
		sqlite3_backup_finish(b);
	sqlite3_close(bck->db);
	(void)pysqlite_check_connection;          /* referenced by the fix, unused pre-fix */
	return 0;
}
