/* row12 — expo-sqlite PR #34992, BINDING-FAITHFUL repro (level 2, behavioral).
 *
 * Models expo-sqlite's native module (packages/expo-sqlite,
 * NativeDatabaseBinding.cpp on Android / SQLiteModule.swift on iOS): a database
 * binding object owning a `sqlite3 *db` handle.
 *
 * The bug: closeDatabase() calls sqlite3_close(db). With a live (un-finalized)
 * prepared statement, sqlite3_close returns SQLITE_BUSY and frees nothing — but
 * the pre-fix module IGNORED the SQLITE_BUSY return, cleared its handle
 * (db = NULL / marked closed), and reported success. A later native operation
 * that reads database state then dereferences the now-NULL handle.
 *   github.com/expo/expo/pull/34992: with React Query / Drizzle leaving prepared
 *   statements live, close() hit SQLITE_BUSY; the module nulled its handle anyway,
 *   yielding a NullPointerException on close and, on the native side, a NULL db
 *   dereference. The fix adds sqlite3_finalize_all_statement() (iterate
 *   sqlite3_next_stmt + sqlite3_finalize) so close() succeeds before the handle
 *   is cleared.
 *
 * Here the later "read database state" is sqlite3_get_autocommit(this->db): a
 * real SQLite API (expo checks transaction state this way) that dereferences
 * db->autoCommit with no NULL guard (API armor off) -> SEGV on NULL.
 *
 * BEHAVIORAL TRANSLATION: expo's native binding is C++/JNI and cannot build in a
 * bare Capstone domain; this file reproduces the SQLite-facing C slice — close
 * ignoring SQLITE_BUSY, clearing the handle with a statement still live, then a
 * later use dereferencing the cleared handle. Oracle: null-deref (SEGV on NULL).
 *
 * Faithful to: the owned `sqlite3 *db` handle, sqlite3_close returning
 * SQLITE_BUSY on a live statement, the pre-fix "ignore BUSY + clear handle"
 * behavior, and a later state read that derefs the cleared handle. Approximated:
 * C++/JNI object + Java handle -> a C struct. Self-contained: links real SQLite.
 */
#include <stdlib.h>
#include "sqlite3.h"

/* Mirror of the NativeDatabaseBinding's SQLite-facing state. */
typedef struct {
	sqlite3 *db;
	int closed;
} NativeDatabaseBinding;

/* closeDatabase(): PRE-FIX — does not finalize live statements, ignores BUSY. */
static int native_close_database(NativeDatabaseBinding *self)
{
	int ret = sqlite3_close(self->db);   /* SQLITE_BUSY: a statement is still live */
	/* BUG: the return code is dropped; the handle is cleared regardless. The fix
	 * would finalize all statements (sqlite3_next_stmt + sqlite3_finalize) first
	 * and only clear the handle when close actually returned SQLITE_OK. */
	self->db = NULL;
	self->closed = 1;
	return ret;                          /* caller ignores this too */
}

/* A later native operation that reads database state (e.g. isInTransaction). */
static int native_in_transaction(NativeDatabaseBinding *self)
{
	/* sqlite3_get_autocommit(NULL) dereferences db->autoCommit -> SEGV. */
	return sqlite3_get_autocommit(self->db) == 0;
}

int main(void)
{
	NativeDatabaseBinding *self = calloc(1, sizeof(*self));
	sqlite3_open(":memory:", &self->db);

	sqlite3_stmt *live = NULL;
	sqlite3_prepare_v2(self->db, "SELECT 1", -1, &live, 0);  /* left un-finalized */

	native_close_database(self);        /* close() -> SQLITE_BUSY, handle cleared to NULL */
	return native_in_transaction(self); /* reads state on the NULL handle -> null-deref */
}
