/* row2 — rusqlite RUSTSEC-2021-0128, BINDING-FAITHFUL repro (level 2).
 *
 * PROVENANCE (real upstream artifacts, not paraphrase):
 *   Advisory : RUSTSEC-2021-0128 "Incorrect Lifetime Bounds on Closures in
 *              rusqlite" (== CVE-2021-45713, GHSA-q89g-4vhh-mvvm).
 *   Affected : rusqlite 0.25.0-0.25.3, 0.26.0-0.26.1.  Patched: 0.25.4, 0.26.2.
 *   Issue    : https://github.com/rusqlite/rusqlite/issues/1048
 *   Glue     : rusqlite src/functions.rs (call_boxed_closure / free_boxed_value),
 *              src/hooks.rs (update_hook).
 *
 * The VERBATIM reproducer from issue #1048 (update_hook path):
 *
 *     use rusqlite::{hooks::Action, Connection};
 *     use std::sync::{Arc, Mutex};
 *     fn main() {
 *         let db = Connection::open_in_memory().unwrap();
 *         {
 *             let locked: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
 *             let hook = |_: Action, _: &str, _: &str, _: i64| {
 *                 if let Ok(ref mut mutex) = locked.try_lock() {
 *                     **mutex = ();
 *                 }
 *             };
 *             db.update_hook(Some(hook));
 *         };                                   // <- `locked` (the Arc) is dropped here
 *         db.execute("CREATE TABLE tbl(i integer)", []).unwrap();
 *         db.execute("INSERT INTO tbl (i) VALUES (1)", []).unwrap();  // fires hook -> UAF
 *     }
 *
 * The bug: `hook` is a NON-move closure, so it BORROWS `locked` rather than owning
 * it. rusqlite boxes the closure and registers it with SQLite; SQLite keeps it and
 * invokes it on every row change. When the inner scope ends, `locked` (an Arc, heap
 * refcount) is dropped, but the hook stays registered. The INSERT fires the hook,
 * which derefs the dropped Arc -> use-after-free. Adding `move` (ownership into the
 * closure) or the fix's `+ 'static` bound (which rejects the borrow at compile time)
 * both prevent it.
 *
 * The REAL glue this models (rusqlite, quoted):
 *   - the fix = a lifetime bound on the closure:
 *       pub fn create_scalar_function<F, N: Name, T>(...) -> Result<()>
 *       where F: Fn(&Context<'_>) -> Result<T> + Send + 'static, ...
 *     (update_hook grew the analogous bound in hooks.rs.)
 *   - the boxed closure is stored as SQLite user-data and reached on each call:
 *       unsafe extern "C" fn call_boxed_closure<F, T>(ctx, argc, argv) {
 *           let boxed_f: *const F = ffi::sqlite3_user_data(ctx).cast::<F>();
 *           (*boxed_f)(&ctx)                  // <- deref of the boxed closure
 *       }
 *   - free_boxed_value::<F> is the xDestroy that frees the box on drop.
 *
 * FIDELITY: this lowers the ACTUAL #1048 reproducer (update_hook, a borrowed Arc
 * dropped at scope end, an INSERT firing the still-registered hook). The essence
 * before.c uses create_scalar_function and frees pApp itself; this models the real
 * defect — the box stays live, the value it BORROWED does not. Same crash class
 * (heap-use-after-free). Self-contained: real SQLite, real sqlite3_update_hook +
 * update-hook dispatch; the Arc/closure/borrow-checker are lowered to C.
 */
#include <stdlib.h>
#include "sqlite3.h"

/* The boxed closure environment. The non-move closure captures `locked` by
 * reference, so the box holds a BORROWED pointer to it — not ownership. */
struct boxed_closure {
	int *locked;   /* &Arc<Mutex<()>> : borrowed, not owned (the non-'static capture) */
};

/* rusqlite hooks.rs update-hook trampoline: reach the boxed closure through the
 * user-data pointer SQLite stored, then run the closure body. */
static void call_update_hook(void *p_arg, int op, const char *db,
                             const char *table, sqlite3_int64 rowid)
{
	struct boxed_closure *hook = p_arg;
	(void)op;
	(void)db;
	(void)table;
	(void)rowid;
	/* closure body: `if let Ok(mut m) = locked.try_lock() { **m = () }` */
	volatile int touch = *hook->locked;   /* UAF: `locked` (the Arc) was dropped */
	(void)touch;
}

int main(void)
{
	sqlite3 *db = 0;
	sqlite3_open(":memory:", &db);

	struct boxed_closure *hook = malloc(sizeof(*hook));

	{
		/* let locked: Arc<Mutex<()>> = Arc::new(Mutex::new(())); */
		int *locked = malloc(sizeof(*locked));
		*locked = 0;
		hook->locked = locked;                       /* closure borrows `locked` */
		sqlite3_update_hook(db, call_update_hook, hook);  /* db.update_hook(Some(hook)) */
		free(locked);                                /* end of scope: Arc dropped */
	}

	/* db.execute(CREATE TABLE ...); db.execute(INSERT ...) — the INSERT fires the
	 * still-registered update hook, which derefs the dropped Arc. */
	sqlite3_exec(db, "CREATE TABLE tbl(i integer)", 0, 0, 0);
	sqlite3_exec(db, "INSERT INTO tbl (i) VALUES (1)", 0, 0, 0);   /* -> use-after-free */

	free(hook);
	sqlite3_close(db);
	return 0;
}
