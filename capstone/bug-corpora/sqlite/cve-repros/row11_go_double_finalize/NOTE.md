# Expected-class mismatch

Passing the same dangling `sqlite3_stmt *` to `sqlite3_finalize()` twice does
fail under ASan, but SQLite 3.53.3 first follows a stale database mutex pointer.
ASan therefore reports a read SEGV in `pthread_mutex_lock`, not
`double-free`. The `oracle` retains the proposal's target class so the mismatch
is visible rather than silently reclassified.
