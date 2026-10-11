# Logical lifetime violation only

The program intentionally calls `sqlite3_column_type()` after
`sqlite3_reset()`. SQLite 3.53.3 returns a value without touching poisoned
memory, so ASan has no report. The printed before/after values document the
current behavior; the post-reset query is outside the valid row lifetime.
