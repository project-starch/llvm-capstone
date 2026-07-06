# No SQLite C-API memory error

Binding text at parameter index 2 when the statement only has `?1` returns
`SQLITE_RANGE` (`25`). The node-sqlite3 abort depends on binding-layer
type-conversion logic that is not part of the SQLite C API, so this standalone
C model intentionally expects no ASan report.
