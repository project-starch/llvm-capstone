# Logical lifetime violation only

The first row's `sqlite3_column_text()` pointer is used after
`sqlite3_step()` advances to the second row. SQLite 3.53.3 reuses or retains
addressable storage without an allocator-level invalid access, so this is a
stale-result contract violation rather than an ASan-detectable UAF.
