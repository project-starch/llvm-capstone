# row17_node_bind_typecast_abort

Upstream: https://github.com/TryGhost/node-sqlite3/issues/1449
Table 3 row: 17
Class: stale-state
Essence: A malformed host bind request is submitted at an invalid parameter index and SQLite rejects it without a memory-safety violation.
Observed: no ASan report; bind rc=25 (column index out of range) - see NOTE.md
