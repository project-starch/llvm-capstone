# row18_php_columntype_after_reset

Upstream: https://bugs.php.net/bug.php?id=79294
Table 3 row: 18
Class: stale-state
Essence: Host code asks for a result-column type after resetting the statement, outside the valid row lifetime.
Observed: no ASan report; before=1 after-reset=5 - see NOTE.md

Stage-2 "after" (Capstone): borrow-revoke template
`tests/runtime-qemu/sqlite-borrow-revoke-probe/`, with the revoke fired at
sqlite3_reset. Weak fit: the Stage-1 defect is a logical stale-scalar read, so the
"after" models the column metadata as a row-scoped borrow that reset revokes — a
subsequent read then faults rather than returning stale metadata.
