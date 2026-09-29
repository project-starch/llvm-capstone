# row19_php_result_lifecycle

Upstream: https://github.com/php/php-src/pull/5204
Table 3 row: 19
Class: stale-state
Essence: A borrowed result pointer from one row is read after stepping the statement to the next row.
Observed: no ASan report; borrowed-after-restep=alpha current=beta - see NOTE.md

Stage-2 "after" (Capstone): the validated borrow-revoke template
`tests/runtime-qemu/sqlite-borrow-revoke-probe/` demonstrates exactly this shape —
a borrowed row pointer valid before the step and TRAPPED after the next step
revokes it. Row 19 is the most direct instance of that probe (the revoke fires at
the second sqlite3_step); the mechanism is identical, only the narrated lifecycle
point differs.
