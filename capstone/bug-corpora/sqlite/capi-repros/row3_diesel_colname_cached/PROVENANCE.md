# PROVENANCE — row3 (diesel RUSTSEC-2021-0037)

- Upstream : https://rustsec.org/advisories/RUSTSEC-2021-0037.html
- Fix PR   : https://github.com/diesel-rs/diesel/pull/2663
- Affected : diesel `< 1.4.6` (`SqliteConnection::query_by_name`). Patched: `1.4.6`.
- Tier     : **LITERAL-traceable**

**Real memory-safety bug: YES.** diesel cached the string slices returned by
`sqlite3_column_name` as borrowed field names, then called `sqlite3_step()`, which
invalidates those pointers → use-after-free.

Verbatim invalidation rule (SQLite docs, quoted by the advisory):
> "The returned string pointer is valid until either the prepared statement is
> destroyed by sqlite3_finalize() or until the statement is automatically
> reprepared by the first call to sqlite3_step()."

Verbatim reproducer: none in the advisory.

**How our repro maps:** row3 B2 (task-008, the literal R representative) revokes
SQLite's own column pointer at `sqlite3_finalize` — one of the two documented
invalidation points. NOTE: the advisory's *actual* trigger is the next
`sqlite3_step()` (auto-reprepare); finalize is the other documented invalidation.
Both are faithful to the same borrow-lifetime rule. A stricter lowering would fire
the revoke at `sqlite3_step` (see row19, which does exactly that borrow-at-step).
