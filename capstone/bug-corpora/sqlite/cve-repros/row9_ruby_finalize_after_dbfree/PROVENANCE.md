# PROVENANCE — row9 (sqlite3-ruby #49 finalize-after-dbfree UAF)

- Upstream : https://github.com/sparklemotion/sqlite3-ruby/issues/49
- Tier     : **LITERAL-traceable** (real UAF; no *minimal* reproducer published)

**Real memory-safety bug: YES** (use-after-free). During teardown the database
connection is closed/freed before its prepared statements are finalized:
`clear_cache!` iterates the cached statements and calls each statement's `close`,
but the statement references memory in the already-freed database, so
`sqlite3_finalize` dereferences freed memory → segfault
(`ruby 1.9.2p290 [x86_64-linux]`, 2011).

Verbatim reproducer: none minimal — the issue reports it manifesting during
ActiveRecord's migration-test teardown (connection reset + statement-cache clear).
The lifecycle sequence documented in the issue:
- `clear_cache!` iterates cached statements
- each statement's `close` is invoked
- the statement pointer references freed database memory
- `sqlite3_finalize` dereferences invalid memory → segfault

**How our repro maps:** row9 is a Ruby HIERARCHICAL representative — the database
wrapper is the parent; the cached statement is its child. Parent (database)
teardown revokes the child statement's authority before the late finalize, which
then faults. Faithful to the freed-db → stmt-finalize UAF above. Agent-B task-012
promotes row9 to a literal real-SQLite matched pair (Ruby side of the H family).
