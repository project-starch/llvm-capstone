# PROVENANCE — row10 (sqlite3-ruby stmt-reuse-after-close UAF)

- Upstream : https://groups.google.com/g/sqlite3-ruby/c/SGRQE_2MZ8I
- Fix      : commit `a8cb493` (master + 1-3-stable), released in sqlite3-ruby 1.3.7
- Tier     : **LITERAL-traceable**

**Real memory-safety bug: YES** (use-after-free). A prepared statement that failed
on a constraint violation stays in the cache; during Rails' reload cycle the
connection pool clears cached statements via `close()`, but a subsequent access
operates on freed memory bindings → segfault.

Verbatim minimal reproducer (Aaron Patterson, non-Rails):
```ruby
db = SQLite3::Database.new ':memory:'
db.execute 'CREATE TABLE "things" ("number" float NOT NULL)'
stmt = db.prepare 'INSERT INTO things (number) VALUES (?)'
stmt.execute(nil) rescue nil
stmt.close rescue nil
stmt.close   # => segv
```

**How our repro maps:** row10 is a Ruby HIERARCHICAL/use-after-close member (sibling
of row9). NOTE: the minimal repro above is a *double-close* of one statement (closer
to the LINEAR double-finalize shape of row11); the broader scenario is stmt reuse
after connection teardown. Either way it is a real UAF on a closed statement handle.
