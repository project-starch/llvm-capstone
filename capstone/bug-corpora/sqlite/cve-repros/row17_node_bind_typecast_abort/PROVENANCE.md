# PROVENANCE — row17 (node-sqlite3 #1449 bind typecast abort)

- Upstream : https://github.com/TryGhost/node-sqlite3/issues/1449
- Tier     : **N/A for the capability model** (crash is in the Node/napi glue, not SQLite)

**Correction to the essence.** The row's essence says SQLite "rejects it without a
memory-safety violation" (SQLITE_RANGE). The issue actually reports a **fatal
abort/segfault** — but in the **Node napi binding layer**, not SQLite: when a bind
value's type coercion fails, `statement.cc:BindParameters` calls
`Napi::String::Utf8Value`, whose error path mis-drives `Napi::Error::New` /
`napi_get_last_error_info` and aborts.

Verbatim reproducer (issue #1449):
```javascript
var sqlite3 = require('sqlite3').verbose();
var db = new sqlite3.Database(':memory:');
db.serialize(function() {
  try { db.run("INSERT INTO lorem VALUES (?)", [{toString: 23}]) }
  catch(e) { console.log("caught exception") }
});
db.close();
```

**Consequence.** The crash is a Node-binding error-handling abort, **not** an SQLite
C-API pointer-lifetime bug — there is no SQLite handle whose lifetime a capability
could protect. row17 stays **N/A** for the corpus (no in-scope memory-safety event
in SQLite). The host repro is memory-safe w.r.t. SQLite (oracle `none`).
