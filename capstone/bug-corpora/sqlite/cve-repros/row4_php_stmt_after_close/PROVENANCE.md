# PROVENANCE — row4 (PHP #66550 statement-after-close UAF)

- Upstream : https://bugs.php.net/bug.php?id=66550
- Fix      : commit `5ae20c624781bdd39ba14b2f856234c168f7ea38` (adds
             `SQLITE3_CHECK_INITIALIZED` to statement methods)
- Tier     : **LITERAL-traceable**

**Real memory-safety bug: YES** (use-after-free). `$db->close()` runs
`sqlite3_close()`, which tears down the prepared statements via
`php_sqlite3_free_list_dtor()` (`sqlite3_finalize` each, sets `initialised = 0`),
but the PHP wrapper objects stay live — so a later `$stmt->reset()` operates on
freed memory.

Verbatim reproducer (bug #66550):
```php
<?php
$db = new SQLite3(':memory:');
$db->exec('CREATE TABLE foo (id INTEGER, bar STRING)');
$stmt = $db->prepare('SELECT bar FROM foo WHERE id=:id');
$db->close();
$stmt->reset();
?>
```
Verbatim backtrace:
```
#0 sqlite3VdbeHalt (sqlite3.c:59669)
#1 sqlite3VdbeReset (sqlite3.c:59874)
#2 sqlite3_reset (sqlite3.c:60898)
#3 zim_sqlite3stmt_reset (sqlite3.c:1316)
```

**How our repro maps:** row4 is a HIERARCHICAL member — the connection wrapper is
the parent; the statement is its child, revoked when the parent closes; the later
`reset` on the child faults. Faithful freed-parent → child-deref UAF.
