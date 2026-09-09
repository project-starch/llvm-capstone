# PROVENANCE — row18 (PHP #79294 columnType after reset)

- Upstream / fix : https://bugs.php.net/bug.php?id=79294
                   (commit `f133f0024ec801dc9636ee5bf84a93de1300d4b2`)
- Tier           : **STALE-STATE, NOT memory-safety** (confirms N/A-for-literal)

**Real memory-safety bug: NO.** Title: "::columnType() may fail after
SQLite3Stmt::reset()". After a `SQLite3Stmt::reset()` (vs `SQLite3Result::reset()`),
`columnType()` keeps returning FALSE instead of the expected value — wrong output,
no memory unsafety. Marked "Incorrect behavior."

Verbatim reproducer (bug #79294):
```php
<?php
$db = new SQLite3(':memory:');
$db->exec("CREATE TABLE foo (bar INT)");
$db->exec("INSERT INTO foo VALUES (1)");
$stmt = $db->prepare("SELECT * FROM foo");
$res = $stmt->execute();
var_dump($res->fetchArray() !== false);
var_dump($res->columnType(0));
$stmt->reset();
var_dump($res->fetchArray() !== false);
var_dump($res->columnType(0));   // expected int(1); returns bool(false)
?>
```

**Consequence.** Logic/stale-state, memory-safe on the host (ASan-clean, oracle
`none`). No free/revoke event. row18 stays a BORROW-REVOKE **probe by nature** — a
row-metadata borrow we *could* impose, but the real bug is not memory unsafety.
Confirms the fidelity-ceiling decision.
