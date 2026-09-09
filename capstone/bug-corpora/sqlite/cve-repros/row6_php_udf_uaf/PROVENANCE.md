# PROVENANCE — row6 (PHP #77977 heap-UAF via UDF)

- Upstream : https://bugs.php.net/bug.php?id=77977
- Status   : "Analyzed", WON'T-FIX (devs deemed it a documentation matter —
             comprehensive prevention infeasible).
- Tier     : **LITERAL-traceable** (real heap UAF, real reproducer; no fix)

**Real memory-safety bug: YES** (heap use-after-free). A UDF registered via
`createFunction()` closes the database connection *while the function is
executing*, violating SQLite's rule that a running function "must not close the
database connection nor finalize or reset the prepared statement in which the
function is running." Freed memory is then dereferenced on the subsequent callback.

Verbatim reproducer (bug #77977):
```php
<?php
$udf_counter = 0;
$db = new SQLite3("main3.db");
$db->createFunction('my_udf', 'my_udf');
$db->exec("CREATE TABLE IF NOT EXISTS t (f TEXT)");
$results = $db->query('SELECT f, my_udf(f) as m FROM t');
while ($row = $results->fetchArray()) { var_dump($row); }
$db->close();
function my_udf($string) {
    global $db;
    $db->close();          // closes the connection mid-execution -> UAF
    return "x";
}
```
Verbatim crash chain: `php_sqlite3_callback_func()` → `sqlite3_result_text()` →
`sqlite3VdbeMemSetStr()` (freed heap accessed).

**How our repro maps:** row6 is a SEALED-CALLBACK member. NOTE: the defect here is
callback *re-entrancy* (the UDF closing its own connection) rather than a context
freed by an *outside* unregister — a variant of the callback-lifetime family. The
context-UAF is faithful; seal-proper residual as in row1/row2.
