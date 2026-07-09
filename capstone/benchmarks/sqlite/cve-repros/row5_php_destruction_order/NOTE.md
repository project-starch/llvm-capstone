# row5 — fidelity note (essence vs binding-faithful)

Two "before" variants are kept side by side so the paper can choose the fidelity
it wants per row:

| File | Level | What it models |
|---|---|---|
| `before.c` | **Essence** | A flat `struct owner{db;stmt}`; close the db, free the owner, finalize a dangling stmt. Minimal; maps loosely to the advisory. |
| `before-faithful.c` | **Binding-faithful** | PHP `ext/sqlite3`'s real object model: `php_sqlite3_db_object` with a `zend_llist free_list` of statements, and `php_sqlite3_stmt_object` with a `db_obj` back-pointer. The statement destructor (`php_sqlite3_compare_stmt_free`) walks the parent's `free_list` **after** the parent was freed. |

## Why the faithful version matters here

Row5 is a HIERARCHICAL-REVOKE shape, and the bug *is* the ownership relationship:
the connection object owns its statements (`free_list`), and the statement holds a
back-reference to the connection. The defect is the teardown order between those
two host wrapper objects — precisely what the paper's senior/hierarchical
revocation is meant to make safe (a revoke on the connection invalidates the
statement's handle and the back-reference, so the statement destructor hits a
revoked capability locally instead of freed memory). The essence version can't
carry that argument because it has no ownership tree to revoke.

## Faithful to / approximated

- **Faithful:** the two glue structs and field names (`php_sqlite3_structs.h`),
  the statement→connection back-pointer, the `free_list` tracking relationship,
  the real crash function name, and the exact wrong destruction order (connection
  freed before statement).
- **Approximated:** `zend_llist` is reduced to a minimal linked list; the full
  Zend object/GC machinery is out of scope and irrelevant to the lifetime bug.
- **Self-contained:** links the real SQLite amalgamation; no Zend/PHP runtime.

## Verified (host)

```
gcc -std=c11 -g -O0 -fsanitize=address -I<sqlite-src> \
    before-faithful.c <sqlite-src>/sqlite3.c -o row5-faithful -lpthread -ldl -lm
./row5-faithful
# => AddressSanitizer: heap-use-after-free (READ) in php_sqlite3_compare_stmt_free
#    freed by php_sqlite3_object_free_storage  (SQLite 3.53.3)
```

Oracle unchanged: `heap-use-after-free`.

## Sources

- Bug: https://bugs.php.net/bug.php?id=69971 (dev note: "php_sqlite3_db_object is
  destroyed before php_sqlite3_stmt ... stmt tries to access
  intern->db_obj->free_list, which is in memory already freed").
- Structs: `php/php-src` `ext/sqlite3/php_sqlite3_structs.h`.
