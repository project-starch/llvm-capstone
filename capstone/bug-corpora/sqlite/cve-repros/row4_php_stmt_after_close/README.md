# row4_php_stmt_after_close

Upstream: https://bugs.php.net/bug.php?id=66550
Table 3 row: 4
Class: use-after-close
Essence: A host statement wrapper is destroyed while closing its connection, then its statement field is read to perform another step.
Observed: ERROR: AddressSanitizer: heap-use-after-free
