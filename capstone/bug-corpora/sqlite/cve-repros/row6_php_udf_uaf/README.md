# row6_php_udf_uaf

Upstream: https://bugs.php.net/bug.php?id=77977
Table 3 row: 6
Class: UAF
Essence: A user-defined function remains registered after its backing state is freed and dereferences that state when invoked.
Observed: ERROR: AddressSanitizer: heap-use-after-free
