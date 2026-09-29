# row10_ruby_stmt_reuse_after_close

Upstream: https://groups.google.com/g/sqlite3-ruby/c/SGRQE_2MZ8I
Table 3 row: 10
Class: use-after-close
Essence: Statement reuse reads a statement handle from a binding wrapper that connection teardown already freed.
Observed: ERROR: AddressSanitizer: heap-use-after-free
