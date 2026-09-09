# row13_cpython_deleted_rowfactory

Upstream: https://github.com/python/cpython/issues/149738
Table 3 row: 13
Class: null-deref
Essence: Host code deletes a row-factory callback but later invokes the now-NULL callback without checking it.
Observed: ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000

Stage-2 "after" (Capstone): borrow-revoke of the row_factory callback capability
(`tests/runtime-qemu/sqlite-borrow-revoke-probe/` shape), with the revoke fired
when the host deletes the attribute. A later invocation through the revoked
callback capability faults deterministically instead of a NULL-pointer deref.
