# PROVENANCE — row13 (CPython #149738 deleted row_factory null-deref)

- Upstream : https://github.com/python/cpython/issues/149738
- Fix PR   : https://github.com/python/cpython/pull/149754 (backports 3.13/3.14/3.15)
- Tier     : **LITERAL-traceable bug, but NOT literal-convertible** (host-language event)

**Real memory-safety bug: YES** (null-pointer dereference / segfault). `del
db.row_factory` sets the attribute to NULL; later code checks it against `Py_None`
and calls `Py_INCREF(NULL)` → crash.

Verbatim reproducer (issue #149738):
```python
import sqlite3
db = sqlite3.connect(":memory:")
del db.row_factory
db.execute("test")            # segfault
```

**Why row13 stays a probe (per `design/sqlite-corpus-fidelity-ceiling.md`):** the
lifecycle event that "revokes" the callback is a **Python attribute deletion — no
SQLite C-API call**. There is no SQLite pointer-lifetime event for a capability to
hang a fault on, so a LITERAL real-SQLite matched pair is not meaningful here;
imposing one would be *less* faithful. The bug is real; it is simply outside the
layer the hardware capability model protects. BORROW-REVOKE probe is the honest
ceiling.
