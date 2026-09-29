# PROVENANCE — row14 (CPython uninitialised connection)

- Upstream / fix : https://github.com/python/cpython/pull/27472
- Tier           : **LITERAL-traceable**

**Real memory-safety bug: YES.** A `sqlite3.Connection` allocated via `__new__()`
without `__init__()` has an uninitialised (NULL) `sqlite3 *db`; operating on it
segfaulted.

Verbatim fix guard added to `Modules/_sqlite/connection.c`:
```c
if (!self->initialized) {
    PyErr_SetString(pysqlite_ProgrammingError,
                    "Base Connection.__init__ not called.");
    return NULL;
}
```
Verbatim regression test added by the fix:
```python
class UninitialisedConnectionTests(unittest.TestCase):
    def setUp(self):
        self.cx = sqlite.Connection.__new__(sqlite.Connection)
    def test_uninit_operations(self):
        funcs = (lambda: self.cx.isolation_level,
                 lambda: self.cx.total_changes,
                 lambda: self.cx.in_transaction,
                 lambda: self.cx.iterdump(),
                 lambda: self.cx.cursor(),
                 lambda: self.cx.close())
        for func in funcs:
            with self.subTest(func=func):
                self.assertRaisesRegex(sqlite.ProgrammingError,
                    "Base Connection.__init__ not called", func)
```

**How our repro maps:** row14 (literal U representative, task-009/010) models the
connection as a genuine UNINIT capability; a pre-`csinit`/pre-open read faults
(cause 26), and initialisation makes it usable — the same "used before init"
defect. RESIDUAL (documented in `design/sqlite-corpus-fidelity-ceiling.md`): real
`sqlite3_open` allocates fresh rather than initialising a caller region, so the
UNINIT `db` is minted; a fully-literal U needs an owner-reload UNINIT refinement.
