# PROVENANCE — row1 (CPython #142830 progress-handler UAF)

- Upstream : https://github.com/python/cpython/issues/142830
- Fix PR   : https://github.com/python/cpython/pull/143245 (backports #143322, #143323)
- Tier     : **LITERAL-traceable** (ASan-confirmed use-after-free)

**Real memory-safety bug: YES.** A progress handler returning an object with a
custom `__bool__` re-enters and calls `set_progress_handler(None, 1)` *during the
callback*, freeing the callback context; when SQLite aborts the query after the
exception, `print_or_clear_traceback(ctx)` reads the freed context → heap UAF.
AddressSanitizer confirms heap-use-after-free at `print_or_clear_traceback()` on
memory from `create_callback_context()`.

Verbatim reproducer (issue #142830):
```python
import sqlite3
conn = sqlite3.connect(":memory:")
class Evil:
    def __bool__(self):
        conn.set_progress_handler(None, 1)   # frees callback context
        raise RuntimeError                    # force PyObject_IsTrue to fail
conn.set_progress_handler(lambda: Evil(), 1)
conn.execute("select 1")
```

**How our repro maps:** row1 is a SEALED-CALLBACK member — the callback context is
revoked (unregister) and a later access faults. Faithful to the context-UAF; the
seal proper (domain crossing) is the residual tracked in
`design/sqlite-corpus-fidelity-ceiling.md` (Agent-B task-011, on hold).
