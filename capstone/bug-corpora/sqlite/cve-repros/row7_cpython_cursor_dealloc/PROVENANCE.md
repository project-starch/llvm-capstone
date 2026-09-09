# PROVENANCE — row7 (CPython #99886 cursor-subclass dealloc crash)

- Upstream  : https://github.com/python/cpython/issues/99886
- Fix PR    : https://github.com/python/cpython/pull/99902
  ("gh-99886: Fix crash when freeing objects with managed dictionaries")
- Root cause / fix commits: a8b9350 / de388c0
- Tier      : **MODEL-mapping** — real crash, but NOT a statement-lifecycle UAF.

**Honest finding (surfaced by this provenance sweep):** the actual upstream bug is
a CPython **managed-dictionary teardown crash**, not a hierarchical connection →
statement use-after-free. The crash is in `_PyObject_ClearInstanceAttributes()`
from `subtype_clear()` during deallocation of a `sqlite3.Cursor` *subclass that
sets a custom attribute*, dereferencing freed-memory poison `0xdddddddddddddddd`.
It reproduces on Python 3.11.0 only (not 3.10, not main); adding
`__slots__ = ('some_attr',)` avoids it — i.e. it is about managed `__dict__`
clearing, and the `sqlite3.Cursor` subclass is merely the trigger object.

Verbatim reproducer (issue #99886):
```python
import sqlite3
class MyCursor(sqlite3.Cursor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.some_attr = None
conn = sqlite3.connect(':memory:')
cur = conn.cursor(MyCursor)
cur.close()
del cur
print('never get here with -X dev')
```

**Consequence for the corpus.** row7's *mechanism* repro (task-010 literal H) is a
real-SQLite connection→statement hierarchy fault and stands on its own as a
demonstration of the hierarchical primitive. But its **provenance to a statement-
lifecycle UAF is INTERPRETIVE**, not a faithful lowering of #99886. The genuinely
faithful HIERARCHICAL bugs in the corpus are **row5 (PHP #69971)** and **row9
(sqlite3-ruby #49)** — both real freed-parent → child-deref UAFs. Prefer row5/row9
as the H family's faithful anchors (Agent-B task-012 makes them literal); treat
row7 as a mechanism demonstration, not a CVE lowering.
