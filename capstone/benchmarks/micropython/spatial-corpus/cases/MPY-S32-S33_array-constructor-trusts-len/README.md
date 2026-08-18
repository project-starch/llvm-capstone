# MPY-S32 / MPY-S33: the array constructors trust a user `__len__`

Two open upstream issues in the same function pair, measured together because they
share `py/objarray.c:array_construct` and differ only in how the trusted length goes
wrong.

- **#18617** — `__len__` reports 1, `__iter__` yields 1000. The buffer is sized from
  the first and filled from the second, and nothing re-checks the bound.
- **#18620** — `__len__` returns `1 << 61`, so `typecode_size * len` wraps to zero:
  the allocation is empty while the logical length stays huge.

Both are OPEN, so both are present at the pinned tree and need no parent build. Both
SIGSEGV on the host at the pin, verified before anything was built for the domain.

Three scripts, one image, one boot, ordered so the arm expected to return goes
first. `01` is the discriminator: without it a faulting run cannot say whether the
hardware stopped the write or something downstream died of it. See RESULT.txt.
