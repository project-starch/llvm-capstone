# Provenance

**Upstream fix:** `gh-143543`, *"Fix re-entrant use-after-free in
itertools.groupby"* (GH-143738), back-patched into the 3.13 branch as GH-144627
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13:
nothing has to be reverted to reach it.

**Consumer:** `Modules/itertoolsmodule.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

`groupby` compares the current key with the target key. Both are borrowed
pointers held in the iterator's own struct:

```c
int rcmp;                       /* pre-fix: compares gbo->tgtkey / gbo->currkey
                                   directly, holding no reference */
```

A user-defined `__eq__` can re-enter the iterator during that comparison. The
fix's own comment states it:

> A user-defined `__eq__` can re-enter groupby and advance the iterator,
> mutating `gbo->tgtkey` / `gbo->currkey` while we are comparing them. Take
> local snapshots and hold strong references so INCREF/DECREF apply to the same
> objects even under re-entrancy.

Advancing the iterator drops the last reference to the key being compared. The
object is freed, its block goes back on its pymalloc pool's free list, and the
comparison continues through the stale pointer.

Upstream's regression test is the shape in Python:

```python
class Key:
    def __eq__(self, other):
        if self.do_advance:
            self.do_advance = False
            next(g)                    # re-enters groupby, frees the key
            return NotImplemented
        return False

g = itertools.groupby([None, None], keys().send)
next(g)
next(g)                                # "must pass with address sanitizer"
```

## Why the memory is invisible

The key is an ordinary small Python object, so `PyObject_Malloc` serves it from
pymalloc: a block inside a pool inside an arena. Freeing it returns the block to
the pool's free list — **no `free()` reaches `malloc`**, so there is no event for
a malloc-level tool.

That is not our inference. CPython documents it in
`Doc/using/configure.rst`, under `--with-address-sanitizer`:

> "To improve ASan detection capabilities you may also want to combine this with
> `--without-pymalloc` to disable the specialized small-object allocator **whose
> allocations are not tracked by ASan**."

Which is also why upstream's test can say "must pass with address sanitizer": it
is meaningful only in a build where pymalloc has been switched off. The corpus's
claim is the same sentence read the other way round — with the allocator on,
nothing sees it.

## Size

The stale pointer is to an object *struct*, not to a payload buffer, so it is far
below the 512-byte threshold and is pymalloc memory on any input. This case
therefore needs no size caveat. The buffer-carrying cases in this corpus — the
decompressors' `next_in`, `bytes.join`, `bytearray` — do, and must say which
input size they assume.

## What a driver would reduce

Real: `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

Reduced: the interpreter. Reaching the real defect needs a running CPython, its
iterator protocol and a user-defined `__eq__`; the port explicitly does not
execute the interpreter in a domain. The driver models the sequence the allocator
sees — allocate a small object, hold a second pointer to it, drop the owning
reference, allocate again into the freed block, read through the stale pointer —
which is what the re-entrant `__eq__` produces.

That makes this a **model-consumer / real-allocator** case, the weaker of the two
tiers used in the PostgreSQL corpus, and it is labelled so in `case.json`.

## Status

Provenance only. No driver is written and nothing has been run.
