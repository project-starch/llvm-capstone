# Provenance

**Upstream fix:** `gh-146613`, *"Fix re-entrant use-after-free in
`itertools._grouper`"* (GH-147962), back-patched into the 3.13 branch **after**
our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/itertoolsmodule.c`, `_grouper_next`.
**CVE:** `NO VERIFIED CVE`.

**Sibling issue:** this is the child-iterator half of
`../gh-143543_groupby_reentrant_eq/`, which is the parent's. They were reported
and fixed separately. If you arrived here looking for the `groupby` object
itself, read that one.

## The defect

`_grouper_next` compares the grouper's own target key with the parent
`groupby`'s current key. Both are borrowed:

```c
assert(gbo->currkey != NULL);
-    rcmp = PyObject_RichCompareBool(igo->tgtkey, gbo->currkey, Py_EQ);
+    /* A user-defined __eq__ can re-enter the grouper and advance the iterator,
+       mutating gbo->currkey while we are comparing them.
+       Take local snapshots and hold strong references so INCREF/DECREF
+       apply to the same objects even under re-entrancy. */
+    PyObject *tgtkey = Py_NewRef(igo->tgtkey);
+    PyObject *currkey = Py_NewRef(gbo->currkey);
+    rcmp = PyObject_RichCompareBool(tgtkey, currkey, Py_EQ);
```

The comparison runs a user `__eq__`, which re-enters and advances the parent.
Advancing replaces `gbo->currkey`, dropping the last reference to the object the
comparison is still holding a borrowed pointer to.

**What distinguishes it from `gh-143543`:** two keys are live at once, in two
different objects, and only one of them dies. The surviving alias is what makes
the pair worth keeping apart — the driver checks the survivor is intact after
the free, so a fault on the stale one cannot be confused with the allocator
having damaged everything.

## Why the memory is invisible

An ordinary small Python object: `PyObject_Malloc` serves it from a pymalloc
pool, and freeing it returns the block to that pool's free list without any
`free()` reaching `malloc`. See `../README.md` and
`Doc/using/configure.rst`'s note that `--with-address-sanitizer` should be
combined with `--without-pymalloc` because pymalloc's allocations "are not
tracked by ASan".

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation, free and
reallocation calls.

**Reduced:** the interpreter. Reaching the defect in place needs a running
CPython, the iterator protocol and a user-defined `__eq__`; the port does not
execute the interpreter in a domain. The driver performs the sequence the
allocator sees — two live keys, free one, allocate into the block it released,
read through the kept pointer, with the survivor checked — which is what the
re-entrant advance produces.

**model-consumer / real-allocator**, the weaker of the two fidelity tiers, and
labelled so in `case.json`.

## Case in the driver

`shared/defects.c`, case **1**. Shape: free / reuse / stale read.
