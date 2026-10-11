# Provenance

**Upstream fix:** `gh-142829`, *"Fix use-after-free in `Context.__eq__` via
re-entrant `ContextVar.set`"* (GH-142905), back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Python/hamt.c`, `_PyHamt_Eq`. **CVE:** `NO VERIFIED CVE`.

## The defect

`_PyHamt_Eq` walks one hash-array-mapped trie while looking keys up in the
other. Nothing along that walk holds a reference — not the two maps, not the
key, not either value:

```c
+    Py_INCREF(v);
+    Py_INCREF(w);
...
                 case F_FOUND: {
+                    Py_INCREF(v_key);
+                    Py_INCREF(v_val);
+                    Py_INCREF(w_val);
                     int cmp = PyObject_RichCompareBool(v_val, w_val, Py_EQ);
+                    Py_DECREF(v_key);
+                    Py_DECREF(v_val);
+                    Py_DECREF(w_val);
```

`PyObject_RichCompareBool` runs a user `__eq__`, which can call
`ContextVar.set`. That rebuilds the context's map and can drop the last
reference to the node the walk is standing in.

**What distinguishes it:** `PyHamtIteratorState` holds a pointer *into* the
node — `i_nodes[level]` plus a position — not to its start. So the stale access
is through an **interior pointer** into a freed block, which is the case that
separates a per-block revocation from anything that only validates the base
address of an allocation.

## A note on which allocator layer this belongs to

The freed things are of two kinds, and only one is covered by the port. The
*containers* `v` and `w` are `Context` objects, and contexts have a per-type
free list (`pycore_freelist.h`), so freeing one can stop above pymalloc. The
*keys and values* are arbitrary user objects and are ordinary pymalloc blocks.
The driver models the key/value path, which is the one the port can see. This
was read from the source rather than inferred from the file the fix touches;
`../../../../docs/ref/cpython-pymalloc-defects.md` records that file-based layer
assignment is a proxy and a lower bound.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter, the trie, and `contextvars`. The driver allocates
a block, takes an interior pointer 16 bytes in, frees the block, allocates into
it again, and reads through the interior pointer — the sequence the walk
produces when the re-entrant `set` drops the node.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **2**. Shape: interior pointer into a freed block.
