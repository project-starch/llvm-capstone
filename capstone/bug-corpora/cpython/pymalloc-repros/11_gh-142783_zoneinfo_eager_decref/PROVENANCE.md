# Provenance

**Upstream fix:** `gh-142783`, *"possible use after free in the zoneinfo module"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_zoneinfo.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** the weak cache returned by PyObject_GetAttrString.
**What ends its life:** Py_XDECREF on the line after the lookup, on the assumption that the type held a reference.

```c
-        // We are assuming that the type lives at least as long as the function
-        // that calls get_weak_cache, and that it holds a reference to the
-        // cache, so we'll return a "borrowed reference".
-        Py_XDECREF(cache);
-        return cache;
+        return PyObject_GetAttrString((PyObject *)type, "_weak_cache");
```

**What distinguishes it:** no re-entrancy and no user callback. Every other case here needs something to run between the free and the use; this one has the two on consecutive lines, with a comment explaining why it is safe.

## Why the memory is invisible

The freed block comes from a pymalloc pool, and the free returns it to that
pool's free list without any `free()` reaching `malloc`. Upstream states the
consequence itself in `Doc/using/configure.rst`: to use AddressSanitizer you
should combine it with `--without-pymalloc`, "to disable the specialized
small-object allocator whose allocations are not tracked by ASan". See
`../README.md`.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation, free and
reallocation calls.

**Reduced:** the interpreter and the module. Reaching the defect in place needs
a running CPython, which the port does not put in a domain. The driver performs
the sequence the allocator sees, in the same order.

**model-consumer / real-allocator**, the weaker of the two fidelity tiers.

## Case in the driver

`shared/defects.c`, case **11**. Shape: free and use on adjacent lines.
