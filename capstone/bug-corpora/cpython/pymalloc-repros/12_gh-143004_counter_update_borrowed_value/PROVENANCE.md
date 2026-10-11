# Provenance

**Upstream fix:** `gh-143004`, *"possible use-after-free in collections.Counter.update()"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_collectionsmodule.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** oldval, borrowed from the mapping.
**What ends its life:** a user __add__ invoked by PyNumber_Add mutates or clears the dict.

```c
+                /* oldval is a borrowed reference.  Keep it alive across
+                   PyNumber_Add(), which can execute arbitrary user code and
+                   mutate (or even clear) the underlying dict. */
+                Py_INCREF(oldval);
```

**What distinguishes it:** the container is emptied and KEPT, where case 4's is emptied and abandoned. The stale pointer is a value rather than a key.

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

`shared/defects.c`, case **12**. Shape: free / reuse / stale read.
