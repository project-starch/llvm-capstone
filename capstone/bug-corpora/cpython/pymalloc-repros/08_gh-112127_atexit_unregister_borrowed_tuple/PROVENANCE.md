# Provenance

**Upstream fix:** `gh-112127`, *"possible use-after-free in atexit.unregister()"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/atexitmodule.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** a callback tuple borrowed out of the live callbacks list.
**What ends its life:** a user __eq__ calls atexit.unregister again and mutates the list.

```c
PyObject *tuple = Py_NewRef(PyList_GET_ITEM(callbacks, i));
```

**What distinguishes it:** the loop carries on to the next index through the same list after the free, so the container outlives the entry.

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

`shared/defects.c`, case **8**. Shape: free / reuse / stale read.
