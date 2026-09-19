# Provenance

**Upstream fix:** `gh-151403`, *"use-after-free when an argv item's __fspath__ mutates args"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_posixsubprocess.c`. **CVE:** `NO VERIFIED CVE`.

**Sibling issue:** `../gh-151416_spawnv_fspath/` is the same trigger in another module, reported and fixed separately.

## The defect

**What is freed:** borrowed_arg, from PySequence_Fast_GET_ITEM(fast_args, arg_num).
**What ends its life:** the item's __fspath__ runs Python that drops fast_args' last reference to it.

```c
+            /* borrowed_arg is only borrowed; its __fspath__() may run Python
+               that drops fast_args' last reference to it. */
+            Py_INCREF(borrowed_arg);
```

**What distinguishes it:** sibling of gh-151416, the same trigger one module over, reported and fixed separately.

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

`shared/defects.c`, case **16**. Shape: free / reuse / stale read.
