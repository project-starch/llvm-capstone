# Provenance

**Upstream fix:** `gh-149449`, *"use-after-free in _PyUnicode_GetNameCAPI"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/unicodedata.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** the _PyUnicode_Name_CAPI struct, a bare PyMem_Malloc block.
**What ends its life:** the capsule's destructor frees it when unicodedata leaves sys.modules, while other code has cached the pointer.

```c
+    // Statically allocated so that any cached pointers stay valid after unicodedata
+    // is removed from sys.modules and the capsule is gc'd (gh-149449).
+    static _PyUnicode_Name_CAPI capi = { ... };
```

**What distinguishes it:** the freed thing is not a PyObject at all. It is pymalloc's because obmalloc sets PYMEM_DOMAIN_MEM to PYMALLOC_ALLOC, so PyMem_Malloc reaches the same pools PyObject_Malloc does -- this case exists to exercise that path.

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

`shared/defects.c`, case **15**. Shape: bare PyMem block, cached by a third party.
