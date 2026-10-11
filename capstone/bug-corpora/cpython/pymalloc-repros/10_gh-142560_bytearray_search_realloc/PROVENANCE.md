# Provenance

**Upstream fix:** `gh-142560`, *"use-after-free in bytearray search-like methods"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Objects/bytearrayobject.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** the bytearray's storage, cached as PyByteArray_AS_STRING(self).
**What ends its life:** user code resizes the bytearray, and the REALLOC moves the block.

```c
+    /* Increase exports to prevent bytearray storage from changing during op. */
+    self->ob_exports++;
+    res = op(PyByteArray_AS_STRING(self), Py_SIZE(self), sub, start, end);
+    self->ob_exports--;
```

**What distinguishes it:** the only case in the corpus whose block is ended by a realloc rather than a free. The driver ASSERTS the realloc moved the block; a realloc that returned the same address would leave it testing nothing.

**SIZE.** The storage is pymalloc's only while the bytearray is small; a large one falls through to malloc, where ASan sees the same defect. The driver pins 48 bytes growing to 300, both below the threshold.

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

`shared/defects.c`, case **10**. Shape: realloc moved the block.
