# Provenance

**Upstream fix:** `gh-153539`, *"use-after-free in TextIOWrapper.tell() with a reentrant decoder"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_io/textio.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** next_input, the borrowed snapshot bytes object.
**What ends its life:** the decoder's getstate runs Python that seeks the file and replaces the snapshot.

```c
+    /* Own next_input: a reentrant or concurrent seek can drop the snapshot. */
+    Py_INCREF(next_input);
```

**What distinguishes it:** the freed object is a bytes snapshot of buffered input.

**SIZE.** A large buffer puts the snapshot above the 512-byte threshold, where the same defect is an ordinary malloc use-after-free that ASan sees. The driver pins the small size.

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

`shared/defects.c`, case **19**. Shape: free / reuse / stale read.
