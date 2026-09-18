# Provenance

**Upstream fix:** `gh-144833`, *"use-after-free in the SSL module when SSL_new() fails"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_ssl.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** the PySSLSocket itself.
**What ends its life:** Py_DECREF(self) on the error path, before self is read.

```c
+        _setSSLError(get_state_ctx(sslctx), NULL, 0, __FILE__, __LINE__);
         Py_DECREF(self);
-        _setSSLError(get_state_ctx(self), NULL, 0, __FILE__, __LINE__);
```

**What distinguishes it:** the stale access is to the freed object ITSELF rather than to anything it pointed at, and no second party is involved at all. The fix is a two-line swap.

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

`shared/defects.c`, case **13**. Shape: interior pointer into the object that was just released.
