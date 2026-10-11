# Provenance

**Upstream fix:** `gh-146011`, *"use-after-free in signaldict_repr after deletion"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_decimal/_decimal.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** traps->flags, a borrowed pointer INTO the context object's storage.
**What ends its life:** context_clear releases the context without clearing the signal dict's interior pointer, and the signal dict survives it.

```c
+    /* Since traps and flags hold a borrowed reference to the
+       flags stored in the context object, these references need
+       to be cleared when the context object is deallocated
+       because traps and flags can survive. See gh-146011. */
```

**What distinguishes it:** the gap is unbounded. Every other case's stale access happens inside the operation that created it or on the next API call; this one waits for a repr() that may never come.

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

`shared/defects.c`, case **14**. Shape: dangling pointer parked in a surviving object.
