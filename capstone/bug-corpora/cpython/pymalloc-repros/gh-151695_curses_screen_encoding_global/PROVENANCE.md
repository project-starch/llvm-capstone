# Provenance

**Upstream fix:** `gh-151695`, *"use-after-free of the curses screen encoding"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_cursesmodule.c`. **CVE:** `NO VERIFIED CVE`.

## The defect

**What is freed:** curses_screen_encoding, a module-level pointer into the window object's encoding string.
**What ends its life:** the window object initscr() returned is deallocated while module-level functions keep reading through the static.

```c
-static const char *curses_screen_encoding = NULL;
+/* ... This is a private copy: the window object that initscr() returns
+   may be deallocated while these functions are still in use. */
+static char *curses_screen_encoding = NULL;
```

**What distinguishes it:** the dangling pointer outlives every frame. Case 14's lived in another object; this one lives in a global, so nothing in the program's structure bounds when it is next read.

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

`shared/defects.c`, case **18**. Shape: dangling pointer parked in a global.
