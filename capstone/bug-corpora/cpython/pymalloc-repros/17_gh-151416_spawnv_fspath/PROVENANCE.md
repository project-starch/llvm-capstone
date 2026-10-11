# Provenance

**Upstream fix:** `gh-151416`, *"borrowed ref use after free via fspath in os.spawnv/spawnve"*, back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/posixmodule.c`. **CVE:** `NO VERIFIED CVE`.

**Sibling issue:** `../gh-151403_fork_exec_fspath/` is the same trigger in another module, reported and fixed separately.

## The defect

**What is freed:** the item returned by the getitem function pointer over argv.
**What ends its life:** the item's __fspath__ mutates the list, releasing its reference.

```c
+        // The item must be a strong reference because of possible
+        // side-effects of PyUnicode_FS{Converter,Decoder}() in
+        // fsconvert_strdup(): an item's __fspath__() can mutate a list
+        // *argv*, releasing the list's reference to the item (gh-151416).
```

**What distinguishes it:** sibling of gh-151403, fixed months apart. Keeping both is the corpus's point about how narrowly each instance gets patched.

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

`shared/defects.c`, case **17**. Shape: free / reuse / stale read.
