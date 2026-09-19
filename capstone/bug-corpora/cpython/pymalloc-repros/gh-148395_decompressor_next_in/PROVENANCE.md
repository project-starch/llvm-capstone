# Provenance

**Upstream fix:** `gh-148395`, *"Fix a possible UAF in
`{LZMA,BZ2,_Zlib}Decompressor`"* (GH-148396), back-patched into the 3.13 branch
**after** our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_bz2module.c`, `Modules/_lzmamodule.c`,
`Modules/zlibmodule.c` — the same defect, three times.
**CVE:** `NO VERIFIED CVE`.

## The defect

Each decompressor keeps a stream struct whose `next_in` points into the caller's
input buffer while a `decompress()` call is running. On the error path the
struct was left holding that pointer:

```c
 error:
+    bzs->next_in = NULL;
     Py_XDECREF(result);
     return NULL;
```

and identically for `lzs->next_in` and `self->zst.next_in`. The caller then
releases the input buffer, and the **next** `decompress()` resumes from
`next_in`.

**What distinguishes it:** the stale pointer lives in a **long-lived struct
field across two API calls**, not inside a single operation. Everything else in
this corpus goes wrong within one call, which means a checker with a
call-scoped view could in principle catch them. This one is dormant between
calls: the object is consistent, the pointer is stale, and nothing touches it
until the next entry point. It is also the smallest fix in the corpus — three
one-line assignments — which is a fair illustration of how little the defect
looks like anything.

## Why the memory is invisible

The released input buffer is pymalloc memory when it is small: `obmalloc.c` sets
`PYMEM_DOMAIN_MEM` to `PYMALLOC_ALLOC`, so `PyMem_Malloc` reaches pymalloc too,
not only `PyObject_Malloc`. The free returns the block to a pool free list and
no `free()` reaches `malloc`. See `../README.md`, and see
`../gh-151295_bytes_join_reentrant_buffer/PROVENANCE.md` for what happens above
the 512-byte threshold — the same caveat applies in principle here, and the
driver pins the small size for the same reason.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter and the compression libraries. The driver allocates
a stream struct and an input buffer, stores the buffer pointer in the struct's
`next_in` field, frees the buffer, allocates into the released block, and then
reads through `next_in` — the sequence the second `decompress()` produces.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **7**. Shape: cursor surviving in a struct field across
two API calls.
