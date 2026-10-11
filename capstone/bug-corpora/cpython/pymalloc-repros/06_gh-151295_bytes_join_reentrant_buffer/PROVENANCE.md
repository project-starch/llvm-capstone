# Provenance

**Upstream fix:** `gh-151295`, *"Fix use-after-free in
`bytes.join()`/`bytearray.join()` via re-entrant `__buffer__`"* (GH-151296),
back-patched into the 3.13 branch **after** our `v3.13.7` tag. The backport is
the proof the defect was in 3.13.

**Consumer:** `Objects/stringlib/join.h`, `STRINGLIB(bytes_join)`.
**CVE:** `NO VERIFIED CVE`.

## The defect

`bytes_join` fills an array of `Py_buffer` from the sequence's items and then
copies out of them. Acquiring a buffer runs the item's `__buffer__`, which is
user code:

```c
         else {
+            /* item is only borrowed; its __buffer__() may run Python that
+               drops the sequence's last reference to it. */
+            Py_INCREF(item);
             if (PyObject_GetBuffer(item, &buffers[i], PyBUF_SIMPLE) != 0) {
+                Py_DECREF(item);
```

`item` is borrowed from the sequence. Its `__buffer__` can mutate the sequence
and drop its last reference, freeing the object — while `buffers[i].buf` already
points **into its payload**. The join then memcpys from released memory.

**What distinguishes it:** the stale pointer is into a **payload buffer**, not an
object header, and that is what makes this the one case in the corpus where size
decides whether the defect is visible at all.

## The 512-byte caveat — this case only

`pymalloc` serves requests up to 512 bytes; above that the allocation falls
through to `malloc`. So:

* **payload ≤ 512 bytes** — the buffer is a pymalloc block, the free returns it
  to a pool free list, no `free()` reaches `malloc`, and a malloc-level tool has
  no event to see. This is what the driver exercises, at 48 bytes.
* **payload > 512 bytes** — the same defect, in the same source line, is an
  ordinary heap use-after-free and **ASan reports it**.

One defect, two outcomes, decided by the input. The driver pins the small size
and says so here, because a specimen that quietly grew would stop demonstrating
anything while still passing. The other seven cases hold pointers to object
structs, which are far below the threshold on any input and need no caveat.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter, the buffer protocol and the sequence. The driver
allocates a payload, takes an interior pointer into it (`buffers[i].buf`), frees
the payload, allocates into the released block, and reads through the interior
pointer — the sequence the re-entrant `__buffer__` produces.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **6**. Shape: payload buffer, pinned below 512 bytes.
