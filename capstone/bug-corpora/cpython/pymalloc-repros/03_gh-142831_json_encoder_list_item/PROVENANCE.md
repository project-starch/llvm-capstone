# Provenance

**Upstream fix:** `gh-142831`, *"Fix use-after-free in json encoder during
re-entrant mutation"* (gh-142851), back-patched into the 3.13 branch **after**
our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Modules/_json.c`, `encoder_listencode_dict` and
`encoder_listencode_list`. **CVE:** `NO VERIFIED CVE`.

**Sibling issue:** `../gh-145244_json_encoder_dict_key/` is a second, separately
reported use-after-free in the same file, on the `PyDict_Next` path and on the
error path. Read that one if your symptom is a crash while an exception is being
formatted.

## The defect

Every item the encoder walks is borrowed, and encoding an item can run arbitrary
Python — a `default` callback, a `__str__`, a subclass's `keys()`:

```c
         for (Py_ssize_t  i = 0; i < PyList_GET_SIZE(items); i++) {
             PyObject *item = PyList_GET_ITEM(items, i);
+            // gh-142831: encoder_encode_key_value() can invoke user code
+            // that mutates the items list, invalidating this borrowed ref.
+            Py_INCREF(item);
```

The fix adds the same protection in three places: the items list, the
`PyDict_Next` loop, and `encoder_listencode_list`'s sequence walk.

**What distinguishes it:** the stale pointer is not a local the compiler could
have kept in a register — it is reached through `ob_item[i]` of a list that is
**still alive**. The container survives; one of the things it points at does
not. A checker that tracked only objects reachable from live containers would
call this reachable.

## Why the memory is invisible

The freed item is an ordinary small Python object from a pymalloc pool; the free
returns its block to the pool's free list and no `free()` reaches `malloc`. See
`../README.md`.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter and the JSON encoder. The driver allocates the
list's storage array, allocates an item, stores the item into the array, frees
the item, allocates into the released block, and then reads through the pointer
**still sitting in the live array** — the sequence the re-entrant mutation
produces.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **3**. Shape: stale entry in a live array.
