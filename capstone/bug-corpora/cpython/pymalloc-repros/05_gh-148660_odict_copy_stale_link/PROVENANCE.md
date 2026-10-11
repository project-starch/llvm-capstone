# Provenance

**Upstream fix:** `gh-148660`, *"Fix use-after-free in OrderedDict.copy() on
reentrant mutation"* (GH-151573), back-patched into the 3.13 branch **after**
our `v3.13.7` tag. The backport is the proof the defect was in 3.13.

**Consumer:** `Objects/odictobject.c`, `OrderedDict_copy_impl`.
**CVE:** `NO VERIFIED CVE`.

## The defect

`OrderedDict.copy()` walks the linked list of nodes with `_odict_FOREACH` and,
for each, reads the key and value out of the node it is standing on:

```c
     if (PyODict_CheckExact(od)) {
         _odict_FOREACH(od, node) {
-            PyObject *key = _odictnode_KEY(node);
-            PyObject *value = _odictnode_VALUE(node, od);
+            PyObject *key = Py_NewRef(_odictnode_KEY(node));
+            Py_hash_t hash = _odictnode_HASH(node);
+            PyObject *value = PyODict_GetItemWithError(od, key);
...
+            if (_PyODictObject_CAST(od)->od_state != state)
+                goto mutated;
```

Copying into the destination can run arbitrary Python — a subclass's
`__setitem__`, a key's `__hash__` — which can mutate the source and free the
node the loop is standing on. The fix takes a strong reference to the key **and**
adds an `od_state` check, because holding the key does not keep the *node*
alive.

**What distinguishes it, and why it is the most interesting case here:**
`_odict_FOREACH` advances by reading `node->next` **out of the node it has just
processed**. So when the node is freed, the loop does not merely read stale
data — it reads a **pointer** from a released block and then follows it. After
the block is reused by another node, the link reads back as a perfectly valid
pointer and the walk continues, silently visiting the wrong node and copying the
wrong entry. Nothing crashes; the answer is just wrong.

That is the strongest single argument in this corpus for revocation over
detection: the spatial arm here does not merely fail to notice, it produces a
plausible result.

## Why the memory is invisible

An odict node is a small block from a pymalloc pool. The free returns it to the
pool's free list and no `free()` reaches `malloc`. See `../README.md`.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter and the odict. The driver builds two nodes linked
together, frees the first, allocates a replacement node into its block, and then
reads the link field through the stale cursor and follows it — asserting in the
spatial arm that the follow lands on the still-live second node, which is the
"plausible wrong answer" the defect produces in place.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **5**. Shape: pointer load out of a freed block.
