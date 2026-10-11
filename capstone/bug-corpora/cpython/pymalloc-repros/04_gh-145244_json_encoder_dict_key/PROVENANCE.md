# Provenance

**Upstream fix:** `gh-145244`, *"Fix use-after-free on borrowed dict key in json
encoder"* (GH-145245), fixed on `main` and **never back-ported to 3.13**. That
means the backport cannot be the proof, so this case is live for a different
reason, stated below.

**Consumer:** `Modules/_json.c`. **CVE:** `NO VERIFIED CVE`.

**Sibling issue:** `../gh-142831_json_encoder_list_item/` is the other
use-after-free in this file, reported separately and back-ported.

## Why it is live at `v3.13.7` — read, not inferred

The apply test used for the never-back-ported group says nothing here: the fix
does not apply to `v3.13.7`, because upstream renamed the function to
`_encoder_iterate_dict_lock_held` and changed its signature. **A fix that fails
to apply is not evidence the defect is gone**, so the pinned source was read
instead.

`Modules/_json.c:1621` of `v3.13.7`:

```c
    } else {
        Py_ssize_t pos = 0;
        while (PyDict_Next(dct, &pos, &key, &value)) {
            if (encoder_encode_key_value(s, writer, &first, key, value,
                                         new_newline_indent,
                                         current_item_separator) < 0)
                goto bail;
        }
    }
```

`key` and `value` are borrowed from `PyDict_Next` and handed straight to
`encoder_encode_key_value` with **no `Py_INCREF` at all** — not even the
free-threading-only pair that the pre-fix `main` had. The pinned tree is if
anything more exposed than the code the fix was written against.

## The defect

The upstream fix's own description: `key` is a borrowed reference from
`PyDict_Next`; if the `default` callback mutates or clears the dict, `key`
becomes dangling, and the error path then calls `_PyErr_FormatNote("%R", key)`
on freed memory.

**What distinguishes it:** two things. The free is a **bulk** one — clearing the
dict releases every entry at once, so the stale pointer is one of many blocks
returned together. And the stale access is on the **error path**, while an
exception is already being formatted, which is where a fault is least expected
and least likely to be reached by a test.

## Why the memory is invisible

Small Python objects from pymalloc pools. See `../README.md`.

## What a driver reduces

**Real:** `obmalloc.c` from the pinned 3.13.7, and the allocation and free calls.

**Reduced:** the interpreter, the dict and the encoder. The driver allocates
four entries, keeps a pointer to one, frees all four in a single sweep
(`PyDict_Clear` in the callback), allocates into the released space, and reads
through the kept pointer.

**model-consumer / real-allocator.**

## Case in the driver

`shared/defects.c`, case **4**. Shape: bulk free, stale read on the error path.
