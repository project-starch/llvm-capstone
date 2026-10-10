# 012a179785ab — Fix a double free.

## The defect

`pf_dir_path_copy = pf_dir_path` is a plain assignment, so the variable named *copy* is an **alias**. `get_dirname()` truncates the buffer in place and the later `g_free(pf_dir_path_copy)` destroys the very buffer `pf_dir_path` still points at; the next statement passes that freed pointer to `ws_mkdir`, and on failure it is returned to the caller, which frees it again.

## Upstream defect

- **Fix:** `012a179785ab`, *"Fix a double free."*, `wsutil/filesystem.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin makes the copy real.

## The vulnerable code, quoted from the fix's parent

```c
        pf_dir_path_copy = pf_dir_path;
        pf_dir_parent_path = get_dirname(pf_dir_path_copy);
        ...
        g_free(pf_dir_path_copy);
        ret = ws_mkdir(pf_dir_path, 0755);
```

## The fix

```c
        pf_dir_path_copy = g_strdup(pf_dir_path);
        pf_dir_parent_path = get_dirname(pf_dir_path_copy);
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no profile directory and no filesystem calls. The two variables are upstream's, one allocation stands for the path, and `ws_mkdir`'s use of it is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
