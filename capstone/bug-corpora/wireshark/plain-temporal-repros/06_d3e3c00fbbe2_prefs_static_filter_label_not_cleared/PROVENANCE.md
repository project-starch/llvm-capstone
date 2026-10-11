# d3e3c00fbbe2 — prefs: fix crash when importing old filter expression preference

## The defect

`set_pref()` keeps the legacy filter label in a function-static across calls and frees it when it sees the matching expression — but never sets it back to NULL. A preferences file with two `gui.filter_expr` entries makes the second pass pass the freed string to `filter_expression_new()` and then free it again.

## Upstream defect

- **Fix:** `d3e3c00fbbe2`, *"prefs: fix crash when importing old filter expression preference"*, `epan/prefs.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin clears the static.

## The vulnerable code, quoted from the fix's parent

```c
    } else if (strcmp(pref_name, PRS_GUI_FILTER_EXPR) == 0) {
        filter_expr = g_strdup(value);
        /* Comments not supported for "old" preference style */
        filter_expression_new(filter_label, filter_expr, "", filter_enabled);
        g_free(filter_label);
```

## The fix

```c
        filter_expression_new(filter_label, value, "", filter_enabled);
        g_free(filter_label);
        filter_label = NULL;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no preferences file and no filter expressions. One allocation stands for the label, the static is a local that survives the free, and the next entry's use of it is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
