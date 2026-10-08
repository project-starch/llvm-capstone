# cfc15838bdec — Fix crash (double-free) on refreshing interfaces list

## The defect

The caller frees `global_capture_opts.ifaces_err_info` and passes its address in as the out-parameter, trusting the callee to overwrite it. When `sync_interface_list_open()` fails but extcap interfaces are found, the `g_list_length(if_list) == 0` guard is false and the function returns with `*err_str` untouched — leaving the caller's global pointing at memory it just freed, which the next refresh frees again. `append_extcap_interface_list`'s `err_str` parameter is marked `_U_` and never assigned, so nothing on that path can overwrite it.

## Upstream defect

- **Fix:** `cfc15838bdec`, *"Fix crash (double-free) on refreshing interfaces list"*, `capchild/capture_ifinfo.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin initialises the out-parameter unconditionally.

## The vulnerable code, quoted from the fix's parent

```c
    *err = 0;
    ...
        if ( g_list_length(if_list) == 0 ) {
            ...
            if (err_str) {
                *err_str = primary_msg;
            } else {
                g_free(primary_msg);
            }
            ...
        }
        return if_list;
```

## The fix

```c
    *err = 0;
    *err_str = NULL;
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no interface enumeration and no extcap. One allocation stands for the message, a single variable stands for the global, and the next refresh's free is reduced to a read through it.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
