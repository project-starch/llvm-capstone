# 7dcf69480de8 — PEAK CAN TRC: Fix a double free

## The defect

`clean_trc_state()` unrefs the reader's regex and then frees the state struct itself. The not-mine path in `peak_trc_open()` called it and then called `g_free(trc_state)` on the same pointer. The path is reached whenever the probe decides the file is not a PEAK TRC capture.

## Upstream defect

- **Fix:** `7dcf69480de8`, *"PEAK CAN TRC: Fix a double free"*, `wiretap/peak-trc.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The caller's free is deleted at the pin.

## The vulnerable code, quoted from the fix's parent

```c
    if (open_val != WTAP_OPEN_MINE)
    {
        clean_trc_state(trc_state);
        g_free(trc_state);
        wth->priv = NULL;
        return open_val;
    }
```

## The fix

```c
    if (open_val != WTAP_OPEN_MINE)
    {
        clean_trc_state(trc_state);
        wth->priv = NULL;
        return open_val;
    }
```

## What is real here, and what is reduced

**Real:** which allocation's lifetime ends, the call that ends it, the pointer left holding the
freed address, and the access that follows. The object is a plain allocation because upstream's is
— this code is outside `epan`'s wmem scopes.

**Reduced:** no TRC file and no regex. The helper is reduced to the free it performs and the caller's second release to a read through the stale pointer.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — on the buggy arm the
stale pointer reaches storage that now belongs to a different live object, and under the fix it
does not. The arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions, and
the CheriBSD one is explicitly conditional on revocation sweep timing. Nor upstream reachability of
the specific sequence chosen here.
