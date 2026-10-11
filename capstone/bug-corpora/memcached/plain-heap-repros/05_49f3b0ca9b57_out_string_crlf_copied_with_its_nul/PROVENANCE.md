# 49f3b0ca9b57 — Another buffer overrun fix.

## The defect

`out_string` guards on `(len + 2) > c->wsize`, reserving room for a CR and an LF, and then copies the literal `"\r\n"` with length **3** — which also writes the literal's NUL. At the exact boundary `len + 2 == wsize` that third byte lands at index `wsize`, one past the buffer. The very next line, `c->wbytes = len + 2`, shows only two bytes were intended.

## Upstream defect

- **Fix:** `49f3b0ca9b57`, *"Another buffer overrun fix."*, `memcached.c`.
- **CVE:** none assigned.
- **Live at our 1.6.45 pin: NO.** The pin copies 2.

## The vulnerable code, quoted from the fix's parent

```c
    if ((len + 2) > c->wsize) { ... }
    memcpy(c->wbuf, str, len);
    memcpy(c->wbuf + len, "\r\n", 3);
    c->wbytes = len + 2;
```

## The fix

```c
    memcpy(c->wbuf + len, "\r\n", 2);
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is — neither the slab allocator nor `cache.c` is involved.

**Reduced:** no server, no connection, no protocol parse. The buffer is a bare allocation and the copy is a byte loop, so the crossing is attributable to the labelled probe rather than to libc's memcpy. `wsize` is 16 and `len` is 14, the exact boundary the guard admits.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific trigger chosen here.
