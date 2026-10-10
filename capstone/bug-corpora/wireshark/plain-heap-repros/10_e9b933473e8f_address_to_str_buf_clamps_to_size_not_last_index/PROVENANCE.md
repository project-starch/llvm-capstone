# `e9b933473e8f` — `epan/to_str.c`

## The defect

In the `AT_URI` arm, `copy_len` is clamped to `buf_len` rather than `buf_len - 1`. When `addr->len >= buf_len` the copy fills the whole buffer and the terminator is then stored at `buf[buf_len]` — one byte past the end. The `memmove` itself is in bounds; only the terminator crosses.

## Upstream defect

- **Fix:** `e9b933473e8f`, `epan/to_str.c`. **The commit SUBJECT names an individual, so this fix is cited by HASH AND PATH ONLY** — see `citation_constraint` in case.json. This tree's naming rule is absolute and applies to every committed file.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The function no longer exists at the pin.

## The vulnerable code, quoted from the fix's parent

```c
  case AT_URI: {
    int copy_len = addr->len < buf_len ? addr->len : buf_len;
    memmove(buf, addr->data, copy_len );
    buf[copy_len] = '\0';
    }
```

## The fix

```c
    int copy_len = addr->len < (buf_len - 1) ? addr->len : (buf_len - 1);
    memmove(buf, addr->data, copy_len );
    buf[copy_len] = '\0';
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no addresses and no URI data. The output buffer is a bare allocation, the source is longer than it, and the terminator store is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
