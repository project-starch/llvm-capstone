# 4b15bf76a7f7 — Fix an off-by-one error.  Fixes bug 698, possibly others.

## The defect

The loop reserves only 15 bytes of tail room, but the worst case needs 16 — `"."` plus ten digits plus `".>>>"` plus the terminator. `bufp += g_snprintf(...)` advances by the **would-be** length, which `g_snprintf` returns even when it truncated, so the cursor can reach `buf + buf_len` and the final `*bufp = '\0'` writes one byte past the end.

## Upstream defect

- **Fix:** `4b15bf76a7f7`, *"Fix an off-by-one error.  Fixes bug 698, possibly others."*, `epan/to_str.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The function no longer exists at the pin.

## The vulnerable code, quoted from the fix's parent

```c
    if ((bufp - buf) > (buf_len - 15)) { 
      bufp += g_snprintf(bufp, buf_len-(bufp-buf), ".>>>");
      break;
    }
  ...
  *bufp = '\0';
```

## The fix

```c
#define OID_STR_LIMIT (1 + 10 + 4 + 1) /* "." + 10 digits + ".>>>" + '\0' */
...
    if ((bufp - buf) > (buf_len - OID_STR_LIMIT)) {
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no OIDs and no formatting. The buffer is a bare allocation, the cursor arithmetic is upstream's reservation and g_snprintf's would-be-length advance, and the final terminator is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
