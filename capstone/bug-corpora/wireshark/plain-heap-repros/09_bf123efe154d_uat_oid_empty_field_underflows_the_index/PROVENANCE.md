# bf123efe154d — BER: Fix segmentation fault when configuring new OIDs

## The defect

With an empty field the validation loop does not execute and the code falls straight into `strptr[len-1]`. `len` is a `guint`, so `len - 1` wraps to `4294967295u` **before** the pointer arithmetic and is zero-extended — making this a read roughly 4 GiB past the buffer rather than one byte before it. The fix rejects `len == 0` up front.

## Upstream defect

- **Fix:** `bf123efe154d`, *"BER: Fix segmentation fault when configuring new OIDs"*, `epan/uat.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin carries the guard.

## The vulnerable code, quoted from the fix's parent

```c
    for(i = 0; i < len; i++)
      if(!(g_ascii_isdigit(strptr[i]) || strptr[i] == '.')) {
        *err = g_strdup("Only digits [0-9] and \".\" allowed in an OID");
        return FALSE;
      }

    if(strptr[len-1] == '.') {
```

## The fix

```c
    if (len == 0) {
      *err = g_strdup("Empty OID");
      return FALSE;
    }
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no UAT field and no OID. The buffer is a bare allocation, the length is zero as the trigger requires, and the probe touches the first byte OUTSIDE the allocation rather than the wrapped index -- a 4 GiB offset is a wild access, not a reportable crossing, and the true offset is recorded in `extent`.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
