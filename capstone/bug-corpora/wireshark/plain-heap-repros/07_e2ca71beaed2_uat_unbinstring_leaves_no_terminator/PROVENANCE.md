# e2ca71beaed2 — Allocate enough space to make proper NULL terminated string in uat_unesc (bug 2169) and uat_unbinstring,

## The defect

`uat_unbinstring` allocates `in_len/2` bytes — exactly one output byte per two input hex digits — and fills all of them, leaving no terminator, while UAT field handling consumes the result as a C string. The same commit fixes the identical shape in `uat_unesc`, whose common path also produces exactly one output byte per input character.

## Upstream defect

- **Fix:** `e2ca71beaed2`, *"Allocate enough space to make proper NULL terminated string in uat_unesc (bug 2169) and uat_unbinstring,"*, `epan/uat.c`.
- **CVE:** none assigned.
- **Live at our v4.6.8 pin: NO.** The pin allocates len + 1 and zeroes it.

## The vulnerable code, quoted from the fix's parent

```c
	guint len = in_len/2;
	int i = 0;
	...
	buf= g_malloc(len);
	*len_p = len;
```

## The fix

```c
	buf= g_malloc0(len+1);
	if (len_p) *len_p = len;
```

## What is real here, and what is reduced

**Real:** the arithmetic, which allocation is crossed, and the fix's own term. The buffer is a
plain allocation because upstream's is.

**Reduced:** no UAT fields and no hex input. The buffer holds non-zero decoded bytes as the decoder would produce, and the consumer's scan is the labelled probe.

## What the run establishes, and what it does not

**Establishes:** two-sided reproduction from the upstream fix differential — the buggy arm leaves
the allocation, the fixed arm does not, and the arms differ by exactly the fix's term.

**Does not establish** any Capstone or CheriBSD reading; those arms are declared predictions,
though `tools/size-class-audit.py` confirms the allocation leaves no size-class slack for the
crossing to hide in. Nor upstream reachability of the specific input chosen here.
